"""WAN 2.2 Trainer — dual-expert LoRA training in a single run.

Extends :class:`GenericTrainingPipeline` to train BOTH WAN 2.2 experts together:

- **Dual LoRA injection** — the generic ``_apply_peft`` wraps only the primary
  (active) expert. This trainer wraps the OTHER expert too, so BOTH transformers
  carry LoRA adapters.
- **Dual optimizer param groups** — the generic ``_configure_optimization``
  collects params from the primary model only; we extend it to also include the
  second expert's trainable (LoRA) params, so a single optimizer updates both.
- **Router wiring** — an :class:`ExpertRouter` is constructed from the config
  (boundary + ``switch_interval`` + timestep cfg) and attached to the driver,
  which switches the active expert per optimizer step via ``on_optimizer_step``.
- **Dual saver hand-off** — ``_build_trainable_components`` exposes both PEFT
  models as ``unet_high`` / ``unet_low`` so :class:`Wan22Saver` writes two files.

GPU end-to-end training (real dual-transformer forward/backward, real expert
swap VRAM, ComfyUI load-back of both files) is a follow-up that needs real WAN
2.2 weights + a CUDA device. The unit tests exercise the router math, the
dual-adapter wiring, the swap state transitions, and the precision contracts
against the REAL driver/sampler with fakes.
"""

from __future__ import annotations

from typing import Any

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline
from app.engine.models.families.wan_shared.trainer_base import WanTextCacheMixin

from .driver import Wan22Driver
from .expert_router import ExpertRouter
from .loader import Wan22Loader
from .saver import Wan22Saver

logger = structlog.get_logger(__name__)


class Wan22Trainer(WanTextCacheMixin, GenericTrainingPipeline):
    """WAN 2.2 (T2V-A14B / I2V-A14B) dual-expert LoRA trainer.

    ``is_video_family`` is inherited from :class:`PipelineBaseMixin` (derived
    from the model's ``is_video`` capability) — no per-trainer flag needed.
    """

    # ── Setup ────────────────────────────────────────────────────────────

    def _setup_family(self) -> None:
        self.expert_mode = str(self.config.get("expert_mode", "both") or "both").lower()
        self.driver = Wan22Driver(self.definition, self.device)
        self.driver.configure_expert_mode(self.expert_mode)
        # Dual-expert runs DEFER the low expert out of Phase A: both ~28 GB
        # experts must never sit on CPU together through the TE/VAE caching
        # stretch (that ~67 GB host-RAM peak hangs a 64 GB box). The deferred
        # expert is materialised in _load_deferred_experts() once the high
        # expert has moved to the GPU. Single-expert runs load exactly one
        # transformer, so there is nothing to defer.
        defer = self.expert_mode == "both"
        self.loader = Wan22Loader(
            self.device, expert_mode=self.expert_mode, defer_second_expert=defer
        )
        self.saver = Wan22Saver(mode=self.driver.mode)
        self._build_router()

    def _build_router(self) -> ExpertRouter:
        """Construct + attach the ExpertRouter from config + driver boundary.

        Single-expert runs (``expert_mode`` high/low) build a PINNED router — it
        always routes to that one expert and truncates its timesteps to the
        expert's boundary range — and force ``resident`` placement (only one
        transformer is loaded, so there is nothing to swap).
        """
        switch_interval = int(self.config.get("expert_switch_interval", 1))
        seed = int(self.config.get("seed", 0) or 0)
        mode = getattr(self, "expert_mode", "both")
        pinned = None if mode == "both" else mode
        router = ExpertRouter(
            boundary=self.driver.boundary,
            switch_interval=switch_interval,
            timestep_cfg=self.config,
            seed=seed,
            pinned_expert=pinned,
        )
        self.expert_router = router
        self.driver.set_router(router)
        swap = (
            "resident" if pinned else self.config.get("expert_swap_mode", "resident")
        )
        self.driver.configure_swap_mode(swap)
        return router

    # ── Convention delegation (load-bearing — dead-dispatch guard) ────────

    def sample_timesteps(
        self, batch_size: int, latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Delegate to the driver's expert-aware, router-truncated sampler.

        WAN 2.2 is a Mixture-of-Experts: the high-noise expert must only ever
        see timesteps ``>= boundary`` and the low-noise expert only ``< boundary``
        (:class:`ExpertRouter`). ``Wan22Driver.sample_timesteps`` draws from the
        configured distribution truncated to the *active* expert's range.

        The base ``PipelineBaseMixin.sample_timesteps`` is the FULL-range
        flow-match sampler with no knowledge of the expert boundary or the
        router — using it un-overridden (as the real ``pipeline_train`` loop does
        via ``self.sample_timesteps``) would train every expert across the whole
        ``[0, 1000]`` range, defeating the MoE split (the router truncation would
        be dead code on the real path). This override wires the driver's sampler
        to the real training loop — mirrors ``boogu_image``'s convention
        delegation (``families/boogu_image/trainer.py``). Pinned check:
        ``test_wan22_sample_timesteps_wiring.py``.
        """
        return self.driver.sample_timesteps(
            batch_size, self.device, self.config, latents=latents,
        )

    # ── Dual LoRA injection ──────────────────────────────────────────────

    def _apply_peft(self) -> None:
        """Inject LoRA into BOTH experts.

        The base method wraps the primary (currently-active) expert and routes
        it through ``_update_primary_model`` (which updates ``transformer_high``
        when high is active). We then wrap the OTHER expert directly and store it
        back on the driver so both transformers carry adapters.
        """
        from peft import LoraConfig, get_peft_model

        # Safety net: guarantee the deferred low expert is present before we wrap
        # it, independent of the grad-checkpointing hook order (idempotent).
        self._load_deferred_experts()

        # 1. Base PEFT on the active/primary expert (high by default).
        super()._apply_peft()

        # Single-expert: only ONE transformer is loaded — nothing more to wrap,
        # collect, checkpoint, or save. The base wrapped the resident expert;
        # the other slot is intentionally None (see assign_components).
        if getattr(self, "expert_mode", "both") != "both":
            self.logger.info("wan22_single_expert_peft", expert=self.expert_mode)
            return

        # 2. Wrap the inactive expert with the SAME LoRA config.
        driver: Wan22Driver = self.driver  # type: ignore[assignment]
        active = driver.active_expert
        other_model = (
            driver.transformer_low if active == "high" else driver.transformer_high
        )
        if other_model is None:
            self.logger.warning("wan22_second_expert_missing", active=active)
            return

        rank = int(self.config.get("network_rank", 16))
        alpha = float(self.config.get("network_alpha", rank))
        targets = self.get_lora_targets()
        exclude = self.get_lora_exclude_modules()
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=targets,
            exclude_modules=exclude,
            lora_dropout=0.0,
            bias="none",
        )
        wrapped = get_peft_model(other_model, lora_config)

        if active == "high":
            driver.transformer_low = wrapped
        else:
            driver.transformer_high = wrapped
            # If low was active, the base wrapped it via _update_primary_model;
            # keep the active pointer consistent after we re-wrap the other one.
        # Re-point the primary at the (already-wrapped) active expert.
        driver._set_active(active)

        trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
        self.logger.info(
            "wan22_dual_lora_applied",
            second_expert=("low" if active == "high" else "high"),
            second_trainable=trainable,
        )

    # ── Dual optimizer param groups ──────────────────────────────────────

    def _collect_expert_params(self) -> list[torch.nn.Parameter]:
        """All trainable LoRA params across BOTH experts (deduplicated)."""
        driver: Wan22Driver = self.driver  # type: ignore[assignment]
        seen: set[int] = set()
        params: list[torch.nn.Parameter] = []
        for model in (driver.transformer_high, driver.transformer_low):
            if model is None:
                continue
            for p in model.parameters():
                if p.requires_grad and id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
        return params

    def _configure_optimization(self, max_train_steps: int) -> None:
        """Configure optimizer over BOTH experts' params, then base scheduler.

        We temporarily monkey-patch ``_get_primary_model().parameters`` so the
        base ``_configure_optimization`` (which collects from the primary model)
        sees the union of both experts' trainable params. This reuses ALL the
        base LR-scaling / optimizer-factory / scheduler / scaler logic verbatim.
        """
        primary = self._get_primary_model()
        all_params = self._collect_expert_params()
        original_parameters = primary.parameters

        def _both_experts_parameters(*args: Any, **kwargs: Any):
            return iter(all_params)

        primary.parameters = _both_experts_parameters  # type: ignore[method-assign]
        try:
            super()._configure_optimization(max_train_steps)
        finally:
            primary.parameters = original_parameters  # type: ignore[method-assign]

        self.logger.info(
            "wan22_optimizer_configured_dual",
            total_trainable=len(all_params),
        )

    # ── Deferred low-noise expert (host-RAM sequencing) ──────────────────

    def _load_deferred_experts(self) -> None:
        """Materialise the deferred low-noise expert onto CPU (dual-expert runs).

        WAN 2.2 A14B holds TWO ~28 GB experts. To keep peak host RAM at ONE
        expert, the loader leaves the low-noise expert out of Phase A; this
        method loads it back on demand.

        Call site: the TOP of :meth:`_configure_gradient_checkpointing` — the
        earliest Phase-B hook run AFTER ``prepare_for_training`` has moved the
        high expert to the GPU (``_move_component_to_gpu("unet")``) and BEFORE
        anything touches the second expert (grad-checkpointing here, then
        ``_apply_peft`` / optimizer). Sequencing so the high expert is on the GPU
        before the low expert is materialised means host RAM never holds both at
        once; from here the flow is byte-identical to eager loading (the low
        expert is CPU-resident exactly as it would have been).

        Idempotent, and a no-op unless the loader actually deferred an expert
        (so fake-wired unit trainers, single-expert runs, and resumes are
        unaffected).
        """
        if getattr(self, "_deferred_expert_loaded", False):
            return
        # Latch first: a genuine no-op path (no loader / not deferred / already
        # present) should not be retried on every hook call.
        self._deferred_expert_loaded = True

        loader = getattr(self, "loader", None)
        driver: Wan22Driver = self.driver  # type: ignore[assignment]
        if (
            loader is None
            or not getattr(loader, "defer_second_expert", False)
            or getattr(self, "expert_mode", "both") != "both"
            or driver.transformer_low is not None
        ):
            return

        dtype = driver.resolve_loading_dtype()
        low = loader.load_second_expert(
            self.definition, torch_dtype=dtype, initial_device="cpu"
        )
        self.components["unet_low"] = low
        driver.transformer_low = low
        self.logger.info("wan22_deferred_low_expert_materialized", device="cpu")

    # ── Set BOTH experts to train mode + place on devices ────────────────

    def _configure_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on BOTH experts (base does primary)."""
        # Bring the deferred low expert back BEFORE any per-expert work below —
        # grad-checkpointing, then PEFT/optimizer, all expect both present.
        self._load_deferred_experts()
        super()._configure_gradient_checkpointing()
        if not self.config.get("gradient_checkpointing", False):
            return
        driver: Wan22Driver = self.driver  # type: ignore[assignment]
        active = driver.get_primary_model()
        other = (
            driver.transformer_low
            if driver.active_expert == "high"
            else driver.transformer_high
        )
        if other is not None and other is not active:
            if hasattr(other, "gradient_checkpointing_enable"):
                other.gradient_checkpointing_enable()
            elif hasattr(other, "enable_gradient_checkpointing"):
                other.enable_gradient_checkpointing()

    # ── Dual saver hand-off ──────────────────────────────────────────────

    def _build_trainable_components(self) -> dict[str, Any]:
        """Expose BOTH PEFT experts to the saver as ``unet_high``/``unet_low``.

        Keeps ``unet`` = the active expert for any generic consumer (resume
        state, training-log param counts), and adds the explicit high/low keys
        the :class:`Wan22Saver` writes two files from. Also threads the router
        state for resume.
        """
        comps = super()._build_trainable_components()
        driver: Wan22Driver = self.driver  # type: ignore[assignment]
        comps["unet_high"] = driver.transformer_high
        comps["unet_low"] = driver.transformer_low
        router = getattr(self, "expert_router", None)
        if router is not None:
            comps["router_state"] = router.state_dict()
        return comps

    # ── Text Encoding (UMT5-XXL via driver, with lazy cache) ─────────────
    # encode_text / _get_cached_text_embeddings live in WanTextCacheMixin
    # (byte-identical between wan21 and wan22; hoisted to wan_shared).

    # ── Sampler ──────────────────────────────────────────────────────────

    def _create_sampler(self):
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import Wan22Sampler

            return Wan22Sampler(self)
        return None

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep transformer references in sync after PEFT/quant wrapping.

        The base loop wraps/quantizes the ACTIVE expert; mirror the new model
        onto both the trainer aliases and the driver's active-expert slot.
        """
        self.components["unet"] = new_model
        self.transformer = new_model
        driver: Wan22Driver = self.driver  # type: ignore[assignment]
        if driver.active_expert == "high":
            driver.transformer_high = new_model
        else:
            driver.transformer_low = new_model
        driver.transformer = new_model
