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
from app.engine.models.families.wan_shared.trainer_base import (
    DualExpertDeferredLoadMixin,
    WanTextCacheMixin,
)

from .driver import Wan22Driver
from .expert_router import ExpertRouter
from .loader import Wan22Loader

logger = structlog.get_logger(__name__)


class Wan22Trainer(
    WanTextCacheMixin, DualExpertDeferredLoadMixin, GenericTrainingPipeline
):
    """WAN 2.2 (T2V-A14B / I2V-A14B) dual-expert LoRA trainer.

    ``is_video_family`` is inherited from :class:`PipelineBaseMixin` (derived
    from the model's ``is_video`` capability) — no per-trainer flag needed.
    """

    DEFERRED_EXPERT_LOG_EVENT = "wan22_deferred_low_expert_materialized"

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

    # ── Dual-expert EMA (W3.T10) ──────────────────────────────────────────

    def _ema_parameters(self) -> dict[str, torch.nn.Parameter] | None:
        """EMA-shadow BOTH experts' trainable params on a ``both``-mode run.

        Without this override, ``_configure_ema`` binds ``EMAHandler`` to
        ``_get_primary_model()`` — the single ACTIVE expert — so only that
        expert's LoRA gets an EMA shadow; the other expert's saved file is
        raw (un-EMA'd) weights. Names are prefixed ``high.``/``low.`` so the
        two experts' identically-named parameters (e.g. both have a
        ``blocks.0.attn1.to_q.lora_A.weight``) don't collide in one shadow
        dict.

        Single-expert runs (``expert_mode`` high/low) fall back to ``None``
        (base primary-model behavior) — there is only one transformer to
        shadow, byte-identical to the un-overridden path.
        """
        driver: Wan22Driver = self.driver  # type: ignore[assignment]
        if getattr(self, "expert_mode", "both") != "both":
            return None
        out: dict[str, torch.nn.Parameter] = {}
        for prefix, model in (
            ("high", driver.transformer_high),
            ("low", driver.transformer_low),
        ):
            if model is None:
                continue
            for name, param in model.named_parameters():
                if param.requires_grad:
                    out[f"{prefix}.{name}"] = param
        return out

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

        # Explicit start placement (Task W3.T2): both experts now carry PEFT
        # adapters and the optimizer holds both experts' params — place them
        # on their configured devices BEFORE the training loop starts. This
        # used to be a pure side effect of the step-0 baseline SAMPLER's
        # device-ensure loop, which is skipped whenever sampling is disabled
        # (sample_every_n_steps=0), declined (sample_before_training=False),
        # or raises (swallowed at the call site) — any of which left the
        # deferred low expert CPU-resident until the first router flip hit it
        # mid-forward (wave-3 audit 2026-07-26). Single-expert runs are
        # unaffected: place_experts_for_start() no-ops on the missing expert.
        #
        # Block-swapping is a SEPARATE hazard, fixed in the same wave-3 review
        # round: ``_configure_block_swapping()`` (pipeline_optimization.py,
        # step 6b) runs BEFORE this method and may have handed the active
        # expert's deep blocks to a ``BlockSwappingManager``, which owns their
        # CPU<->GPU placement via forward hooks — a bulk ``.to(device)`` on
        # that expert would force every swapped block onto GPU at once,
        # defeating the swap. The driver can't see ``self._block_swap_managers``
        # (it lives here, on the trainer/pipeline), so we hand off explicitly:
        # ``block_swap_active_expert`` tells ``place_experts_for_start()`` (and
        # the ``_set_active`` guard) which expert, if any, to leave alone.
        driver: Wan22Driver = self.driver  # type: ignore[assignment]
        if getattr(self, "_block_swap_managers", None):
            driver.block_swap_active_expert = driver.active_expert
        driver.place_experts_for_start()

        self.logger.info(
            "wan22_optimizer_configured_dual",
            total_trainable=len(all_params),
        )

    # ── Deferred low-noise expert (host-RAM sequencing) ──────────────────
    # _load_deferred_experts lives in DualExpertDeferredLoadMixin (shared with
    # bernini_r's 14B). Call sites here: the top of _apply_peft and
    # _configure_gradient_checkpointing — the earliest Phase-B hooks run AFTER
    # prepare_for_training has moved the high expert to the GPU and BEFORE
    # anything touches the second expert.

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
