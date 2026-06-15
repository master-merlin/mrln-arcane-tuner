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

from .driver import Wan22Driver
from .expert_router import ExpertRouter
from .loader import Wan22Loader
from .saver import Wan22Saver

logger = structlog.get_logger(__name__)


class Wan22Trainer(GenericTrainingPipeline):
    """WAN 2.2 (T2V-A14B / I2V-A14B) dual-expert LoRA trainer.

    ``is_video_family`` is inherited from :class:`PipelineBaseMixin` (derived
    from the model's ``is_video`` capability) — no per-trainer flag needed.
    """

    # ── Setup ────────────────────────────────────────────────────────────

    def _setup_family(self) -> None:
        self.expert_mode = str(self.config.get("expert_mode", "both") or "both").lower()
        self.driver = Wan22Driver(self.definition, self.device)
        self.driver.configure_expert_mode(self.expert_mode)
        self.loader = Wan22Loader(self.device, expert_mode=self.expert_mode)
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

    # ── Dual LoRA injection ──────────────────────────────────────────────

    def _apply_peft(self) -> None:
        """Inject LoRA into BOTH experts.

        The base method wraps the primary (currently-active) expert and routes
        it through ``_update_primary_model`` (which updates ``transformer_high``
        when high is active). We then wrap the OTHER expert directly and store it
        back on the driver so both transformers carry adapters.
        """
        from peft import LoraConfig, get_peft_model

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

    # ── Set BOTH experts to train mode + place on devices ────────────────

    def _configure_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on BOTH experts (base does primary)."""
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

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, batch: dict | None = None
    ) -> Any:
        """Encode captions through UMT5-XXL with in-memory caching."""
        if not self.config.get("cache_text_embeddings", True):
            out = self.driver.encode_text(captions, dtype)
            return out.embeddings if hasattr(out, "embeddings") else out
        return self._get_cached_text_embeddings(captions, dtype)

    def _get_cached_text_embeddings(
        self, captions: list[str], dtype: torch.dtype
    ) -> torch.Tensor:
        results: list[torch.Tensor | None] = []
        uncached: list[tuple[int, str]] = []

        for i, cap in enumerate(captions):
            if cap in self.text_cache:
                results.append(self.text_cache[cap])
            else:
                uncached.append((i, cap))
                results.append(None)

        if uncached and self.text_encoder is not None:
            for orig_idx, cap in uncached:
                out = self.driver.encode_text([cap], dtype)
                emb = out.embeddings if hasattr(out, "embeddings") else out
                self.text_cache[cap] = emb.cpu()
                results[orig_idx] = emb.cpu()
        elif uncached:
            raise RuntimeError(
                "Text encoder unavailable for uncached caption(s): "
                + ", ".join(cap[:50] for _, cap in uncached)
            )

        return torch.cat(
            [r.to(self.device, dtype=dtype) for r in results if r is not None], dim=0
        )

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
