"""Bernini-R trainer — SD3-mode timesteps + target-token flow-match loss.

Bernini-R is a renderer-only video-EDIT DiT built from stock Wan components, so
this trainer mirrors :class:`Wan21Trainer`'s base-class shape exactly
(``(WanTextCacheMixin, GenericTrainingPipeline)``): UMT5-XXL text encoding with
the shared lazy + disk cache, a frozen text encoder, plain captions,
``max_sequence_length`` 512, no chat template. All the generic training mechanics
(optimizer, EMA, grad-accum, checkpointing, logging, sampling wiring) come from
:class:`GenericTrainingPipeline`.

The Bernini-specific pieces:

- **Timestep sampling** (:meth:`sample_timesteps`) — the upstream renderer's
  ``NoiseScheduler`` for the video tasks: SD3 ``mode`` weighting followed by the
  per-task shift-warp, in the RAW ``[0, 1000]`` space (the wan flow-match
  ``add_noise`` lerp does the ``/1000``). This OVERRIDES the base flow-match
  sampler. The formula is transcribed verbatim from upstream ``bernini/training/
  data.py`` ``NoiseScheduler`` (see ``.agent/workdir/bernini-r-recon.md`` §4).

- **Loss** — plain flow-match velocity ``noise - x0`` on the TARGET tokens only.
  No trainer override needed (see the driver docstring).

- **Dual-expert range-split training (14B MoE ONLY)** — see below. The 1.3B
  single-expert path never enters any of the dual code and is byte-identical.

────────────────────────────────────────────────────────────────────────────────
DUAL-EXPERT RANGE-SPLIT — THE DELIBERATE DIVERGENCE FROM wan22
────────────────────────────────────────────────────────────────────────────────
The 14B is a Mixture-of-Experts: a HIGH-noise expert (``transformer``) serves
``t >= boundary·1000`` and a LOW-noise expert (``transformer_2``) serves
``t < boundary`` (boundary = ``switch_dit_boundary`` = 0.875; recon §3). Both are
LoRA-trained in one run (two files), the active expert switched per optimizer
step by a REUSED :class:`wan22.ExpertRouter`, and the packed forward + optimizer
+ grad-checkpointing follow :class:`Wan22Trainer` verbatim (shared machinery).

The ONE divergence — the reason this trainer overrides ``sample_timesteps``
rather than delegating to the router's ``sample_timesteps_for``:

  * **wan22** samples each expert's band from the GENERIC configured distribution
    (``TimestepSampler.sample_scaled(mode, …)``) truncated to the band.

  * **bernini** must RANGE-SPLIT using BERNINI's OWN SD3-``mode``+per-task-shift
    formula (recon §4), redraw-clamped (rejection) into the active expert's band,
    so the high expert NEVER sees ``t < boundary`` and the low expert NEVER sees
    ``t >= boundary`` — mirroring upstream's SEPARATE ``bernini_renderer_high.yaml``
    (``noise_tmin 0.875``/``noise_tmax 1.0``) and ``bernini_renderer_low.yaml``
    (``0.0``/``0.875``) dedicated per-expert runs. Using the router's generic
    ``sample_timesteps_for`` would train each band off BERNINI's distribution (the
    flowmatch-timestep gotcha class), so the band sampler is bernini-specific.

STEP SELECTION: for a one-run ``both`` dual-LoRA the reused wan22
:class:`ExpertRouter` still chooses which expert each optimizer step updates
(shared machinery — its ``p_high``-weighted Bernoulli picks how OFTEN each expert
trains); bernini only substitutes the per-step BAND timesteps. The single-expert
modes (``expert_mode`` high / low) are the faithful reproduction of upstream's two
dedicated ``bernini_renderer_{high,low}.yaml`` runs — a PINNED router trains that
one expert exclusively on its band.

REDRAW vs RESCALE: recon §4 is explicit — timesteps outside the active band are
REDRAWN (upstream's ``while True`` rejection loop clamped to
``[noise_tmin, noise_tmax]``), NOT rescaled. Rejection preserves the mode+shift
weighting SHAPE within the band exactly (a rescale would distort it), which is
what the band-sampling distribution test pins.
"""

from __future__ import annotations

import math

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline
from app.engine.models.families.wan22.expert_router import HIGH, LOW, ExpertRouter
from app.engine.models.families.wan_shared.trainer_base import (
    DualExpertDeferredLoadMixin,
    WanTextCacheMixin,
)

from .driver import BerniniRDriver
from .loader import BerniniRLoader
from .saver import BerniniRDualSaver, BerniniRSaver

logger = structlog.get_logger(__name__)

# Hard cap on band redraw rounds so a pathological distribution (almost no mass
# on one side of the boundary) can't spin forever; on exhaustion the remaining
# slots are clamped strictly into the band (negligible bias vs. hanging).
_MAX_BAND_REDRAW_ROUNDS = 64


class BerniniRTrainer(
    WanTextCacheMixin, DualExpertDeferredLoadMixin, GenericTrainingPipeline
):
    """Bernini-R (renderer-only video edit) LoRA trainer — 1.3B single / 14B MoE.

    ``is_video_family`` is inherited from :class:`PipelineBaseMixin` (derived from
    the model's ``is_video`` capability) — no per-trainer flag needed.
    """

    DEFERRED_EXPERT_LOG_EVENT = "bernini_r_deferred_low_expert_materialized"

    # ── Timestep-sampling constants (upstream ``bernini/training/data.py``) ──
    # SD3 mode weighting scale. Recon §4 leaves ``mode_scale`` symbolic in the
    # quoted formula; the SD3 convention (and the BR3 brief) pin it at 1.29.
    DEFAULT_MODE_SCALE: float = 1.29
    # Per-task shift-warp. v2v = 5.0 (upstream ``shift_config``); the definition
    # (BR4) supplies the effective value via config; this is the family default.
    DEFAULT_TIMESTEP_SHIFT: float = 5.0

    # ── Setup ────────────────────────────────────────────────────────────
    def _setup_family(self) -> None:
        self.driver = BerniniRDriver(self.definition, self.device)
        if getattr(self.driver, "is_dual", False):
            # 14B MoE — mirror Wan22Trainer's dual setup.
            self.expert_mode = str(
                self.config.get("expert_mode", "both") or "both"
            ).lower()
            self.driver.configure_expert_mode(self.expert_mode)
            # Dual-expert runs DEFER the low expert out of Phase A (mirrors
            # Wan22Trainer): both ~28 GB experts must never sit on CPU together
            # through the TE/VAE caching stretch. The deferred expert is
            # materialised by DualExpertDeferredLoadMixin once the high expert
            # has moved to the GPU. Single-expert runs have nothing to defer.
            defer = self.expert_mode == "both"
            self.loader = BerniniRLoader(
                self.device, expert_mode=self.expert_mode, defer_second_expert=defer
            )
            self.saver = BerniniRDualSaver(mode=self.driver.mode)
            self._build_router()
        else:
            # 1.3B single expert — byte-identical to the v1 path.
            self.loader = BerniniRLoader(self.device)
            self.saver = BerniniRSaver(mode=self.driver.mode)

    def _build_router(self) -> ExpertRouter:
        """Construct + attach the ExpertRouter from config + driver boundary.

        Single-expert runs (``expert_mode`` high/low) build a PINNED router — it
        always routes to that one expert and truncates its timesteps to the
        expert's boundary range — and force ``resident`` placement (only one
        transformer is loaded, so there is nothing to swap). NOTE: bernini uses
        the router for STEP SELECTION + swap/state only; the actual per-step
        timesteps come from :meth:`sample_timesteps` (range-split band sampling).

        Task W3.T3: the reused :class:`ExpertRouter`'s ``p_high`` Monte-Carlo
        estimate defaults to the GENERIC ``TimestepSampler`` formula for
        ``config["timestep_sampling"]`` — but bernini's ACTUAL per-step
        timesteps (:meth:`sample_timesteps` / :meth:`_sample_band`) come from
        ITS OWN SD3-``mode`` + per-task shift-warp transform
        (:meth:`_mode_shift_timesteps`), which the generic ``"mode"`` formula
        does not include (no shift warp at all). Under the 14B v2v defaults
        (``mode_scale=1.29``, ``shift=5.0``) that mismatch estimates
        ``p_high ≈ 0.16`` against the real ``≈ 0.29`` — the high expert was
        step-selected for a step-frequency far below the timestep-mass its
        band actually carries, under-training it roughly 2×. Passing
        ``timestep_draw`` fixes the estimate to the REAL distribution; it is
        harmless (never invoked) for a pinned single-expert router, which
        skips the Monte-Carlo estimate entirely.
        """
        switch_interval = int(self.config.get("expert_switch_interval", 1))
        seed = int(self.config.get("seed", 0) or 0)
        mode = getattr(self, "expert_mode", "both")
        pinned = None if mode == "both" else mode
        mode_scale, shift = self._timestep_params()
        router = ExpertRouter(
            boundary=self.driver.boundary,
            switch_interval=switch_interval,
            timestep_cfg=self.config,
            seed=seed,
            pinned_expert=pinned,
            timestep_draw=lambda n: self._mode_shift_timesteps(
                torch.rand(n), mode_scale, shift
            ),
        )
        self.expert_router = router
        self.driver.set_router(router)
        swap = "resident" if pinned else self.config.get("expert_swap_mode", "resident")
        self.driver.configure_swap_mode(swap)
        return router

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep transformer references in sync after PEFT/quant wrapping.

        The base method updates ``self.components['unet']`` / ``self.model`` but
        does NOT reach into ``self.driver.transformer``; without this the packed
        forward would run the un-wrapped (non-LoRA'd) transformer (mirrors
        :class:`Wan21Trainer`). On the 14B the base loop wraps the ACTIVE expert,
        so the new model is also mirrored onto that expert's slot (mirrors
        :class:`Wan22Trainer`).
        """
        self.transformer = new_model
        self.components["unet"] = new_model
        driver: BerniniRDriver = self.driver  # type: ignore[assignment]
        driver.transformer = new_model
        if getattr(driver, "is_dual", False):
            if driver.active_expert == HIGH:
                driver.transformer_high = new_model
            else:
                driver.transformer_low = new_model

    def _create_sampler(self):
        """Bernini-R v2v in-training preview sampler (Task BR4).

        Created only when sampling is configured (``sample_every_n_steps > 0``),
        mirroring wan21/wan22. Lazily imported so family discovery / the BR2/BR3
        tests never require the sampler module.
        """
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import BerniniRSampler

            return BerniniRSampler(self)
        return None

    # ── Timestep sampling (upstream NoiseScheduler, video tasks) ─────────
    @staticmethod
    def _mode_shift_timesteps(
        raw: torch.Tensor, mode_scale: float, shift: float
    ) -> torch.Tensor:
        """The upstream ``NoiseScheduler`` transform → RAW ``[0, 1000]``::

        u      = 1 - raw - mode_scale * (cos(pi * raw / 2) ** 2 - 1 + raw)
        sigmas = shift * u / (1 + (shift - 1) * u)
        ts     = sigmas * 1000
        """
        u = 1.0 - raw - mode_scale * (torch.cos(math.pi * raw / 2.0) ** 2 - 1.0 + raw)
        sigmas = shift * u / (1.0 + (shift - 1.0) * u)
        return sigmas * 1000.0

    def _timestep_params(self) -> tuple[float, float]:
        mode_scale = float(self.config.get("mode_scale", self.DEFAULT_MODE_SCALE))
        shift = float(self.config.get("timestep_shift", self.DEFAULT_TIMESTEP_SHIFT))
        return mode_scale, shift

    def sample_timesteps(
        self, batch_size: int, latents: torch.Tensor | None = None
    ) -> torch.Tensor:
        """SD3 ``mode`` weighting → per-task shift-warp → RAW ``[0, 1000]``.

        Single expert (1.3B): the FULL-range upstream transform (byte-identical
        to v1). Dual expert (14B): RANGE-SPLIT — restricted to the ACTIVE expert's
        serving band by redraw (see :meth:`_sample_band` and the module
        docstring's divergence note). ``shift`` (v2v default 5.0) and
        ``mode_scale`` (1.29) are read from config so the definition / user can
        select the per-task shift; the family defaults reproduce the v2v recipe.
        The result is RAW ``[0, 1000]`` — the wan flow-match ``add_noise`` lerp
        applies the ``/1000`` (the pure-noise gotcha: the frozen time embedder
        must see the un-scaled value).
        """
        driver = getattr(self, "driver", None)
        if (
            driver is not None
            and getattr(driver, "is_dual", False)
            and getattr(driver, "router", None) is not None
        ):
            return self._sample_band(driver.active_expert, batch_size)

        mode_scale, shift = self._timestep_params()
        raw = torch.rand(batch_size, device=self.device)
        return self._mode_shift_timesteps(raw, mode_scale, shift)

    def _expert_band(self, expert: str) -> tuple[float, float]:
        """Raw ``[0, 1000]`` serving band for ``expert`` (inclusive-lo).

        HIGH → ``[boundary·1000, 1000]`` (t >= boundary); LOW → ``[0,
        boundary·1000)`` (t < boundary). Matches
        :meth:`BerniniRDriver.expert_for_timestep`.
        """
        b = float(getattr(self.driver, "boundary_timestep", 875.0))
        if expert == HIGH:
            return b, 1000.0
        return 0.0, b

    def _sample_band(self, expert: str, batch_size: int) -> torch.Tensor:
        """Redraw-clamped SD3-mode timesteps in ``expert``'s serving band.

        Draws from BERNINI's ``_mode_shift_timesteps`` distribution and keeps only
        samples inside the band, redrawing the rest (rejection) — exactly the
        upstream ``while True`` clamp to ``[noise_tmin, noise_tmax]`` (recon §4).
        Rejection (not rescale) preserves the mode+shift weighting shape within
        the band. On redraw exhaustion the remaining slots clamp into the band.
        """
        mode_scale, shift = self._timestep_params()
        lo, hi = self._expert_band(expert)
        out = torch.empty(batch_size, device=self.device)
        filled = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        def _in_band(t: torch.Tensor) -> torch.Tensor:
            # HIGH band is inclusive of the boundary (t >= lo); LOW band is
            # [0, boundary) (t < hi). Both stay within [0, 1000].
            if expert == HIGH:
                return t >= lo
            return t < hi

        rounds = 0
        while not bool(filled.all()) and rounds < _MAX_BAND_REDRAW_ROUNDS:
            need = int((~filled).sum().item())
            cand = self._mode_shift_timesteps(
                torch.rand(need, device=self.device), mode_scale, shift
            )
            ok = _in_band(cand)
            if bool(ok.any()):
                idx_unfilled = torch.nonzero(~filled, as_tuple=False).flatten()
                accept_pos = idx_unfilled[ok]
                out[accept_pos] = cand[ok]
                filled[accept_pos] = True
            rounds += 1

        if not bool(filled.all()):
            # Pathological distribution: clamp the remaining slots strictly into
            # the band rather than hang (logged so it's visible).
            self.logger.warning(
                "bernini_r_band_redraw_exhausted",
                expert=expert,
                rounds=rounds,
                unfilled=int((~filled).sum().item()),
            )
            eps = 1e-3 * 1000.0
            out[~filled] = lo if expert == HIGH else max(hi - eps, 0.0)
        return out

    # ── Dual LoRA injection (14B; mirrors Wan22Trainer) ──────────────────
    def _apply_peft(self) -> None:
        """Inject LoRA into the active expert, and (14B ``both``) the other too."""
        from peft import LoraConfig, get_peft_model

        # Safety net: guarantee the deferred low expert is present before we
        # wrap it, independent of the grad-checkpointing hook order (idempotent;
        # mirrors Wan22Trainer).
        self._load_deferred_experts()

        # 1. Base PEFT on the active/primary expert (high by default on 14B).
        super()._apply_peft()

        driver: BerniniRDriver = self.driver  # type: ignore[assignment]
        if not getattr(driver, "is_dual", False):
            return  # 1.3B single expert — nothing more to wrap.
        if getattr(self, "expert_mode", "both") != "both":
            self.logger.info("bernini_r_single_expert_peft", expert=self.expert_mode)
            return

        # 2. Wrap the inactive expert with the SAME LoRA config.
        active = driver.active_expert
        other_model = (
            driver.transformer_low if active == HIGH else driver.transformer_high
        )
        if other_model is None:
            self.logger.warning("bernini_r_second_expert_missing", active=active)
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
        if active == HIGH:
            driver.transformer_low = wrapped
        else:
            driver.transformer_high = wrapped
        driver._set_active(active)

        trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
        self.logger.info(
            "bernini_r_dual_lora_applied",
            second_expert=(LOW if active == HIGH else HIGH),
            second_trainable=trainable,
        )

    def _collect_expert_params(self) -> list[torch.nn.Parameter]:
        """All trainable LoRA params across BOTH experts (deduplicated)."""
        driver: BerniniRDriver = self.driver  # type: ignore[assignment]
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
        """Configure the optimizer over BOTH experts' params (14B ``both``).

        Temporarily patches ``_get_primary_model().parameters`` so the base
        ``_configure_optimization`` (which collects from the primary model) sees
        the union of both experts' trainable params — reusing ALL the base
        LR-scaling / optimizer-factory / scheduler / scaler logic verbatim. The
        1.3B / single-expert path is byte-identical to the base.
        """
        driver = getattr(self, "driver", None)
        if (
            not getattr(driver, "is_dual", False)
            or getattr(self, "expert_mode", "both") != "both"
        ):
            super()._configure_optimization(max_train_steps)
            return

        primary = self._get_primary_model()
        all_params = self._collect_expert_params()
        original_parameters = primary.parameters

        def _both_experts_parameters(*args, **kwargs):
            return iter(all_params)

        primary.parameters = _both_experts_parameters  # type: ignore[method-assign]
        try:
            super()._configure_optimization(max_train_steps)
        finally:
            primary.parameters = original_parameters  # type: ignore[method-assign]

        # Explicit start placement (Task W3.T2, mirrors Wan22Trainer): both
        # experts now carry PEFT adapters and the optimizer holds both
        # experts' params — place them on their configured devices BEFORE the
        # training loop starts, independent of whether the step-0 baseline
        # sampler runs (previously the ONLY thing that placed the deferred low
        # expert).
        driver.place_experts_for_start()

        self.logger.info(
            "bernini_r_optimizer_configured_dual", total_trainable=len(all_params)
        )

    def _configure_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on BOTH experts (14B ``both``)."""
        # Bring the deferred low expert back BEFORE any per-expert work below —
        # grad-checkpointing, then PEFT/optimizer, all expect both present.
        self._load_deferred_experts()
        super()._configure_gradient_checkpointing()
        driver = getattr(self, "driver", None)
        if (
            not getattr(driver, "is_dual", False)
            or getattr(self, "expert_mode", "both") != "both"
            or not self.config.get("gradient_checkpointing", False)
        ):
            return
        active = driver.get_primary_model()
        other = (
            driver.transformer_low
            if driver.active_expert == HIGH
            else driver.transformer_high
        )
        if other is not None and other is not active:
            if hasattr(other, "gradient_checkpointing_enable"):
                other.gradient_checkpointing_enable()
            elif hasattr(other, "enable_gradient_checkpointing"):
                other.enable_gradient_checkpointing()

    def _build_trainable_components(self) -> dict:
        """Expose BOTH PEFT experts to the dual saver as ``unet_high``/``unet_low``.

        1.3B / single-expert: base behaviour (byte-identical). 14B: also thread
        the router state for resume (mirrors :class:`Wan22Trainer`).
        """
        comps = super()._build_trainable_components()
        driver = getattr(self, "driver", None)
        if not getattr(driver, "is_dual", False):
            return comps
        comps["unet_high"] = driver.transformer_high
        comps["unet_low"] = driver.transformer_low
        router = getattr(self, "expert_router", None)
        if router is not None:
            comps["router_state"] = router.state_dict()
        return comps

    # ── Text Encoding (UMT5-XXL via driver, with lazy cache) ─────────────
    # encode_text / _get_cached_text_embeddings live in WanTextCacheMixin (shared
    # verbatim with wan21/wan22 — UMT5, plain caption, frozen TE). The warm path
    # (_pre_cache_text_embeddings) is extended below for Bernini's frozen-negative
    # CFG regime.

    def _pre_cache_text_embeddings(self) -> None:
        """Warm the base set, then ALWAYS warm the pinned empty ("") negative.

        Bernini-R's v2v CFG (#5) ALWAYS encodes the pinned empty-string negative,
        regardless of any user-configured ``sample_negative_prompt`` (this
        family's frozen-cond regime ignores it). The base
        :meth:`WanTextCacheMixin._pre_cache_text_embeddings` warms only the
        CONFIGURED negative (``config.sample_negative_prompt or ""``) — so with a
        non-empty configured negative, ``""`` never enters the cache. The UMT5
        encoder is then offloaded, and the first preview's uncond pass calls
        ``encode_text([""])`` → a miss with no resident encoder →
        ``RuntimeError`` ("Text encoder unavailable"), failing EVERY preview.

        Fix: after the base warm, ensure ``""`` is cached (encoded + persisted via
        the same disk cache convention), and warn-once that a configured
        ``sample_negative_prompt`` is ignored by this family's CFG regime.
        """
        super()._pre_cache_text_embeddings()

        configured_neg = str(self.config.get("sample_negative_prompt", "") or "")
        if configured_neg:
            self._warn_configured_negative_ignored(configured_neg)

        if not self.config.get("cache_text_embeddings", True):
            return
        # Base already warms "" when no negative is configured (the common path).
        if "" in self.text_cache:
            return
        if getattr(self.driver, "text_encoder", None) is None:
            return

        import os

        from app.engine.components.text_embeddings import TextEmbeddingCache

        te_cache_dirs = self._resolve_te_cache_dirs()
        te_quant = self.config.get("te_quantization", "none")
        te1_dir = (
            os.path.join(te_cache_dirs[0], "embeddings", te_quant, "te1")
            if te_cache_dirs
            else ""
        )
        dtype = self._resolve_loading_dtype()

        emb = TextEmbeddingCache.load("", te1_dir, "") if te1_dir else None
        if emb is None:
            with torch.no_grad():
                out = self.driver.encode_text([""], dtype)
            emb_t = out.embeddings if hasattr(out, "embeddings") else out
            emb = emb_t[0:1].cpu()
            if te1_dir:
                TextEmbeddingCache.save("", emb, te1_dir, "")
        self.text_cache[""] = emb

    def _warn_configured_negative_ignored(self, configured_neg: str) -> None:
        """Warn once that this family's CFG ignores a configured negative prompt."""
        if getattr(self, "_warned_negative_ignored", False):
            return
        self._warned_negative_ignored = True
        self.logger.warning(
            "bernini_r_sample_negative_ignored",
            configured=configured_neg[:80],
            message=(
                "Bernini-R's v2v CFG uses a frozen empty ('') negative prompt; "
                "the configured sample_negative_prompt is ignored for previews."
            ),
        )
