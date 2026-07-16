"""Bernini-R trainer — SD3-mode timesteps + target-token flow-match loss.

Bernini-R is a renderer-only video-EDIT DiT built from stock Wan components, so
this trainer mirrors :class:`Wan21Trainer`'s base-class shape exactly
(``(WanTextCacheMixin, GenericTrainingPipeline)``): UMT5-XXL text encoding with
the shared lazy + disk cache, a frozen text encoder, plain captions,
``max_sequence_length`` 512, no chat template. All the generic training mechanics
(optimizer, EMA, grad-accum, checkpointing, logging, sampling wiring) come from
:class:`GenericTrainingPipeline`.

The two Bernini-specific pieces:

- **Timestep sampling** (:meth:`sample_timesteps`) — the upstream renderer's
  ``NoiseScheduler`` for the video tasks: SD3 ``mode`` weighting followed by the
  per-task shift-warp, in the RAW ``[0, 1000]`` space (the wan flow-match
  ``add_noise`` lerp does the ``/1000``). This OVERRIDES the base flow-match
  sampler — the base ``PipelineBaseMixin.sample_timesteps`` uses a different
  (logit-normal default) distribution, so leaving it un-overridden would train
  off the wrong noise schedule (the boogu/wan22 convention-delegation class of
  bug). The formula is transcribed verbatim from upstream ``bernini/training/
  data.py`` ``NoiseScheduler`` (see ``.agent/workdir/bernini-r-recon.md`` §4).

- **Loss** — plain flow-match velocity ``noise - x0`` on the TARGET tokens only.
  This needs NO trainer override: the base ``compute_target`` already returns
  ``noise - latents`` over the target latent, and :class:`BerniniRDriver`'s packed
  forward returns ONLY the target-token slice (condition-token predictions are
  sliced off inside ``vendor/transformer_forward.py`` and can never reach the
  MSE). Condition latents ride in ``batch['control_latents']`` (clean, from BR1)
  and never touch the noise/target path. Pinned by ``test_bernini_r_trainer.py``.

``video_dropout`` is intentionally NOT implemented in v1: v2v CFG keeps the
source video in the *unconditional* branch (recon §5), so dropping it during
training is unnecessary. Text (caption) dropout is the existing generic
``caption_dropout_rate`` seam (upstream ``text_dropout_rate 0.1``).
"""

from __future__ import annotations

import math

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline
from app.engine.models.families.wan_shared.trainer_base import WanTextCacheMixin

from .driver import BerniniRDriver
from .loader import BerniniRLoader
from .saver import BerniniRSaver

logger = structlog.get_logger(__name__)


class BerniniRTrainer(WanTextCacheMixin, GenericTrainingPipeline):
    """Bernini-R (renderer-only video edit, 1.3B v1) LoRA trainer.

    ``is_video_family`` is inherited from :class:`PipelineBaseMixin` (derived from
    the model's ``is_video`` capability) — no per-trainer flag needed.
    """

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
        self.loader = BerniniRLoader(self.device)
        self.saver = BerniniRSaver(mode=self.driver.mode)

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep transformer references in sync after PEFT/quant wrapping.

        The base method updates ``self.components['unet']`` / ``self.model`` but
        does NOT reach into ``self.driver.transformer``; without this the packed
        forward would run the un-wrapped (non-LoRA'd) transformer (mirrors
        :class:`Wan21Trainer`).
        """
        self.transformer = new_model
        self.components["unet"] = new_model
        self.driver.transformer = new_model

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
    def sample_timesteps(
        self, batch_size: int, latents: torch.Tensor | None = None
    ) -> torch.Tensor:
        """SD3 ``mode`` weighting → per-task shift-warp → RAW ``[0, 1000]``.

        Transcribed verbatim from upstream ``NoiseScheduler`` (recon §4)::

            raw    = torch.rand(n)
            u      = 1 - raw - mode_scale * (cos(pi * raw / 2) ** 2 - 1 + raw)
            sigmas = shift * u / (1 + (shift - 1) * u)
            timesteps = sigmas * 1000.0

        ``shift`` (v2v default 5.0) and ``mode_scale`` (1.29) are read from config
        so the definition (BR4) / user can select the per-task shift; the family
        defaults reproduce the v2v recipe. The result is RAW ``[0, 1000]`` — the
        wan flow-match ``add_noise`` lerp applies the ``/1000`` (the pure-noise
        gotcha: the frozen time embedder must see the un-scaled value).
        """
        mode_scale = float(self.config.get("mode_scale", self.DEFAULT_MODE_SCALE))
        shift = float(self.config.get("timestep_shift", self.DEFAULT_TIMESTEP_SHIFT))

        raw = torch.rand(batch_size, device=self.device)
        u = 1.0 - raw - mode_scale * (torch.cos(math.pi * raw / 2.0) ** 2 - 1.0 + raw)
        sigmas = shift * u / (1.0 + (shift - 1.0) * u)
        return sigmas * 1000.0

    # ── Text Encoding (UMT5-XXL via driver, with lazy cache) ─────────────
    # encode_text / _get_cached_text_embeddings / _pre_cache_text_embeddings live
    # in WanTextCacheMixin (shared verbatim with wan21/wan22 — UMT5, plain
    # caption, frozen TE).
