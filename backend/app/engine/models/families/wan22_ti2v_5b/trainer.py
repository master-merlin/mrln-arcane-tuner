"""WAN 2.2 TI2V-5B Trainer — wires the family loader/driver/saver/sampler.

All shared training mechanics (optimizer, EMA, gradient accumulation, noise
offset, checkpointing, signals, logging) live in ``GenericTrainingPipeline``.
This trainer additionally owns the two pieces that need the JOB CONFIG (not
just the model definition), so they can't live on the driver alone:

- :meth:`_attach_conditioning` — per-step i2v gate. A single ``mode: both``
  definition serves both video_modes; ``video_mode`` (t2v/i2v) and
  ``first_frame_conditioning_probability`` are RUN config, so the trainer
  Bernoulli-draws per step and flips ``driver._i2v_active`` (ltx2 precedent:
  ``Ltx2Trainer._attach_conditioning``). ``add_noise``/``forward_pass`` then
  auto-delegate to the driver's real overrides (the structural auto-delegation
  guard — see ``core/hook_dispatch.py``), no further trainer wiring needed.
- :meth:`_compute_step_loss` — excludes frame-0 tokens from the loss on
  engaged steps. TI2V-5B pins the conditioning frame to a KNOWN latent
  (zero noise scale — see ``driver.add_noise``), so the flow-match target
  there is not something the model could ever predict from its input; folding
  it into the MSE would silently teach garbage / dampen the true video loss
  (ltx2 parity: ``Ltx2Trainer._compute_step_loss``'s frame-0-token exclusion).
"""

from __future__ import annotations

import random

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline
from app.engine.models.families.wan_shared.trainer_base import WanTextCacheMixin

from .driver import Wan22Ti2v5bDriver
from .loader import Wan22Ti2v5bLoader
from .saver import Wan22Ti2v5bSaver

logger = structlog.get_logger(__name__)


class Wan22Ti2v5bTrainer(WanTextCacheMixin, GenericTrainingPipeline):
    """WAN 2.2 TI2V-5B (dense, T2V+I2V) LoRA trainer."""

    # ── Setup ────────────────────────────────────────────────────────────

    def _setup_family(self) -> None:
        self.driver = Wan22Ti2v5bDriver(self.definition, self.device)
        self.loader = Wan22Ti2v5bLoader(self.device)
        self.saver = Wan22Ti2v5bSaver()

    def _create_sampler(self):
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import Wan22Ti2v5bSampler

            return Wan22Ti2v5bSampler(self)
        return None

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep transformer references in sync after PEFT/quant wrapping."""
        self.transformer = new_model
        self.components["unet"] = new_model
        self.driver.transformer = new_model

    # ── Per-step i2v gate (needs job config — driver has none) ────────────

    def _attach_conditioning(self, batch: dict, latents: object) -> None:
        """Bernoulli-gate i2v conditioning for THIS step from ``video_mode``.

        Mirrors ``Ltx2Trainer._attach_conditioning``: only relevant when the
        run is configured as ``video_mode == "i2v"``; a fraction (
        ``first_frame_conditioning_probability``, default 0.5 — the WAN i2v
        recipe trains a mix of conditioned + unconditioned steps) of steps
        engage the first-frame pin, the rest train unconditional (plain t2v)
        generation. No batch stash is needed for this scheme (unlike
        wan21/wan22's 36-channel ``BATCH_FIRST_FRAME_LATENT`` — see
        ``driver.add_noise``'s docstring), so this method's only job is
        flipping the driver's per-step flag.
        """
        active = False
        if str(self.config.get("video_mode", "t2v")).lower() == "i2v":
            p = float(self.config.get("first_frame_conditioning_probability", 0.5))
            active = random.random() < p
        self.driver._i2v_active = active

    # ── Loss: exclude the pinned conditioning frame on engaged steps ──────

    def _compute_step_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        batch: dict,
        grad_accum: int,
    ) -> torch.Tensor:
        """Frame-0 tokens are excluded from the loss when i2v is engaged.

        Non-engaged steps (t2v, or an F=1 still on an i2v-active step) are
        byte-identical to the base weighted-MSE loss.
        """
        if self.driver._conditioning_engaged(target):
            pred = pred[:, :, 1:, :, :]
            target = target[:, :, 1:, :, :]
        return super()._compute_step_loss(pred, target, timesteps, batch, grad_accum)

    # ── Text Encoding (UMT5-XXL via driver, with lazy cache) ─────────────
    # encode_text / _get_cached_text_embeddings live in WanTextCacheMixin
    # (byte-identical between wan21/wan22/wan22_ti2v_5b; hoisted to wan_shared).
