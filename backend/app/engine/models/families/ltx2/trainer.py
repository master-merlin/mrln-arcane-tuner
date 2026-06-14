"""LTX 2.3 Trainer — family hooks for the generic training pipeline.

Implements LTX-2-specific behaviour:
- Single frozen Gemma3 text encoder → ``LTX2TextConnectors`` → video/audio emb.
- Flow matching on the ``[0, 1000]`` scale (driver ``add_noise`` override).
- 5D video latents packed via ``_pack_latents`` (patch_size / patch_size_t).
- Optional joint audio stream: when ``train_audio`` is on, the audio VAE +
  vocoder are loaded and the loss adds ``audio_weight * masked_audio_fm``.

When audio is OFF the audio components are never requested and the loss is the
plain video flow-match MSE — identical to the audio-free pipeline path.
"""

from __future__ import annotations

from typing import Any

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline

from .driver import Ltx2Driver
from .loader import Ltx2Loader
from .saver import Ltx2Saver

logger = structlog.get_logger(__name__)


class Ltx2Trainer(GenericTrainingPipeline):
    """LTX 2.3 (joint audio + video) LoRA trainer."""

    # ── Setup ────────────────────────────────────────────────────────────

    def _setup_family(self) -> None:
        """Initialize LTX-2 loader, saver, driver, and audio gating."""
        train_audio = self._resolve_train_audio()
        self.driver = Ltx2Driver(self.definition, self.device)
        self.loader = Ltx2Loader(self.device, train_audio=train_audio)
        self.saver = Ltx2Saver()
        # Surface the resolved flag onto the driver so get_lora_targets,
        # compute_loss, etc. gate the audio sub-stream consistently even before
        # components are assigned (assign_components re-confirms it from arch).
        self.driver.train_audio = train_audio

    def _resolve_train_audio(self) -> bool:
        """Decide whether to train the audio stream for this run.

        Audio training requires BOTH: the user opted in (``train_audio`` config,
        default False) AND the model declares ``has_audio`` in its definition.
        Absent either, the run is video-only.
        """
        arch = getattr(self.definition, "architecture_params", {}) or {}
        model_has_audio = bool(arch.get("has_audio", False))
        user_wants_audio = bool(self.config.get("train_audio", False))
        return model_has_audio and user_wants_audio

    def _create_sampler(self):
        """Create an Ltx2Sampler if sampling is configured."""
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import Ltx2Sampler

            return Ltx2Sampler(self)
        return None

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep self.transformer in sync after PEFT/quantization wrapping."""
        self.transformer = new_model
        self.components["unet"] = new_model
        self.driver.transformer = new_model

    # ── Joint audio + video loss ─────────────────────────────────────────

    def _compute_step_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        batch: dict[str, Any],
        grad_accum: int,
    ) -> torch.Tensor:
        """Route loss through the driver's joint audio+video recipe.

        For video-only runs this is the plain video flow-match MSE (identical to
        the base implementation).  When ``train_audio`` is on, the audio
        prediction/target/mask are read from ``batch`` (populated by the audio
        forward path) and the driver adds ``audio_weight * masked_audio_fm``,
        sharing the SAME timestep.
        """
        audio_pred = batch.get("audio_pred")
        audio_target = batch.get("audio_target")
        audio_mask = batch.get("audio_mask")

        loss = self.driver.compute_loss(
            pred,
            target,
            batch,
            audio_pred=audio_pred,
            audio_target=audio_target,
            audio_mask=audio_mask,
        )
        return loss / grad_accum
