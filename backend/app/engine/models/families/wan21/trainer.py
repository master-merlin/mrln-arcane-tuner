"""WAN 2.1 Trainer — wires the family loader/driver/saver/sampler.

All shared training mechanics (optimizer, EMA, gradient accumulation, noise
offset, checkpointing, signals, logging) live in ``GenericTrainingPipeline``.
This trainer:

- selects the WAN 2.1 loader/driver/saver in ``_setup_family``,
- is treated as a video family (the base ``is_video_family`` property derives
  this from the model's ``is_video`` capability) so the data path collates 5D
  ``[B, C, F, H, W]`` clips,
- delegates text encoding to the driver (UMT5-XXL) with lazy in-memory caching,
- creates a :class:`Wan21Sampler` when sampling is configured.

GPU end-to-end training is a follow-up: it needs real WAN weights + a CUDA
device. The unit tests exercise the family wiring, definitions, LoRA targets,
saver, I2V conditioning, and the flow-match / autocast precision contracts
against the REAL driver/sampler code paths with fakes.
"""

from __future__ import annotations

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline
from app.engine.models.families.wan_shared.trainer_base import WanTextCacheMixin

from .driver import Wan21Driver
from .loader import Wan21Loader
from .saver import Wan21Saver

logger = structlog.get_logger(__name__)


class Wan21Trainer(WanTextCacheMixin, GenericTrainingPipeline):
    """WAN 2.1 (T2V 1.3B/14B, I2V 14B) LoRA trainer.

    ``is_video_family`` is inherited from :class:`PipelineBaseMixin` (derived
    from the model's ``is_video`` capability) — no per-trainer flag needed.
    """

    # ── Setup ────────────────────────────────────────────────────────────

    def _setup_family(self) -> None:
        self.driver = Wan21Driver(self.definition, self.device)
        self.loader = Wan21Loader(self.device)
        self.saver = Wan21Saver(mode=self.driver.mode)

    def _create_sampler(self):
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import Wan21Sampler

            return Wan21Sampler(self)
        return None

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep transformer references in sync after PEFT/quant wrapping."""
        self.transformer = new_model
        self.components["unet"] = new_model
        self.driver.transformer = new_model

    # ── Text Encoding (UMT5-XXL via driver, with lazy cache) ─────────────
    # encode_text / _get_cached_text_embeddings live in WanTextCacheMixin
    # (byte-identical between wan21 and wan22; hoisted to wan_shared).
