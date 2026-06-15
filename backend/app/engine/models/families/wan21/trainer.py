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

from typing import Any

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

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, batch: dict | None = None
    ) -> Any:
        """Encode captions through UMT5-XXL with in-memory caching.

        Returns a raw ``[B, L, D]`` tensor (the WAN transformer takes
        ``encoder_hidden_states`` directly).
        """
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
