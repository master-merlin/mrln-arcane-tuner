"""Qwen-Image 2 trainer -- family hooks for the generic training pipeline.

Mirrors the overrides its siblings carry (LESSONS, krea2 entry: a new family
trainer that does not, crashes only on the GPU):

- ``_update_primary_model`` syncs the DRIVER's transformer reference, the
  trainer alias and ``components["unet"]`` after PEFT/quant wrapping;
- ``encode_text`` returns the ``(embeddings, mask)`` tuple the driver's
  ``forward_pass`` unpacks, through a per-caption in-memory cache keyed by the
  raw caption (the cross-family seam contract);
- ``_pre_cache_text_embeddings`` warms that cache from disk, keyed on the
  caption AND ``driver.te_cache_identity()`` (template, encoder class,
  ``te.max_length``) -- LANE-132 spec 3;
- ``VAE_IMAGE_MODE = "RGBA"``: the 2.1 VAE reads four channels (spec review
  2.01; ``pipeline_data.PipelineDataMixin.VAE_IMAGE_MODE``);
- ``_compile_if_quantized`` is a no-op: the transformer is incompatible with
  ``torch.compile`` (diffusers #14821, open at ``6256aa7666``).
"""

from __future__ import annotations

import os

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline

from .driver import QwenImage2Driver
from .loader import QwenImage2Loader

logger = structlog.get_logger(__name__)


class QwenImage2Trainer(GenericTrainingPipeline):
    """Qwen-Image 2.1 LoRA trainer (single-stream 32-layer DiT, Qwen3-VL TE)."""

    VAE_IMAGE_MODE = "RGBA"

    # -- Setup --

    def _setup_family(self) -> None:
        from .saver import QwenImage2Saver

        self.driver = QwenImage2Driver(self.definition, self.device)
        self.loader = QwenImage2Loader(self.device)
        self.saver = QwenImage2Saver()

    def _create_sampler(self):
        if int(self.config.get("sample_every_n_steps", 0)) > 0:
            from .sampler import QwenImage2Sampler

            return QwenImage2Sampler(self)
        return None

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep the trainer alias, the components dict and the driver in sync."""
        self.transformer = new_model
        self.components["unet"] = new_model
        self.driver.transformer = new_model

    def _compile_if_quantized(self) -> None:
        """Never compile: ``QwenImage21Transformer2DModel`` breaks under
        ``torch.compile`` (diffusers #14821). The FP8 path then runs eager."""
        self.logger.info("qwen_image2_compile_skipped", reason="diffusers#14821")

    # -- Text embeddings --

    def _disk_cache_key(self, caption: str) -> str:
        """The string ``TextEmbeddingCache`` hashes: every input that decides
        the embedding, not the caption alone (LANE-132 spec 3)."""
        return f"{self.driver.te_cache_identity()}::{caption}"

    def _pre_cache_text_embeddings(self) -> None:
        """Warm ``text_cache`` from disk, encode what is missing, persist it.

        ``te_disk_hits`` lists the captions served from disk in this call -- the
        observable the TE-cache tests read (a stale hit is a hit after an
        encoder or max-length change).
        """
        self.te_disk_hits: list[str] = []
        if not self.config.get("cache_text_embeddings", True):
            return
        if self.text_encoder is None:
            return

        from app.engine.components.text_embeddings import TextEmbeddingCache

        te_cache_dirs = self._resolve_te_cache_dirs()
        te_quant = self.config.get("te_quantization", "none")
        base = os.path.join(te_cache_dirs[0], "embeddings", te_quant) if te_cache_dirs else ""
        emb_dir = os.path.join(base, "te1") if base else ""
        mask_dir = os.path.join(base, "te2") if base else ""

        caption_hints = self._build_caption_hints()
        need_encode: list[tuple[str, str]] = []
        for caption, hint in caption_hints.items():
            if caption in self.text_cache:
                continue
            if emb_dir:
                key = self._disk_cache_key(caption)
                emb = TextEmbeddingCache.load(key, emb_dir, hint)
                mask = TextEmbeddingCache.load(key, mask_dir, hint)
                if emb is not None and mask is not None:
                    self.text_cache[caption] = (emb, mask)
                    self.te_disk_hits.append(caption)
                    continue
            need_encode.append((caption, hint))

        self.logger.info(
            "te_disk_cache_status",
            total=len(caption_hints),
            from_disk=len(self.te_disk_hits),
            need_encode=len(need_encode),
        )
        if not need_encode:
            if getattr(self, "_log_writer", None):
                self._log_writer.status("TE Cache Loaded from Disk")
            return

        if getattr(self, "_log_writer", None):
            self._log_writer.status("Caching Text Embeddings (0%)")
        dtype = self._resolve_loading_dtype()
        batch_size = 4
        total = len(need_encode)
        for i in range(0, total, batch_size):
            items = need_encode[i : i + batch_size]
            emb_batch, mask_batch = self._encode_text_direct([c for c, _ in items], dtype)
            for j, (cap, hint) in enumerate(items):
                emb, mask = self._trim_entry(emb_batch[j].cpu(), mask_batch[j].cpu())
                self.text_cache[cap] = (emb, mask)
                if emb_dir:
                    key = self._disk_cache_key(cap)
                    TextEmbeddingCache.save(key, emb, emb_dir, hint)
                    TextEmbeddingCache.save(key, mask, mask_dir, hint)
            if getattr(self, "_log_writer", None):
                pct = int(min(i + batch_size, total) / total * 100)
                self._log_writer.status(f"Caching Text Embeddings ({pct}%)")

        self.logger.info("text_embedding_cache_complete", cached=len(self.text_cache), newly_encoded=total)

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(embeddings [B, L, D], attention_mask [B, L])`` for ``captions``."""
        if self.config.get("cache_text_embeddings", True):
            return self._get_cached_text_embeddings(captions, dtype)
        return self._encode_text_direct(captions, dtype)

    def _encode_text_direct(
        self, captions: list[str], dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.driver.encode_text(captions, dtype)
        return out.embeddings, out.attention_mask

    @staticmethod
    def _trim_entry(emb: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cut one caption out of its batch's padding (right-padded: a prefix)."""
        n = int(mask.sum().item())
        return emb[:n], mask[:n]

    def _get_cached_text_embeddings(
        self, captions: list[str], dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode on first sight, then reassemble ragged entries zero-padded on
        the right to the batch's longest -- what upstream does to a batch."""
        uncached = [c for c in dict.fromkeys(captions) if c not in self.text_cache]
        if uncached:
            if self.text_encoder is None:
                raise RuntimeError(
                    "Text encoder was unloaded but encountered uncached caption(s): "
                    + ", ".join(c[:50] for c in uncached)
                )
            te_device = next(self.text_encoder.parameters()).device
            moved = te_device != self.device
            if moved:
                self.logger.warning("te_cache_miss_after_offload", count=len(uncached))
                self.text_encoder.to(self.device)
            try:
                for cap in uncached:
                    emb, mask = self._encode_text_direct([cap], dtype)
                    self.text_cache[cap] = self._trim_entry(emb[0].cpu(), mask[0].cpu())
            finally:
                if moved:
                    self.text_encoder.to(te_device)

        entries = [self.text_cache[c] for c in captions]
        max_len = max(e.shape[0] for e, _ in entries)
        embs, masks = [], []
        for emb, mask in entries:
            emb = emb.to(self.device, dtype=dtype)
            mask = mask.to(self.device)
            pad = max_len - emb.shape[0]
            if pad:
                emb = torch.cat([emb, emb.new_zeros(pad, *emb.shape[1:])])
                mask = torch.cat([mask, mask.new_zeros(pad)])
            embs.append(emb)
            masks.append(mask)
        return torch.stack(embs), torch.stack(masks)
