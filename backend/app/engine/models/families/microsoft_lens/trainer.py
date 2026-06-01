"""Microsoft Lens trainer -- family hooks over the generic training pipeline.

Lens text features are a 4-layer stack per caption ([4, S, 2880]) plus a bool
mask ([S]). We cache per-caption on CPU and right-pad to the longest in-batch
sequence at assembly time -- mirroring ErnieImageTrainer's variable-length
caching.
"""

from __future__ import annotations

import os

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline
from .driver import MicrosoftLensDriver
from .loader import MicrosoftLensLoader
from .saver import MicrosoftLensSaver

logger = structlog.get_logger(__name__)


def pad_lens_text_batch(
    entries: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad cached ``(feat[4,S,D], mask[S])`` entries to ``[B,4,S_max,D]``."""
    if not entries:
        raise ValueError("pad_lens_text_batch received no entries (empty batch).")
    n_layers = entries[0][0].shape[0]
    feat_dim = entries[0][0].shape[-1]
    s_max = max(e[0].shape[1] for e in entries)
    b = len(entries)

    feats = torch.zeros((b, n_layers, s_max, feat_dim), device=device, dtype=dtype)
    mask = torch.zeros((b, s_max), device=device, dtype=torch.bool)
    for i, (feat, m) in enumerate(entries):
        s = feat.shape[1]
        feats[i, :, :s, :] = feat.to(device=device, dtype=dtype)
        mask[i, :int(m.sum().item())] = True
    return feats, mask


class MicrosoftLensTrainer(GenericTrainingPipeline):
    """Microsoft Lens LoRA trainer."""

    def _setup_family(self) -> None:
        self.driver = MicrosoftLensDriver(self.definition, self.device)
        self.loader = MicrosoftLensLoader(self.device)
        self.saver = MicrosoftLensSaver()
        # 2x2 patchify -> the flow-matching shift sampler computes S = (H/2)(W/2).
        self.config.setdefault("flux_shift_patchify_factor", 2)

    def _create_sampler(self):
        return None  # no in-training sampling in v1

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        self.transformer = new_model
        self.components["unet"] = new_model
        self.driver.transformer = new_model

    # --- Text encoding (cached) ---

    def encode_text(self, captions: list[str], dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.get("cache_text_embeddings", True):
            return self._get_cached_text_embeddings(captions, dtype)
        out = self.driver.encode_text(captions, dtype)
        return out.embeddings, out.attention_mask

    def _get_cached_text_embeddings(self, captions: list[str], dtype: torch.dtype):
        uncached = [c for c in captions if c not in self.text_cache]
        if uncached and self.text_encoder is not None:
            te_device = next(self.text_encoder.parameters()).device
            te_was_offloaded = te_device != self.device
            if te_was_offloaded:
                self.logger.warning(
                    "te_cache_miss_after_offload",
                    count=len(uncached),
                    hint="pre-caching should have covered all captions",
                )
                self.text_encoder.to(self.device)

            for cap in uncached:
                out = self.driver.encode_text([cap], dtype)
                self.text_cache[cap] = (
                    out.embeddings.squeeze(0).cpu(),
                    out.attention_mask.squeeze(0).cpu(),
                )

            if te_was_offloaded:
                self.text_encoder.to("cpu")
                torch.cuda.empty_cache()

            self.logger.debug(
                "text_embeddings_cached",
                new=len(uncached),
                total=len(self.text_cache),
            )
        elif uncached:
            raise RuntimeError(
                "Text encoder unloaded but uncached captions encountered: "
                + ", ".join(c[:50] for c in uncached)
            )
        entries = [self.text_cache[c] for c in captions]
        return pad_lens_text_batch(entries, self.device, dtype)

    def _pre_cache_text_embeddings(self) -> None:
        """Warm the text cache from disk + encode any missing captions.

        Disk layout mirrors ERNIE: ``te1/`` = stacked features, ``te2/`` = mask.
        """
        if not self.config.get("cache_text_embeddings", True):
            return
        if self.text_encoder is None:
            return

        from app.engine.components.text_embeddings import TextEmbeddingCache

        te_cache_dirs = self._resolve_te_cache_dirs()
        te_quant = self.config.get("te_quantization", "none")
        te1_dir = (
            os.path.join(te_cache_dirs[0], "embeddings", te_quant, "te1")
            if te_cache_dirs else ""
        )
        te2_dir = (
            os.path.join(te_cache_dirs[0], "embeddings", te_quant, "te2")
            if te_cache_dirs else ""
        )

        caption_hints = self._build_caption_hints()
        dtype = self._resolve_loading_dtype()

        disk_loaded = 0
        need_encode: list[tuple[str, str]] = []
        for caption, hint in caption_hints.items():
            if caption in self.text_cache:
                continue
            if te1_dir and te2_dir:
                feat = TextEmbeddingCache.load(caption, te1_dir, hint)
                mask = TextEmbeddingCache.load(caption, te2_dir, hint)
                if feat is not None and mask is not None:
                    self.text_cache[caption] = (feat, mask)
                    disk_loaded += 1
                    continue
            need_encode.append((caption, hint))

        total = len(caption_hints)
        self.logger.info(
            "te_disk_cache_status",
            total=total,
            from_disk=disk_loaded,
            need_encode=len(need_encode),
        )

        if not need_encode:
            if getattr(self, "_log_writer", None):
                self._log_writer.status("TE Cache Loaded from Disk")
            self.logger.info(
                "text_embedding_cache_complete",
                cached=len(self.text_cache), source="disk",
            )
            return

        if getattr(self, "_log_writer", None):
            self._log_writer.status("Caching Text Embeddings (0%)")
        encode_total = len(need_encode)
        with torch.no_grad():
            for i, (caption, hint) in enumerate(need_encode):
                out = self.driver.encode_text([caption], dtype)
                feat = out.embeddings.squeeze(0).cpu()
                mask = out.attention_mask.squeeze(0).cpu()
                self.text_cache[caption] = (feat, mask)
                if te1_dir:
                    TextEmbeddingCache.save(caption, feat, te1_dir, hint)
                if te2_dir:
                    TextEmbeddingCache.save(caption, mask, te2_dir, hint)
                pct = int((i + 1) / encode_total * 100)
                if (pct % 10 == 0 or (i + 1) == encode_total) and getattr(self, "_log_writer", None):
                    self._log_writer.status(f"Caching Text Embeddings ({pct}%)")

        self.logger.info(
            "text_embedding_cache_complete",
            cached=len(self.text_cache), newly_encoded=encode_total,
        )
