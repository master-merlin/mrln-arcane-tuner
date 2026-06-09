"""Ideogram 4 trainer -- family hooks over the generic training pipeline.

Ideogram 4 text features are a single concatenated multi-layer stack per
caption (``[S, 13*4096]`` after the driver interleaves the 13 selected
Qwen3-VL hidden states) plus a bool attention mask (``[S]``).  ``S`` varies
per caption (the driver tokenises with ``padding=True`` per batch, but cached
entries are stored at their own real length).  We cache per-caption on CPU and
right-pad to the longest in-batch sequence at assembly time -- mirroring
``microsoft_lens`` / ``ErnieImageTrainer`` variable-length caching.

**Why lens-style caching (not the generic path):** the generic
``PipelineBaseMixin.encode_text`` simply delegates to ``driver.encode_text``
with NO caching and NO batch collation -- it assumes the driver re-encodes the
whole batch every step.  That breaks once the text encoder is offloaded/unloaded
after pre-caching (the default ``cache_text_embeddings=True`` flow), and it
re-runs the multi-billion-param Qwen3-VL every step.  There is no generic
per-caption cache that pads variable-length sequences, so -- exactly like lens
and ernie -- we override ``encode_text`` / ``_get_cached_text_embeddings`` /
``_pre_cache_text_embeddings`` and supply :func:`pad_ideogram4_text_batch`.
Ideogram's per-caption feature is 2D (``[S, D]``) where lens's is 3D
(``[4, S, D]``), so the pad is the 2D analogue: right-pad to ``[B, S_max, D]``
plus a ``[B, S_max]`` bool mask -- precisely the ``(text_feats, text_mask)``
pair ``IdeogramV4Driver.forward_pass`` / ``_build_packed_inputs`` consume.

**No ``flux_shift_patchify_factor``:** lens sets it because the generic
flow-matching *shift* sampler estimates the image sequence length ``S`` from the
spatial latent grid as ``(H/2)(W/2)`` for its 2×2 patchify.  Ideogram's
timestep handling is different: the DiT's ``Ideogram4EmbedScalar`` takes the
flow value in ``[0, 1]`` and the driver divides the trainer ``[0, 1000]``
timestep by ``NUM_TRAIN_TIMESTEPS`` inside ``forward_pass`` (see the driver's
class-attr comment / the ×1000 guard).  The patchify-factor key only feeds the
flow-shift S-estimate, not Ideogram's path, so setting it here would be a no-op
at best and a wrong S-estimate at worst.  Omitted deliberately.
"""

from __future__ import annotations

import os

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline
from .driver import IdeogramV4Driver
from .loader import IdeogramV4Loader
from .saver import IdeogramV4Saver

logger = structlog.get_logger(__name__)


def pad_ideogram4_text_batch(
    entries: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad cached ``(feat[S,D], mask[S])`` entries to ``[B,S_max,D]``.

    Mirrors ``microsoft_lens.pad_lens_text_batch`` but for Ideogram's 2D
    per-caption features (lens carries an extra leading layer dim).  Returns
    the padded features ``[B, S_max, D]`` and a bool mask ``[B, S_max]`` -- the
    ``(text_feats, text_mask)`` pair ``IdeogramV4Driver`` expects.
    """
    if not entries:
        raise ValueError("pad_ideogram4_text_batch received no entries (empty batch).")
    feat_dim = entries[0][0].shape[-1]
    s_max = max(e[0].shape[0] for e in entries)
    b = len(entries)

    feats = torch.zeros((b, s_max, feat_dim), device=device, dtype=dtype)
    mask = torch.zeros((b, s_max), device=device, dtype=torch.bool)
    for i, (feat, m) in enumerate(entries):
        s = feat.shape[0]
        feats[i, :s, :] = feat.to(device=device, dtype=dtype)
        mask[i, : int(m.sum().item())] = True
    return feats, mask


class IdeogramV4Trainer(GenericTrainingPipeline):
    """Ideogram 4 LoRA trainer."""

    def _setup_family(self) -> None:
        self.driver = IdeogramV4Driver(self.definition, self.device)
        self.loader = IdeogramV4Loader(self.device)
        self.saver = IdeogramV4Saver()
        # NOTE: deliberately NO ``flux_shift_patchify_factor`` -- see module
        # docstring. Ideogram divides the [0,1000] timestep by 1000 inside the
        # driver's forward_pass; it does not use the flow-shift S-estimate.

    def _create_sampler(self):
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval <= 0:
            return None
        # Ensure the driver owns the live TE reference (restored above if a
        # re-assignment dropped it), then release our capture alias so the
        # driver remains the single owner.
        self._restore_sampling_text_encoder()
        self._sampling_text_encoder = None
        self._sampling_tokenizer = None
        # Lazy import: the sampler lands in a later task; keeping the import
        # inside the method keeps this module importable until then.
        from .sampler import IdeogramV4Sampler
        return IdeogramV4Sampler(self)

    # --- Text-encoder retention for in-training sampling ---

    def _offload_text_encoders(self) -> None:
        """Offload TEs after caching, retaining a ref for sampling.

        In-training sampling encodes *new* prompts, so it needs the Qwen3-VL
        text encoder after the caching phase. The generic flow re-runs
        ``_assign_components`` during TE-quantization setup; because offload
        pops the TE from ``self.components``, that re-assignment drops the
        driver's reference and ``gc`` frees the module. When sampling is
        enabled and the TE is merely offloaded (not unloaded), keep a strong
        reference so it survives; :meth:`_assign_components` restores it.
        """
        super()._offload_text_encoders()
        if (
            int(self.config.get("sample_every_n_steps", 0)) > 0
            and not self.config.get("unload_text_encoder", False)
        ):
            self._sampling_text_encoder = self.driver.text_encoder
            self._sampling_tokenizer = self.driver.tokenizer

    def _assign_components(self) -> None:
        super()._assign_components()
        # Restore the sampling TE/tokenizer if a re-assignment dropped them
        # (the offloaded TE is no longer in the components dict).
        self._restore_sampling_text_encoder()

    def _restore_sampling_text_encoder(self) -> None:
        """Re-attach the captured TE/tokenizer onto the driver if dropped."""
        te = getattr(self, "_sampling_text_encoder", None)
        if te is not None and self.driver.text_encoder is None:
            self.driver.text_encoder = te
            self.driver.tokenizer = self._sampling_tokenizer
            self.text_encoder = te
            self.tokenizer = self._sampling_tokenizer

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        self.transformer = new_model
        self.components["unet"] = new_model
        self.driver.transformer = new_model

    # --- Text encoding (cached) ---

    def encode_text(
        self, captions: list[str], dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return pad_ideogram4_text_batch(entries, self.device, dtype)

    def _pre_cache_text_embeddings(self) -> None:
        """Warm the text cache from disk + encode any missing captions.

        Disk layout mirrors ERNIE / Lens: ``te1/`` = stacked features,
        ``te2/`` = mask.
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
