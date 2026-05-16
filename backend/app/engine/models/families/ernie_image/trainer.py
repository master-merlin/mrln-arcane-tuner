"""ERNIE-Image Trainer -- family-specific hooks for the generic training pipeline.

All shared logic (optimizer, EMA, gradient accumulation, noise offset,
checkpointing, logging) lives in :class:`GenericTrainingPipeline`.
This module implements only the ERNIE-Image-specific surface:

*  Single text encoder (Mistral3-derived) with per-prompt encoding,
   second-to-last hidden layer extraction, and no chat template.
*  Disk-backed ``(embedding, attention_mask)`` cache (per-prompt
   variable length).
*  ``flux_shift_patchify_factor=2`` so the flow-matching timestep
   shift sampler computes the right sequence length.
*  Forward pass passes ``(text_bth, attention_mask)`` through to the
   driver as a tuple; the driver derives ``text_lens`` from the mask.
"""

import os

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline
from .driver import ErnieImageDriver
from .loader import ErnieImageLoader
from .saver import ErnieImageSaver

logger = structlog.get_logger(__name__)


class ErnieImageTrainer(GenericTrainingPipeline):
    """Baidu ERNIE-Image LoRA trainer."""

    # -- Setup --

    def _setup_family(self) -> None:
        """Initialize ERNIE-Image-specific loader, saver, driver, and caches."""
        self.driver = ErnieImageDriver(self.definition, self.device)
        self.loader = ErnieImageLoader(self.device)
        self.saver = ErnieImageSaver()

        # ERNIE patchifies latents 2x2 before the transformer; tell the
        # flow-matching shift sampler so seq_len = (H/2) * (W/2).
        self.config.setdefault("flux_shift_patchify_factor", 2)

    def _create_sampler(self):
        """Create an ErnieImageSampler if sampling is configured."""
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import ErnieImageSampler
            return ErnieImageSampler(self)
        return None

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep self.transformer in sync after PEFT / quantization wrapping."""
        self.transformer = new_model
        self.components["unet"] = new_model
        self.driver.transformer = new_model

    # -- Disk-backed TE Pre-caching --

    def _pre_cache_text_embeddings(self) -> None:
        """Warm text embedding cache from disk + encode missing.

        ERNIE-Image caches ``(embedding, attention_mask)`` tuples:
        ``te1/`` stores per-prompt hidden-state tensors,
        ``te2/`` stores the attention-mask tensors (which carry
        per-sample lengths via ``mask.sum(dim=-1)``).
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

        disk_loaded = 0
        need_encode: list[tuple[str, str]] = []
        for caption, hint in caption_hints.items():
            if caption in self.text_cache:
                continue
            if te1_dir and te2_dir:
                emb_tensor = TextEmbeddingCache.load(caption, te1_dir, hint)
                mask_tensor = TextEmbeddingCache.load(caption, te2_dir, hint)
                if emb_tensor is not None and mask_tensor is not None:
                    self.text_cache[caption] = (emb_tensor, mask_tensor)
                    disk_loaded += 1
                    continue
            need_encode.append((caption, hint))

        total = len(caption_hints)
        self.logger.info(
            "te_disk_cache_status",
            total=total,
            from_memory=total - disk_loaded - len(need_encode),
            from_disk=disk_loaded,
            need_encode=len(need_encode),
        )

        if not need_encode:
            print("[STATUS:TE Cache Loaded from Disk]", flush=True)
            self.logger.info(
                "text_embedding_cache_complete",
                cached=len(self.text_cache), source="disk",
            )
            return

        # Encode missing captions one-by-one (ERNIE encodes per-prompt; no
        # benefit from batching at the TE level because each prompt has its
        # own variable length).
        print("[STATUS:Caching Text Embeddings (0%)]", flush=True)
        encode_total = len(need_encode)
        dtype = self._resolve_loading_dtype()

        with torch.no_grad():
            for i, (caption, hint) in enumerate(need_encode):
                output = self.driver.encode_text([caption], dtype)
                emb = output.embeddings
                mask = output.attention_mask
                if mask is None:
                    raise RuntimeError(
                        "ErnieImageDriver.encode_text did not return an "
                        "attention_mask; cache cannot recover per-sample lengths.",
                    )

                emb_cpu = emb[0].cpu()
                mask_cpu = mask[0].cpu()
                self.text_cache[caption] = (emb_cpu, mask_cpu)
                if te1_dir:
                    TextEmbeddingCache.save(caption, emb_cpu, te1_dir, hint)
                if te2_dir:
                    TextEmbeddingCache.save(caption, mask_cpu, te2_dir, hint)

                pct = int((i + 1) / encode_total * 100)
                if pct % 10 == 0 or (i + 1) == encode_total:
                    print(f"[STATUS:Caching Text Embeddings ({pct}%)]", flush=True)

        self.logger.info(
            "text_embedding_cache_complete",
            cached=len(self.text_cache), newly_encoded=encode_total,
        )

    # -- Text Encoding --

    def encode_text(
        self, captions: list[str], dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(text_bth, attention_mask)`` -- a tuple consumed by ``forward_pass``.

        The base pipeline passes our return value opaquely to
        ``forward_pass`` which knows to unpack the tuple and derive
        ``text_lens`` from the mask.
        """
        if self.config.get("cache_text_embeddings", True):
            return self._get_cached_text_embeddings(captions, dtype)
        return self._encode_text_direct(captions, dtype)

    def _encode_text_direct(
        self, captions: list[str], dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode captions through the driver and return raw tensors."""
        output = self.driver.encode_text(captions, dtype)
        return output.embeddings, output.attention_mask

    def _get_cached_text_embeddings(
        self, captions: list[str], dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode on first encounter, reuse thereafter (per-caption granularity).

        Each cached entry is ``(embedding [T_i, D], mask [T_i])`` on CPU.
        At retrieval we right-pad to the longest in-batch length so the
        transformer receives a clean ``[B, Tmax, D]`` + ``[B, Tmax]`` pair.
        """
        uncached: list[tuple[int, str]] = []
        for i, cap in enumerate(captions):
            if cap not in self.text_cache:
                uncached.append((i, cap))

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

            for _, cap in uncached:
                single_emb, single_mask = self._encode_text_direct([cap], dtype)
                self.text_cache[cap] = (
                    single_emb.squeeze(0).cpu(),
                    single_mask.squeeze(0).cpu(),
                )

            if te_was_offloaded:
                self.text_encoder.to("cpu")
                torch.cuda.empty_cache()

            self.logger.debug(
                "text_embeddings_cached",
                new=len(uncached), total=len(self.text_cache),
            )
        elif uncached:
            raise RuntimeError(
                "Text encoder was unloaded but encountered uncached caption(s): "
                + ", ".join(cap[:50] for _, cap in uncached),
            )

        # Right-pad cached entries to the longest in-batch length.
        entries = [self.text_cache[cap] for cap in captions]
        per_lens = [int(e[1].sum().item()) for e in entries]
        t_max = max((e[0].shape[0] for e in entries), default=0)
        text_in_dim = entries[0][0].shape[-1] if entries else self.driver.text_in_dim
        batch_size = len(entries)

        text_bth = torch.zeros(
            (batch_size, t_max, text_in_dim),
            device=self.device, dtype=dtype,
        )
        attention_mask = torch.zeros(
            (batch_size, t_max), device=self.device, dtype=torch.long,
        )
        for i, ((emb, mask), n) in enumerate(zip(entries, per_lens)):
            text_bth[i, :emb.shape[0], :] = emb.to(self.device, dtype=dtype)
            attention_mask[i, :n] = 1

        return text_bth, attention_mask
