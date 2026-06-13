"""FLUX.2 Trainer — family-specific hooks for the generic training pipeline.

Implements FLUX.2-specific behaviour:
- Single text encoder: Qwen3 with layer concatenation
- Flow matching with configurable timestep sampling
- Lazy text embedding caching with optional TE unloading
- Latent packing via ``pack_latents`` (4-col IDs for diffusers)
- No guidance embed (Klein: guidance_embeds=false)
- ``Flux2Transformer2DModel`` forward pass (no pooled_projections)

Timestep sampling strategies derived from ostris/ai-toolkit (MIT License).
"""

import os
from typing import Any

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline
from .driver import Flux2Driver
from .loader import Flux2Loader
from .saver import Flux2Saver

logger = structlog.get_logger(__name__)


class Flux2Trainer(GenericTrainingPipeline):
    """FLUX.2 (Klein / Dev) LoRA trainer."""

    # ── Setup ────────────────────────────────────────────────────────────

    def _setup_family(self) -> None:
        """Initialize Flux2-specific loader, saver, driver, and caches."""
        self.driver = Flux2Driver(self.definition, self.device)
        self.loader = Flux2Loader(self.device)
        self.saver = Flux2Saver()

        # Flux2 patchifies latents (2× down per spatial dim) before the
        # transformer.  Tell the flux_shift sampler so it computes
        # seq_len = (H/2)*(W/2) instead of H*W.
        self.config.setdefault("flux_shift_patchify_factor", 2)

    def _create_sampler(self):
        """Create a Flux2Sampler if sampling is configured."""
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import Flux2Sampler
            return Flux2Sampler(self)
        return None

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep self.transformer in sync after PEFT/quantization wrapping."""
        self.transformer = new_model
        self.components["unet"] = new_model
        # Also update driver's reference
        self.driver.transformer = new_model


    # ── Staged VRAM Management ───────────────────────────────────────────


    def _pre_cache_text_embeddings(self) -> None:
        """Warm text embedding cache from disk + encode missing.

        1. Build the full set of captions (training, dropout variants, sampling).
        2. For each caption, try to load from disk cache first.
        3. Encode only truly uncached captions on GPU.
        4. Save newly encoded embeddings to disk for future runs.
        """
        if not self.config.get("cache_text_embeddings", True):
            return
        if self.text_encoder is None:
            return

        from app.engine.components.text_embeddings import TextEmbeddingCache

        # Resolve the TE disk cache directory (te1 for single-TE Flux2)
        te_cache_dirs = self._resolve_te_cache_dirs()
        # Include TE quantization scheme so FP8 / bf16 embeddings don't collide
        te_quant = self.config.get("te_quantization", "none")
        # Use the first dir for now — multi-dataset case handled by
        # pre-caching to a single canonical directory.
        te_cache_dir = os.path.join(te_cache_dirs[0], "embeddings", te_quant, "te1") if te_cache_dirs else ""

        # ── Build full caption set (shared base class logic) ────────────────
        caption_hints = self._build_caption_hints()

        # ── Phase 1: Load from disk cache ─────────────────────────────────
        disk_loaded = 0
        need_encode: list[tuple[str, str]] = []  # (caption, hint)

        for caption, hint in caption_hints.items():
            if caption in self.text_cache:
                continue  # Already in memory (e.g. restored from checkpoint)
            if te_cache_dir:
                tensor = TextEmbeddingCache.load(caption, te_cache_dir, hint)
                if tensor is not None:
                    self.text_cache[caption] = tensor
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
            if getattr(self, "_log_writer", None):
                self._log_writer.status("TE Cache Loaded from Disk")
            self.logger.info(
                "text_embedding_cache_complete",
                cached_embeddings=len(self.text_cache),
                source="disk",
            )
            return

        # ── Phase 2: Encode missing captions on GPU ───────────────────────
        # Batch-encode directly via driver (avoids per-caption forward pass
        # that _get_cached_text_embeddings would do).
        if getattr(self, "_log_writer", None):
            self._log_writer.status("Caching Text Embeddings (0%)")
        encode_total = len(need_encode)
        batch_size = 4
        dtype = self._resolve_loading_dtype()

        with torch.no_grad():
            for i in range(0, encode_total, batch_size):
                batch_items = need_encode[i : i + batch_size]
                batch_caps = [cap for cap, _ in batch_items]

                # Single batched forward pass through the TE
                output = self.driver.encode_text(batch_caps, dtype)
                emb_batch = output.embeddings if hasattr(output, 'embeddings') else output

                # Store each caption's embedding in the in-memory cache
                for j, (cap, hint) in enumerate(batch_items):
                    emb_cpu = emb_batch[j : j + 1].cpu()
                    self.text_cache[cap] = emb_cpu
                    if te_cache_dir:
                        TextEmbeddingCache.save(cap, emb_cpu, te_cache_dir, hint)

                pct = round(min(i + batch_size, encode_total) / encode_total * 100)
                if getattr(self, "_log_writer", None):
                    self._log_writer.status(f"Caching Text Embeddings ({pct}%)")
                if (i + batch_size) % 20 == 0 or (i + batch_size) >= encode_total:
                    self.logger.info(
                        "te_cache_progress",
                        cached=min(i + batch_size, encode_total),
                        total=encode_total,
                    )

        self.logger.info(
            "text_embedding_cache_complete",
            cached_embeddings=len(self.text_cache),
            newly_encoded=encode_total,
        )

    def _offload_text_encoders(self) -> None:
        """Offload or unload text encoder + tokenizer.

        Delegates TE handling (including ``self.components`` cleanup)
        to the base class, then cleans up the tokenizer if unloading.
        """
        unloading = self.config.get("unload_text_encoder", False)
        super()._offload_text_encoders()
        # Base class handles TE + components dict.  Clean tokenizer too.
        if unloading and not hasattr(self, "_tok_cleaned"):
            self.tokenizer = None
            self.components.pop("tokenizer", None)
            self._tok_cleaned = True

    # ── Text Encoding ────────────────────────────────────────────────────

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, batch: dict | None = None
    ) -> Any:
        """Text encoding with lazy caching.

        Uses driver for fresh encoding, wraps with in-memory cache.
        Always returns a raw tensor ``[B, L, D*N]``.
        """
        if self.config.get("cache_text_embeddings", True):
            return self._get_cached_text_embeddings(captions, dtype)
        # Non-cached path: unwrap TextEncoderOutput → raw tensor
        output = self.driver.encode_text(captions, dtype)
        return output.embeddings if hasattr(output, 'embeddings') else output


    def _get_cached_text_embeddings(
        self, captions: list[str], dtype: torch.dtype
    ) -> Any:
        """Encode on first encounter; reuse thereafter."""
        results: list[torch.Tensor] = []
        uncached_caps: list[tuple[int, str]] = []

        for i, cap in enumerate(captions):
            if cap in self.text_cache:
                results.append(self.text_cache[cap])
            else:
                uncached_caps.append((i, cap))
                results.append(None)  # placeholder

        if uncached_caps and self.text_encoder is not None:
            # Guard: if TE was offloaded to CPU, temporarily move back
            te_device = next(self.text_encoder.parameters()).device
            te_was_offloaded = te_device != self.device
            if te_was_offloaded:
                self.logger.warning(
                    "te_cache_miss_after_offload",
                    count=len(uncached_caps),
                    hint="pre-caching should have covered all captions",
                )
                self.text_encoder.to(self.device)

            for orig_idx, cap in uncached_caps:
                emb = self.driver.encode_text([cap], dtype)
                # TextEncoderOutput → extract embeddings tensor
                emb_tensor = emb.embeddings if hasattr(emb, 'embeddings') else emb
                self.text_cache[cap] = emb_tensor.cpu()
                results[orig_idx] = emb_tensor.cpu()

            if te_was_offloaded:
                self.text_encoder.to("cpu")
                torch.cuda.empty_cache()

            self.logger.debug(
                "text_embeddings_cached",
                new=len(uncached_caps),
                total=len(self.text_cache),
            )
        elif uncached_caps:
            # TE was fully unloaded — fill with zeros
            te_max_len = getattr(self.driver, 'te_max_length', 512)
            concat_layers = getattr(self.driver, 'te_concat_layers', 3)
            for orig_idx, cap in uncached_caps:
                self.logger.error("text_encoder_unavailable", caption=cap[:50])
                dummy = torch.zeros(
                    1, te_max_len,
                    4096 * concat_layers,
                    dtype=dtype,
                )
                results[orig_idx] = dummy

        return torch.cat(
            [r.to(self.device, dtype=dtype) for r in results], dim=0
        )

    def compute_loss_weight(self, timesteps: torch.Tensor) -> torch.Tensor | None:
        """Per-timestep loss weights for RADC mode.

        Uses the same dynamic center/width as the sampler at the
        current training step, ensuring consistent weighting.

        Returns ``None`` for non-RADC modes (uniform weighting).
        """
        if self.config.get("timestep_sampling") != "radc":
            return None

        from app.engine.strategies.timestep_sampling import (
            _make_radc_pdf, _radc_center,
        )
        max_steps = getattr(self, "max_train_steps", 1)
        progress = getattr(self, "global_step", 0) / max(max_steps, 1)
        center = _radc_center(self.config, progress, latents=None)
        width = float(self.config.get("radc_width", 0.5))
        pdf = _make_radc_pdf(center, width)
        indices = timesteps.long().clamp(1, 1000) - 1
        return pdf[indices.cpu()].to(timesteps.device, dtype=timesteps.dtype)
