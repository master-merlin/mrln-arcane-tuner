"""Krea2 Trainer — family-specific trainer for Krea-2 (Phase 2).

Phase 1 vendored the transformer + conditioning helpers and scaffolded the
family / loader / definitions.  Phase 2 (this commit) wires in Krea2Driver.

Driver, sampler, and saver are lazy-imported inside ``_setup_family`` so
that registry discovery (which merely imports this module) never trips on a
missing Phase 3+ module.
"""

from __future__ import annotations

import os
from typing import Any

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline

logger = structlog.get_logger(__name__)


class Krea2Trainer(GenericTrainingPipeline):
    """Krea-2 LoRA trainer.

    28-layer MMDiT with Qwen3-VL 12-layer stacked text encoder,
    AutoencoderKLQwenImage VAE, and flow-matching noise schedule.
    """

    # -- Setup --

    def _setup_family(self) -> None:
        """Initialize Krea2-specific loader and driver.

        All family-specific classes are lazy-imported here so that registry
        discovery (which just imports this module) never trips on missing
        Phase 3+ modules.
        """
        from .loader import Krea2Loader  # noqa: PLC0415
        from .driver import Krea2Driver  # noqa: PLC0415

        self.loader = Krea2Loader(self.device)
        self.driver = Krea2Driver(self.definition, self.device)

    def _create_sampler(self):
        """Create a Krea2Sampler if sampling is configured."""
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import Krea2Sampler  # noqa: PLC0415
            return Krea2Sampler(self)
        return None

    # -- Component Assignment --

    def _assign_components(self) -> None:
        """Wire components via driver + set Krea-2-specific aliases."""
        super()._assign_components()
        self.model = self.driver.model

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep self.model and driver.model in sync after PEFT/quantization wrapping.

        The base class updates ``self.components["unet"]`` and ``self.model``
        (if the attribute exists), but does NOT reach into ``self.driver.model``.
        Without this override the PEFT-wrapped model is absent from the driver's
        forward graph and the optimizer's trainable-param list is empty (C3).
        """
        super()._update_primary_model(new_model)
        self.model = new_model
        # Sync the driver's primary model reference so forward_pass uses the
        # LoRA-wrapped model, not the original frozen weights.
        self.driver.model = new_model

    @property
    def transformer(self) -> torch.nn.Module | None:
        """Alias for sampler compatibility.

        Krea2Driver stores its primary model on ``self.driver.model`` (not
        ``self.driver.transformer``).  The base ``_assign_components`` loop
        therefore sets ``self.transformer = None``.  This property short-circuits
        that by delegating directly to the driver, so sampler code that calls
        ``trainer.transformer`` or ``next(self.pipeline.transformer.parameters())``
        never hits None (C4).
        """
        return self.driver.model if self.driver is not None else None

    # -- Disk-backed TE Pre-caching --

    def _sample_prompt_texts(self) -> list[str]:
        """Expanded sample-prompt strings to pre-cache.

        Mirrors the sampler's wildcard expansion (GenericSamplingPipeline calls
        ``_expand_wildcards`` → ``expand_prompt_wildcards`` before ``encode_prompt``)
        so the cache key matches the exact string the sampler requests via
        :meth:`encode_text`.
        """
        from app.engine.core.sampling import expand_prompt_wildcards

        texts: list[str] = []
        for sp in self.config.get("sample_prompts", []) or []:
            raw = (
                sp.get("prompt", "")
                if isinstance(sp, dict)
                else getattr(sp, "prompt", "")
            )
            if raw:
                expanded = expand_prompt_wildcards(raw, self.config)
                if expanded not in texts:
                    texts.append(expanded)
        return texts

    def _pre_cache_text_embeddings(self) -> None:
        """Warm text embedding cache from disk + encode missing.

        Mirrors ``QwenImageTrainer._pre_cache_text_embeddings``:
        - te1/ stores embedding tensors [L, 12, 2560]
        - te2/ stores attention mask tensors [L]

        The expanded SAMPLE prompts and negative prompt are also warmed here:
        the sampler runs after the TE is offloaded, so it must serve all prompts
        from ``self.text_cache`` — without this, sampling causes a VRAM spike by
        reloading the 8 GB Qwen3-VL encoder from CPU.
        """
        if not self.config.get("cache_text_embeddings", True):
            return
        if self.text_encoder is None:
            return

        from app.engine.components.text_embeddings import TextEmbeddingCache

        te_cache_dirs = self._resolve_te_cache_dirs()
        te_quant = self.config.get("te_quantization", "none")
        te1_dir = os.path.join(te_cache_dirs[0], "embeddings", te_quant, "te1") if te_cache_dirs else ""
        te2_dir = os.path.join(te_cache_dirs[0], "embeddings", te_quant, "te2") if te_cache_dirs else ""

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

        # Warm sample + negative prompts so the TE stays offloaded during sampling.
        # The sampler expands wildcards identically via _expand_wildcards before
        # calling encode_text, so the key produced by _sample_prompt_texts matches.
        sample_texts = self._sample_prompt_texts()
        queued = {cap for cap, _ in need_encode}
        for sp in sample_texts:
            if sp not in self.text_cache and sp not in queued:
                need_encode.append((sp, ""))
                queued.add(sp)
        if sample_texts:
            neg = str(self.config.get("sample_negative_prompt", "") or "")
            if neg not in self.text_cache and neg not in queued:
                need_encode.append((neg, ""))

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
                cached=len(self.text_cache), source="disk",
            )
            return

        if getattr(self, "_log_writer", None):
            self._log_writer.status("Caching Text Embeddings (0%)")
        encode_total = len(need_encode)
        batch_size = 4
        dtype = self._resolve_loading_dtype()

        with torch.no_grad():
            for i in range(0, encode_total, batch_size):
                batch_items = need_encode[i : i + batch_size]
                batch_caps = [cap for cap, _ in batch_items]

                emb_batch, mask_batch = self._encode_text_direct(batch_caps, dtype)

                for j, (cap, hint) in enumerate(batch_items):
                    emb_cpu = emb_batch[j].cpu()
                    mask_cpu = mask_batch[j].cpu()
                    self.text_cache[cap] = (emb_cpu, mask_cpu)
                    if te1_dir:
                        TextEmbeddingCache.save(cap, emb_cpu, te1_dir, hint)
                    if te2_dir:
                        TextEmbeddingCache.save(cap, mask_cpu, te2_dir, hint)

                pct = int(min(i + batch_size, encode_total) / encode_total * 100)
                if pct % 10 == 0 or (i + batch_size) >= encode_total:
                    if getattr(self, "_log_writer", None):
                        self._log_writer.status(f"Caching Text Embeddings ({pct}%)")

        self.logger.info(
            "text_embedding_cache_complete",
            cached=len(self.text_cache), newly_encoded=encode_total,
        )

    # -- Text Encoding --

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode captions through Qwen3-VL with 12-layer stacking.

        ``batch`` is accepted for hook compatibility and ignored here.

        Returns a ``(embeddings, attention_mask)`` tuple.  The base pipeline
        passes this opaquely to ``forward_pass()`` which passes it to
        ``driver.forward_pass()`` that unpacks the tuple (C1/C2 fix).

        Args:
            captions: Processed captions.
            dtype: Target dtype.
            batch: Ignored (kept for paired-edit hook compatibility).

        Returns:
            (text_embeddings [B, L, 12, 2560], attention_mask [B, L]).
        """
        if self.config.get("cache_text_embeddings", True):
            return self._get_cached_text_embeddings(captions, dtype)

        return self._encode_text_direct(captions, dtype)

    def _encode_text_direct(
        self, captions: list[str], dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode captions directly via driver (no cache).

        Delegates to ``driver.encode_text`` which returns a
        ``TextEncoderOutput`` with 4-D embeddings ``[B, L, 12, 2560]``.
        Unwraps to the ``(embeddings, attention_mask)`` tuple contract.

        Returns:
            (embeddings [B, L, 12, 2560], attention_mask [B, L]).
        """
        out = self.driver.encode_text(captions, dtype)
        return out.embeddings, out.attention_mask

    def _get_cached_text_embeddings(
        self, captions: list[str], dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode on first encounter, reuse thereafter.

        Mirrors ``QwenImageTrainer._get_cached_text_embeddings``.

        Returns:
            (text_embeddings [B, L, 12, 2560], attention_mask [B, L]).
        """
        emb_results: list[torch.Tensor] = []
        mask_results: list[torch.Tensor] = []
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
                "text_embeddings_cached", new=len(uncached), total=len(self.text_cache),
            )
        elif uncached:
            raise RuntimeError(
                "Text encoder was unloaded but encountered uncached caption(s): "
                + ", ".join(cap[:50] for _, cap in uncached)
            )

        for cap in captions:
            cached_emb, cached_mask = self.text_cache[cap]
            emb_results.append(cached_emb.to(self.device, dtype=dtype))
            mask_results.append(cached_mask.to(self.device))

        return torch.stack(emb_results, dim=0), torch.stack(mask_results, dim=0)
