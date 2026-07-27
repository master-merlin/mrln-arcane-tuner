"""LongCat-Image Trainer -- family-specific hooks for the generic pipeline.

All shared logic (optimizer, EMA, gradient accumulation, noise offset,
checkpointing, signals, logging) lives in ``GenericTrainingPipeline``.
This module implements LongCat-Image-specific behaviour:
- Single Qwen2.5-VL text encoder (text-only mode, LongCat prompt template)
- Flow matching with configurable timestep sampling
- LongCatImageTransformer2DModel forward (pack 2×2 → transformer → unpack)

The trainer override trio (``encode_text`` tuple contract,
``_update_primary_model`` driver sync, read-only ``transformer`` property)
is MANDATORY — dropping any of them is the historical krea2 C1–C4 bug class
(pinned by backend/tests/engine/families/test_trainer_seam_contract.py).
"""

import os

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline
from .driver import TOKENIZER_MAX_LENGTH, LongCatImageDriver, te_template_fingerprint
from .loader import LongCatImageLoader
from .saver import LongCatImageSaver

logger = structlog.get_logger(__name__)

# Disk-cache key template identity (the qwen_image/boogu_image precedent).
# ``TextEmbeddingCache.caption_to_filename`` hashes ONLY the string it is
# given; passing the raw caption meant a future edit to the driver's prefix/
# suffix chat template, quotation-aware tokenization, OR effective
# ``te.max_length`` (truncation/padding length) would silently reuse
# embeddings encoded under the OLD template/length. Baking
# ``te_template_fingerprint()`` into the hashed string makes any of those
# changes always produce a fresh on-disk filename. The IN-MEMORY
# ``self.text_cache`` stays keyed by the raw caption (the krea2/ernie/
# ideogram4 convention the cross-family seam contract expects).
#
# ``_disk_cache_key`` is a TRAINER INSTANCE METHOD (not a module-level
# function, and not a ``@staticmethod``) because it must fold in
# ``self.max_length`` — the EFFECTIVE, possibly per-definition-overridden
# ``te.max_length`` for this run (kept in lock-step with the driver by
# ``_assign_components``) — which by definition cannot be known by a
# function with no access to instance state. This is the same shape chosen
# for ``DreamLiteTrainer._disk_cache_key`` (dropped from ``@staticmethod``
# for the identical reason), normalizing the convention across both
# families.
_TE_TEMPLATE_VERSION = "v1"


class LongCatImageTrainer(GenericTrainingPipeline):
    """LongCat-Image LoRA trainer.

    ~11.9B Flux-like DiT (19 double + 38 single blocks) with a single
    Qwen2.5-VL text encoder and flow-matching noise schedule.
    """

    # -- Setup --

    def _setup_family(self) -> None:
        """Initialize LongCat-Image loader, saver, driver, and caches."""
        self.driver = LongCatImageDriver(self.definition, self.device)
        self.loader = LongCatImageLoader(self.device)
        self.saver = LongCatImageSaver()

    def _create_sampler(self):
        """Create a LongCatImageSampler if sampling is configured."""
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import LongCatImageSampler
            return LongCatImageSampler(self)
        return None

    # -- Component Assignment --

    def _assign_components(self) -> None:
        """Wire components via driver + set LongCat-specific aliases."""
        super()._assign_components()
        self.model = self.components["unet"]
        # Keep the driver's context window in lock-step with the trainer.
        arch = getattr(self.definition, "architecture_params", {}) or {}
        self.max_length = int(arch.get("te.max_length", TOKENIZER_MAX_LENGTH))
        self.driver.max_length = self.max_length

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep self.model in sync after PEFT/quantization wrapping."""
        self.model = new_model
        self.components["unet"] = new_model
        # Also update driver's reference
        self.driver.model = new_model

    @property
    def transformer(self) -> torch.nn.Module:
        """Alias for sampler compatibility (sampler accesses .transformer)."""
        return self.model

    def _disk_cache_key(self, caption: str) -> str:
        """Compose the string hashed by ``TextEmbeddingCache.caption_to_filename``.

        Baking the template fingerprint into the hashed string (instead of
        the raw caption) means a future template OR effective-max_length
        change produces a DIFFERENT on-disk filename for the same caption
        text, instead of silently reusing a stale embedding encoded under
        the old template/length. Passes ``self.max_length`` — the EFFECTIVE
        ``te.max_length`` resolved for this run (synced from the driver in
        ``_assign_components``, which itself resolves any per-definition
        ``te.max_length`` override) — into ``te_template_fingerprint`` so a
        definition override is captured, not just the module default.
        """
        template_id = (
            f"longcat_image/quotation_chat_template/"
            f"{_TE_TEMPLATE_VERSION}/{te_template_fingerprint(self.max_length)}"
        )
        return f"{template_id}::{caption}"

    # -- Disk-backed TE Pre-caching --

    def _sample_prompt_texts(self) -> list[str]:
        """Expanded sample-prompt strings to pre-cache.

        Mirrors the sampler's wildcard expansion (GenericSamplingPipeline
        calls ``_expand_wildcards`` → ``expand_prompt_wildcards`` before
        ``encode_prompt``) so the cache key matches the exact string the
        sampler requests via :meth:`encode_text`.
        """
        from app.engine.core.sampling import (  # noqa: PLC0415
            expand_prompt_wildcards,
        )

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

        LongCat-Image caches (embedding, mask) tuples:
        - embeddings/{te_quant}/te1/ stores embedding tensors
        - embeddings/{te_quant}/te2/ stores attention mask tensors

        The expanded SAMPLE prompts and negative prompt are also warmed
        here: the sampler runs after the TE is offloaded, so it must serve
        all prompts from ``self.text_cache`` — without this, sampling
        causes a VRAM spike (offload) or a hard error (unload).
        """
        if not self.config.get("cache_text_embeddings", True):
            return
        if self.text_encoder is None:
            return

        from app.engine.components.text_embeddings import TextEmbeddingCache

        te_cache_dirs = self._resolve_te_cache_dirs()
        # Include TE quantization scheme so FP8 / bf16 embeddings don't collide
        te_quant = self.config.get("te_quantization", "none")
        te1_dir = os.path.join(te_cache_dirs[0], "embeddings", te_quant, "te1") if te_cache_dirs else ""
        te2_dir = os.path.join(te_cache_dirs[0], "embeddings", te_quant, "te2") if te_cache_dirs else ""

        # -- Build full caption set (shared base class logic) --
        caption_hints = self._build_caption_hints()

        # -- Phase 1: Load from disk --
        disk_loaded = 0
        need_encode: list[tuple[str, str]] = []

        for caption, hint in caption_hints.items():
            if caption in self.text_cache:
                continue
            if te1_dir and te2_dir:
                emb_tensor = TextEmbeddingCache.load(
                    self._disk_cache_key(caption), te1_dir, hint
                )
                mask_tensor = TextEmbeddingCache.load(
                    self._disk_cache_key(caption), te2_dir, hint
                )
                if emb_tensor is not None and mask_tensor is not None:
                    self.text_cache[caption] = (emb_tensor, mask_tensor)
                    disk_loaded += 1
                    continue
            need_encode.append((caption, hint))

        # Warm sample + negative prompts so the TE stays offloaded during
        # sampling. The sampler expands wildcards identically before calling
        # encode_text, so the key produced by _sample_prompt_texts matches.
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

        # -- Phase 2: Encode missing (batched) --
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
                emb_batch, mask_batch = self._encode_text_direct(batch_caps, dtype)

                for j, (cap, hint) in enumerate(batch_items):
                    emb_cpu = emb_batch[j].cpu()
                    mask_cpu = mask_batch[j].cpu()
                    self.text_cache[cap] = (emb_cpu, mask_cpu)
                    if te1_dir:
                        TextEmbeddingCache.save(
                            self._disk_cache_key(cap), emb_cpu, te1_dir, hint
                        )
                    if te2_dir:
                        TextEmbeddingCache.save(
                            self._disk_cache_key(cap), mask_cpu, te2_dir, hint
                        )

                pct = int(min(i + batch_size, encode_total) / encode_total * 100)
                if pct % 10 == 0 or (i + batch_size) >= encode_total:
                    if getattr(self, "_log_writer", None):
                        self._log_writer.status(f"Caching Text Embeddings ({pct}%)")

        self.logger.info(
            "text_embedding_cache_complete",
            cached=len(self.text_cache), newly_encoded=encode_total,
        )

    # -- TE Offloading --

    # _offload_text_encoders — inherited from base class.
    # Base uses _get_text_encoders() to discover and offload/unload TEs
    # and properly cleans self.components to prevent stale references.

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode captions through Qwen2.5-VL (LongCat template).

        ``batch`` is accepted for hook compatibility and ignored.

        Returns a ``(embeddings, attention_mask)`` tuple.  The base
        pipeline passes this opaquely to ``forward_pass()`` which unpacks it.

        Args:
            captions: Processed captions.
            dtype: Target dtype.

        Returns:
            (text_embeddings [B, L, D], attention_mask [B, L]).
        """
        if self.config.get("cache_text_embeddings", True):
            return self._get_cached_text_embeddings(captions, dtype)

        return self._encode_text_direct(captions, dtype)

    def _encode_text_direct(
        self, captions: list[str], dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode captions directly via the driver (no cache).

        Delegates to ``driver.encode_text`` (single source of truth for the
        LongCat quotation-aware template encoding) and unwraps the
        ``TextEncoderOutput`` to the ``(embeddings, attention_mask)`` tuple
        contract the base pipeline hands opaquely to ``forward_pass``.
        """
        out = self.driver.encode_text(captions, dtype)
        return out.embeddings, out.attention_mask

    def _get_cached_text_embeddings(
        self, captions: list[str], dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode on first encounter, reuse thereafter.

        Caches both embeddings and attention masks.

        Returns:
            (text_embeddings [B, L, D], attention_mask [B, L]).
        """
        emb_results: list[torch.Tensor] = []
        mask_results: list[torch.Tensor] = []
        uncached: list[tuple[int, str]] = []

        for i, cap in enumerate(captions):
            if cap not in self.text_cache:
                uncached.append((i, cap))

        if uncached and self.text_encoder is not None:
            # Guard: if TE was offloaded to CPU, temporarily move back
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
                single_ctx, single_mask = self._encode_text_direct([cap], dtype)
                self.text_cache[cap] = (
                    single_ctx.squeeze(0).cpu(),
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

    # -- Forward Pass --
    # Delegated to ``LongCatImageDriver.forward_pass`` via the base
    # ``PipelineBaseMixin.forward_pass`` (pack → transformer → unpack).
