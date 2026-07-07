"""DreamLiteTrainer — family-specific trainer for DreamLite (Base / Mobile).

All shared logic (optimizer, EMA, gradient accumulation, noise offset,
checkpointing, signals, logging) lives in ``GenericTrainingPipeline``.
This module implements the DreamLite-specific behaviour and carries the
historical C1–C4 override trio:

- ``encode_text`` returns a ``(embeddings, attention_mask)`` TUPLE that
  ``driver.forward_pass`` unpacks (C1/C2);
- ``_update_primary_model`` also syncs ``self.driver.model`` so the
  PEFT-wrapped model is in the forward graph (C3);
- the read-only ``transformer`` property delegates to the driver's model
  so sampler code never sees ``None`` (C4).

Prompt-prefix convention (mirrors ``DreamLitePipeline.__call__``, which
encodes ``[negative_prompt, "[Generate]: " + prompt]``): every POSITIVE
text — dataset captions and sample prompts — goes through
:meth:`encode_text`, which applies the ``"[Generate]: "`` prefix before
encoding/caching; CFG negatives go through :meth:`encode_uncond_text`,
which does not. Cache keys carry the transformation, so an identical
string used as caption and as negative never collides.
"""

from __future__ import annotations

import os

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline

logger = structlog.get_logger(__name__)


class DreamLiteTrainer(GenericTrainingPipeline):
    """DreamLite LoRA trainer.

    ~0.39B DreamLite **U-Net** (GQA/MQA + qk_norm + depthwise-separable
    convs — NOT a DiT) with a Qwen3-VL text encoder (hidden 2048),
    AutoencoderTiny VAE, and flow-matching noise schedule. Base supports
    CFG with negative prompts; Mobile is CFG-distilled (4 steps).
    """

    # -- Setup --

    def _setup_family(self) -> None:
        """Initialize DreamLite-specific loader and driver.

        All family-specific classes are lazy-imported here so that registry
        discovery (which just imports this module) never trips on missing
        later-task modules.
        """
        from .loader import DreamLiteLoader  # noqa: PLC0415
        from .driver import DreamLiteDriver  # noqa: PLC0415

        self.loader = DreamLiteLoader(self.device)
        self.driver = DreamLiteDriver(self.definition, self.device)

    def _create_sampler(self):
        """Create a DreamLiteSampler if sampling is configured."""
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import DreamLiteSampler  # noqa: PLC0415

            return DreamLiteSampler(self)
        return None

    # -- Component Assignment --

    def _assign_components(self) -> None:
        """Wire components via driver + set the primary-model alias."""
        super()._assign_components()
        self.model = self.driver.model

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep self.model and driver.model in sync after PEFT/quant wrapping.

        The base class updates ``self.components["unet"]`` and ``self.model``
        (if the attribute exists), but does NOT reach into
        ``self.driver.model``. Without this override the PEFT-wrapped model
        is absent from the driver's forward graph and the optimizer's
        trainable-param list is empty (C3).
        """
        super()._update_primary_model(new_model)
        self.model = new_model
        # Sync the driver's primary model reference so forward_pass uses the
        # LoRA-wrapped model, not the original frozen weights.
        self.driver.model = new_model

    @property
    def transformer(self) -> torch.nn.Module | None:
        """Alias for sampler compatibility.

        DreamLiteDriver stores its primary model on ``self.driver.model``
        (not ``.transformer`` — and for this family the primary is a U-Net),
        so the base ``_assign_components`` loop would set
        ``self.transformer = None``. This property delegates directly to
        the driver so ``trainer.transformer`` never goes stale (C4). The
        sampler consumes this alias for device/dtype introspection.
        """
        return self.driver.model if self.driver is not None else None

    # -- Prompt-prefix helper --

    @staticmethod
    def _positive_key(text: str) -> str:
        """Cache key / encode string for a POSITIVE prompt or caption.

        Mirrors ``DreamLitePipeline.__call__``'s
        ``prompt_str = f"[Generate]: {prompt}"``.
        """
        from .driver import GENERATE_PREFIX  # noqa: PLC0415

        return f"{GENERATE_PREFIX}{text}"

    # -- Disk-backed TE Pre-caching --

    def _sample_prompt_texts(self) -> list[str]:
        """Expanded sample-prompt strings to pre-cache (RAW, un-prefixed).

        Mirrors the sampler's wildcard expansion (GenericSamplingPipeline
        calls ``_expand_wildcards`` → ``expand_prompt_wildcards`` before
        ``encode_prompt``) so the string matches what the sampler passes to
        :meth:`encode_text` (which applies the prefix itself).
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

        Layout (mirrors krea2 / ovis_image):
        - ``embeddings/{te_quant}/te1`` stores embedding tensors ``[L, D]``
        - ``embeddings/{te_quant}/te2`` stores attention masks ``[L]``

        Cache keys: dataset captions and sample prompts are warmed under
        their PREFIXED ``"[Generate]: …"`` key (the string actually
        encoded); the negative prompt is warmed RAW — exactly the strings
        :meth:`encode_text` / :meth:`encode_uncond_text` will request, so
        the sampler runs fully cache-served after the TE is offloaded.
        """
        if not self.config.get("cache_text_embeddings", True):
            return
        if self.text_encoder is None:
            return

        from app.engine.components.text_embeddings import (  # noqa: PLC0415
            TextEmbeddingCache,
        )

        te_cache_dirs = self._resolve_te_cache_dirs()
        te_quant = self.config.get("te_quantization", "none")
        te1_dir = (
            os.path.join(te_cache_dirs[0], "embeddings", te_quant, "te1")
            if te_cache_dirs
            else ""
        )
        te2_dir = (
            os.path.join(te_cache_dirs[0], "embeddings", te_quant, "te2")
            if te_cache_dirs
            else ""
        )

        # Dataset captions encode (and cache) under their PREFIXED key.
        caption_hints = {
            self._positive_key(cap): hint
            for cap, hint in self._build_caption_hints().items()
        }

        disk_loaded = 0
        need_encode: list[tuple[str, str]] = []

        for key, hint in caption_hints.items():
            if key in self.text_cache:
                continue
            if te1_dir and te2_dir:
                emb_tensor = TextEmbeddingCache.load(key, te1_dir, hint)
                mask_tensor = TextEmbeddingCache.load(key, te2_dir, hint)
                if emb_tensor is not None and mask_tensor is not None:
                    self.text_cache[key] = (emb_tensor, mask_tensor)
                    disk_loaded += 1
                    continue
            need_encode.append((key, hint))

        # Warm sample (prefixed) + negative (raw) prompts so the TE stays
        # offloaded during sampling.
        sample_keys = [
            self._positive_key(sp) for sp in self._sample_prompt_texts()
        ]
        queued = {key for key, _ in need_encode}
        for key in sample_keys:
            if key not in self.text_cache and key not in queued:
                need_encode.append((key, ""))
                queued.add(key)
        if sample_keys:
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
                cached=len(self.text_cache),
                source="disk",
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
                batch_keys = [key for key, _ in batch_items]

                emb_batch, mask_batch = self._encode_text_direct(
                    batch_keys,
                    dtype,
                )

                for j, (key, hint) in enumerate(batch_items):
                    emb_cpu = emb_batch[j].cpu()
                    mask_cpu = mask_batch[j].cpu()
                    self.text_cache[key] = (emb_cpu, mask_cpu)
                    if te1_dir:
                        TextEmbeddingCache.save(key, emb_cpu, te1_dir, hint)
                    if te2_dir:
                        TextEmbeddingCache.save(key, mask_cpu, te2_dir, hint)

                pct = int(
                    min(i + batch_size, encode_total) / encode_total * 100,
                )
                if pct % 10 == 0 or (i + batch_size) >= encode_total:
                    if getattr(self, "_log_writer", None):
                        self._log_writer.status(
                            f"Caching Text Embeddings ({pct}%)",
                        )

        self.logger.info(
            "text_embedding_cache_complete",
            cached=len(self.text_cache),
            newly_encoded=encode_total,
        )

    # -- Text Encoding --

    def encode_text(
        self,
        captions: list[str],
        dtype: torch.dtype,
        batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode POSITIVE captions/prompts (``"[Generate]: "`` applied).

        ``batch`` is accepted for hook compatibility and ignored here.

        Returns a ``(embeddings, attention_mask)`` TUPLE. The base pipeline
        passes this opaquely to ``forward_pass()`` which passes it to
        ``driver.forward_pass()`` that unpacks the tuple (C1/C2).

        Returns:
            (text_embeddings ``[B, max_seq, 2048]``, mask ``[B, max_seq]``).
        """
        keys = [self._positive_key(c) for c in captions]
        return self._encode_keys(keys, dtype)

    def encode_uncond_text(
        self,
        texts: list[str],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode CFG NEGATIVE prompts — RAW, no ``"[Generate]: "`` prefix.

        Mirrors the pipeline's ``prompts=[negative_prompt, …]`` where only
        the positive carries the prefix. Used by the sampler's CFG branch.
        """
        return self._encode_keys(list(texts), dtype)

    def _encode_keys(
        self,
        keys: list[str],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cache-aware encode of already-transformed strings."""
        if self.config.get("cache_text_embeddings", True):
            return self._get_cached_text_embeddings(keys, dtype)
        return self._encode_text_direct(keys, dtype)

    def _encode_text_direct(
        self,
        texts: list[str],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode already-transformed strings directly via the driver.

        Delegates to ``driver.encode_text`` (single source of truth for the
        pipeline-identical template/drop encoding) and unwraps the
        ``TextEncoderOutput`` to the tuple contract.
        """
        out = self.driver.encode_text(texts, dtype)
        return out.embeddings, out.attention_mask

    def _get_cached_text_embeddings(
        self,
        keys: list[str],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode on first encounter, reuse thereafter.

        Cache entries are per-key CPU tuples ``(emb [L, D], mask [L])``,
        stacked back to ``([B, L, D], [B, L])`` on retrieval (fixed
        ``max_sequence_length`` L — see driver.encode_text).
        """
        emb_results: list[torch.Tensor] = []
        mask_results: list[torch.Tensor] = []
        uncached: list[tuple[int, str]] = []

        for i, key in enumerate(keys):
            if key not in self.text_cache:
                uncached.append((i, key))

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

            for _, key in uncached:
                single_emb, single_mask = self._encode_text_direct(
                    [key],
                    dtype,
                )
                self.text_cache[key] = (
                    single_emb.squeeze(0).cpu(),
                    single_mask.squeeze(0).cpu(),
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
                "Text encoder was unloaded but encountered uncached caption(s): "
                + ", ".join(key[:50] for _, key in uncached)
            )

        for key in keys:
            cached_emb, cached_mask = self.text_cache[key]
            emb_results.append(cached_emb.to(self.device, dtype=dtype))
            mask_results.append(cached_mask.to(self.device))

        return torch.stack(emb_results, dim=0), torch.stack(mask_results, dim=0)
