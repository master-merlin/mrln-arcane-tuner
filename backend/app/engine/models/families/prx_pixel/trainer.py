"""PRXPixelTrainer — family-specific trainer for pixel-space PRX.

All shared logic (optimizer, EMA, gradient accumulation, noise offset,
checkpointing, signals, logging) lives in ``GenericTrainingPipeline``.
This module implements the PRXPixel-specific behaviour:

- PIXEL PASSTHROUGH: installs the shared
  :class:`PixelPassthroughLatentManager` (no VAE — "latents" are the raw
  ``[B, 3, H, W]`` pixel batch) and no-ops the latent-cache hooks. Unlike
  hidream_o1 the loop's latents ARE consumed: the driver trains on them
  directly (x0 objective), so no ``_compute_step_loss`` override is
  needed — the base ``MSE(pred, target)`` with ``target == clean pixels``
  is exactly the recipe.
- The historical C1–C4 override trio:
  - ``encode_text`` returns a ``(embeddings, attention_mask)`` TUPLE that
    ``driver.forward_pass`` unpacks (C1/C2);
  - ``_update_primary_model`` also syncs ``self.driver.model`` so the
    PEFT-wrapped model is in the forward graph (C3);
  - the read-only ``transformer`` property delegates to the driver's model
    so sampler code never sees ``None`` (C4).
- Disk-backed TE pre-caching (te1 embeddings ``[B, 256, D]`` + te2 BOOL
  masks) — the pixel_transformer archetype keeps ``te_cache=True``
  because the Qwen3-VL TE is a real standalone component.
"""

from __future__ import annotations

import os

import structlog
import torch

from app.engine.components.pixel_latents import PixelPassthroughLatentManager
from app.engine.core.pipeline import GenericTrainingPipeline

logger = structlog.get_logger(__name__)


class PRXPixelTrainer(GenericTrainingPipeline):
    """PRXPixel LoRA trainer.

    ~7B pixel-space cross-attention DiT (24 PRXBlocks, fused QKV/KV
    projections, bottleneck img_in, resolution embeds) with a Qwen3-VL
    text backbone (hidden 2048, bool masks), NO VAE, x0-prediction
    flow-matching recipe (scaled noise ×2.0). Supports CFG with negative
    prompts.
    """

    # -- Setup --

    def _setup_family(self) -> None:
        """Initialize PRXPixel-specific loader, driver, and saver."""
        from .loader import PRXPixelLoader  # noqa: PLC0415
        from .driver import PRXPixelDriver  # noqa: PLC0415
        from .saver import PRXPixelSaver  # noqa: PLC0415

        self.loader = PRXPixelLoader(self.device)
        self.driver = PRXPixelDriver(self.definition, self.device)
        self.saver = PRXPixelSaver()

    def _create_sampler(self):
        """Create a PRXPixelSampler if sampling is configured."""
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import PRXPixelSampler  # noqa: PLC0415

            return PRXPixelSampler(self)
        return None

    # -- Pixel-space latent bypass --

    def _configure_managers(self, max_train_steps: int) -> None:
        """Install the shared pixel-passthrough LatentManager.

        The base creates ``LatentManager(vae=None, ...)`` which raises on
        ``encode_and_cache_batch``. The passthrough returns the pixel batch
        unchanged, so the loop's "latents" are the clean ``[-1, 1]`` pixels
        the x0 objective trains against.
        """
        super()._configure_managers(max_train_steps)
        # Replace after super() sets it — CheckpointManager setup stays
        # intact and only the LatentManager is substituted.
        self.latent_manager = PixelPassthroughLatentManager()
        logger.info("prx_pixel.trainer.pixel_passthrough_latent_manager_installed")

    def _validate_latent_cache(self) -> None:
        """No-op: pixel-space families carry no latent cache."""
        self._latent_cache_missing = 0

    async def _pre_cache_latents(self) -> None:
        """No-op: pixel-space families have no latents to pre-cache."""

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

        PRXPixelDriver stores its primary model on ``self.driver.model``
        (not ``.transformer``), so the base ``_assign_components`` loop
        would set ``self.transformer = None``. This property delegates
        directly to the driver so ``trainer.transformer`` never goes
        stale (C4).
        """
        return self.driver.model if self.driver is not None else None

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

        Layout (mirrors prx / ovis_image / qwen_image):
        - ``embeddings/{te_quant}/te1`` stores embedding tensors ``[L, D]``
        - ``embeddings/{te_quant}/te2`` stores BOOL attention masks ``[L]``

        The expanded SAMPLE prompts and negative prompt are also warmed
        here: the sampler runs after the TE is offloaded, so it must serve
        all prompts from ``self.text_cache`` — without this, sampling
        causes a VRAM spike by reloading the text encoder from CPU.
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
                batch_caps = [cap for cap, _ in batch_items]

                emb_batch, mask_batch = self._encode_text_direct(
                    batch_caps,
                    dtype,
                )

                for j, (cap, hint) in enumerate(batch_items):
                    emb_cpu = emb_batch[j].cpu()
                    mask_cpu = mask_batch[j].cpu()
                    self.text_cache[cap] = (emb_cpu, mask_cpu)
                    if te1_dir:
                        TextEmbeddingCache.save(cap, emb_cpu, te1_dir, hint)
                    if te2_dir:
                        TextEmbeddingCache.save(cap, mask_cpu, te2_dir, hint)

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
        """Encode captions through the Qwen3-VL text encoder.

        ``batch`` is accepted for hook compatibility and ignored here.

        Returns a ``(embeddings, attention_mask)`` TUPLE. The base pipeline
        passes this opaquely to ``forward_pass()`` which passes it to
        ``driver.forward_pass()`` that unpacks the tuple (C1/C2).

        Returns:
            (text_embeddings ``[B, 256, 2048]``, BOOL attention_mask
            ``[B, 256]``).
        """
        if self.config.get("cache_text_embeddings", True):
            return self._get_cached_text_embeddings(captions, dtype)

        return self._encode_text_direct(captions, dtype)

    def _encode_text_direct(
        self,
        captions: list[str],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode captions directly via the driver (no cache).

        Delegates to ``driver.encode_text`` (single source of truth for the
        pipeline-identical PRXPixel prompt encoding) and unwraps the
        ``TextEncoderOutput`` to the tuple contract.
        """
        out = self.driver.encode_text(captions, dtype)
        return out.embeddings, out.attention_mask

    def _get_cached_text_embeddings(
        self,
        captions: list[str],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode on first encounter, reuse thereafter.

        Cache entries are per-caption CPU tuples ``(emb [L, D], mask [L])``,
        stacked back to ``([B, L, D], [B, L])`` on retrieval. Masks keep
        their BOOL dtype through the cache round trip.
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
                single_emb, single_mask = self._encode_text_direct(
                    [cap],
                    dtype,
                )
                self.text_cache[cap] = (
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
                + ", ".join(cap[:50] for _, cap in uncached)
            )

        for cap in captions:
            cached_emb, cached_mask = self.text_cache[cap]
            emb_results.append(cached_emb.to(self.device, dtype=dtype))
            mask_results.append(cached_mask.to(self.device))

        return torch.stack(emb_results, dim=0), torch.stack(mask_results, dim=0)
