"""ChromaTrainer — family-specific trainer for Chroma.

All shared logic (optimizer, EMA, gradient accumulation, noise offset,
checkpointing, signals, logging) lives in ``GenericTrainingPipeline``. This
module implements the Chroma-specific behaviour:

- ``encode_text`` returns an ``(embeddings, attention_mask)`` TUPLE (like
  ovis_image) that ``driver.forward_pass`` unpacks — Chroma's transformer
  DOES consume the attention mask (unlike ovis, where padding is pre-zeroed
  into the embeddings instead), so both tensors must be cached together.
- ``_update_primary_model`` also syncs ``self.driver.transformer`` (flux1
  pattern — Chroma's driver stores its primary model on ``.transformer``,
  not ``.model`` like ovis_image) so the PEFT-wrapped model stays in the
  forward graph.
- Flow-match: no scheduler, no CLIP path anywhere, T5 stays frozen
  (``get_te_lora_targets`` returns ``[]``).
"""

from __future__ import annotations

import os

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline

logger = structlog.get_logger(__name__)


class ChromaTrainer(GenericTrainingPipeline):
    """Chroma (Chroma1-Base / Chroma1-HD) LoRA trainer.

    8.9B FLUX.1-schnell-derived MMDiT (19 double + 38 single blocks) with a
    single T5-XXL text encoder (no CLIP), AutoencoderKL VAE (FLUX.1-schnell's),
    and flow-matching noise schedule. Supports real (non-distilled) CFG with
    negative prompts at sample time.
    """

    # ── Setup ────────────────────────────────────────────────────────────

    def _setup_family(self) -> None:
        """Initialize Chroma-specific loader, driver, and saver."""
        from .loader import ChromaLoader
        from .driver import ChromaDriver
        from .saver import ChromaSaver

        self.loader = ChromaLoader(self.device)
        self.driver = ChromaDriver(self.definition, self.device)
        self.saver = ChromaSaver()

        # Chroma patchifies latents (2× down per spatial dim) before the
        # transformer, same as flux1 — tell the flux_shift/model_shift
        # timestep-sampling modes so mu is computed off the patched seq_len.
        self.config.setdefault("flux_shift_patchify_factor", 2)

    def _create_sampler(self):
        """Create a ChromaSampler if sampling is configured."""
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import ChromaSampler
            return ChromaSampler(self)
        return None

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep self.transformer + driver.transformer in sync after PEFT wrap."""
        self.transformer = new_model
        self.components["unet"] = new_model
        self.driver.transformer = new_model

    # ── Disk-backed TE Pre-caching ───────────────────────────────────────

    def _sample_prompt_texts(self) -> list[str]:
        """Expanded sample-prompt strings to pre-cache (mirrors ovis_image)."""
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
        """Warm T5 embedding + attention-mask cache from disk + encode missing.

        Layout (mirrors ovis_image / krea2 / qwen_image):
        - ``embeddings/{te_quant}/te1`` stores T5 embeddings ``[L, 4096]``
        - ``embeddings/{te_quant}/te2`` stores Chroma's modified attention
          mask ``[L]`` (the "one padding token survives" mask, NOT the raw
          tokenizer mask — see ``driver.encode_text`` docstring)
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

        # Warm sample + negative prompts so the TE can stay offloaded during
        # sampling (krea2/ovis VRAM-spike lesson).
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
            cached=len(self.text_cache),
            newly_encoded=encode_total,
        )

    # ── Text Encoding ────────────────────────────────────────────────────

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode captions through the Chroma T5 text encoder.

        ``batch`` is accepted for hook compatibility and ignored here.

        Returns a ``(embeddings, attention_mask)`` TUPLE — the base pipeline
        passes this opaquely to ``forward_pass()`` which passes it to
        ``driver.forward_pass()`` that unpacks the tuple.

        Returns:
            (T5 embeddings ``[B, 512, 4096]``, modified attention mask
            ``[B, 512]``).
        """
        if self.config.get("cache_text_embeddings", True):
            return self._get_cached_text_embeddings(captions, dtype)
        return self._encode_text_direct(captions, dtype)

    def _encode_text_direct(
        self, captions: list[str], dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode captions directly via the driver (no cache)."""
        out = self.driver.encode_text(captions, dtype)
        return out.embeddings, out.attention_mask

    def _get_cached_text_embeddings(
        self, captions: list[str], dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode on first encounter, reuse thereafter.

        Cache entries are per-caption CPU tuples ``(emb [L, D], mask [L])``,
        stacked back to ``([B, L, D], [B, L])`` on retrieval.
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
                "text_embeddings_cached", new=len(uncached), total=len(self.text_cache),
            )
        elif uncached:
            raise RuntimeError(
                "Text encoder was unloaded but encountered uncached caption(s): "
                + ", ".join(cap[:50] for _, cap in uncached)
            )

        emb_results = []
        mask_results = []
        for cap in captions:
            cached_emb, cached_mask = self.text_cache[cap]
            emb_results.append(cached_emb.to(self.device, dtype=dtype))
            mask_results.append(cached_mask.to(self.device, dtype=dtype))

        return torch.stack(emb_results, dim=0), torch.stack(mask_results, dim=0)
