"""NucleusImageTrainer — family-specific trainer for Nucleus-Image.

All shared logic (optimizer, EMA, gradient accumulation, noise offset,
checkpointing, signals, logging) lives in ``GenericTrainingPipeline``. This
module implements the Nucleus-Image-specific behaviour:

- ``encode_text`` returns an ``(embeddings, attention_mask)`` TUPLE (ovis/
  chroma/lumina2 pattern) that ``driver.forward_pass`` unpacks.
- RAGGED (variable-length) TE disk-cache: the real pipeline tokenizes with
  ``padding="longest"`` (NOT a fixed ``max_length`` — see ``driver.py``
  module docstring §3), so a per-caption cache entry sliced out of one
  encode call carries whatever padding THAT call happened to produce. This
  trainer follows the ``qwen_image`` precedent exactly (``_trim_entry`` /
  ``_get_cached_text_embeddings``, the "W3-4" ragged-cache fix): entries are
  stored TRIMMED to their true (mask-derived) length and re-padded to the
  batch max at retrieval time — a plain ``torch.stack`` over ragged entries
  would raise on any real mixed-length batch.
- NO positive/negative encoding asymmetry (unlike ``lumina2``): the real
  pipeline's ``encode_prompt`` applies the identical chat-template wrapping
  to both the training caption and the CFG negative/uncond prompt (see
  ``driver.py`` module docstring §1), so there is exactly ONE cache
  (``self.text_cache``) shared by both — no separate uncond cache is
  needed, and none would even be correct-by-construction here (a caption
  and a negative prompt with identical text produce IDENTICAL embeddings).
"""

from __future__ import annotations

import os

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline

from .driver import te_template_fingerprint

logger = structlog.get_logger(__name__)

# TE disk-cache key template versioning (boogu_image/qwen_image/lumina2
# precedent). The driver's ``encode_text`` bakes ``NUCLEUS_SYSTEM_PROMPT``
# into every prompt via the chat template — if that string ever changes, an
# on-disk embedding cached under the OLD template must not be silently
# reused under the new one. ``_TE_TEMPLATE_ID`` bakes a template identity
# into the string hashed for the on-disk filename; the IN-MEMORY
# ``self.text_cache`` dict stays keyed by the raw caption (matching every
# other family), only the disk path is template-versioned.
_TE_TEMPLATE_VERSION = "v1"
_TE_TEMPLATE_ID = (
    f"nucleus_image/system_prompt/{_TE_TEMPLATE_VERSION}/{te_template_fingerprint()}"
)


def _disk_cache_key(caption: str) -> str:
    """Compose the string hashed by ``TextEmbeddingCache.caption_to_filename``.

    Baking ``_TE_TEMPLATE_ID`` into the hashed string (instead of passing
    the raw caption) means a future system-prompt edit produces a DIFFERENT
    on-disk filename for the same caption text, instead of silently reusing
    a stale embedding encoded under the old template.
    """
    return f"{_TE_TEMPLATE_ID}::{caption}"


class NucleusImageTrainer(GenericTrainingPipeline):
    """Nucleus-Image LoRA trainer.

    32-block single-stream DiT (first 3 dense, last 29 MoE — expert-choice
    routing, 64 routed experts + 1 shared expert per MoE block) with a
    single frozen Qwen3-VL text encoder, the ``AutoencoderKLQwenImage`` VAE
    (shared with ``qwen_image``), and a NON-reversed flow-matching timestep
    convention with a NEGATED raw model output (see ``driver.py`` module
    docstring §4). Supports true CFG with negative prompts at sample time.
    """

    # ── Setup ────────────────────────────────────────────────────────────

    def _setup_family(self) -> None:
        """Initialize Nucleus-Image-specific loader, driver, and saver."""
        from .loader import NucleusImageLoader  # noqa: PLC0415
        from .driver import NucleusImageDriver  # noqa: PLC0415
        from .saver import NucleusImageSaver  # noqa: PLC0415

        self.loader = NucleusImageLoader(self.device)
        self.driver = NucleusImageDriver(self.definition, self.device)
        self.saver = NucleusImageSaver()

    def _create_sampler(self):
        """Create a NucleusImageSampler if sampling is configured."""
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import NucleusImageSampler  # noqa: PLC0415

            return NucleusImageSampler(self)
        return None

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep self.transformer + driver.transformer in sync after PEFT wrap."""
        self.transformer = new_model
        self.components["unet"] = new_model
        self.driver.transformer = new_model

    # ── Disk-backed TE Pre-caching ───────────────────────────────────────

    def _sample_prompt_texts(self) -> list[str]:
        """Expanded sample-prompt strings to pre-cache (mirrors ovis/chroma/
        lumina2/qwen_image)."""
        from app.engine.core.sampling import expand_prompt_wildcards  # noqa: PLC0415

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
        # Also warm the CFG negative prompt (module docstring — shares the
        # SAME cache/formatting as any other caption for this family).
        neg = str(self.config.get("sample_negative_prompt", "") or "")
        if neg and neg not in texts:
            texts.append(neg)
        return texts

    def _pre_cache_text_embeddings(self) -> None:
        """Warm the Qwen3-VL embedding + attention-mask cache from disk and
        encode whatever is missing.

        Layout (mirrors ovis_image/chroma/lumina2/qwen_image):
        - ``embeddings/{te_quant}/te1`` stores embeddings ``[L, 4096]``
        - ``embeddings/{te_quant}/te2`` stores attention masks ``[L]``
        Entries are stored TRIMMED to their true length (RAGGED cache — see
        module docstring / ``_trim_entry``) and re-padded to the batch max
        at retrieval time.
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
                emb_tensor = TextEmbeddingCache.load(
                    _disk_cache_key(caption), te1_dir, hint,
                )
                mask_tensor = TextEmbeddingCache.load(
                    _disk_cache_key(caption), te2_dir, hint,
                )
                if emb_tensor is not None and mask_tensor is not None:
                    self.text_cache[caption] = (emb_tensor, mask_tensor)
                    disk_loaded += 1
                    continue
            need_encode.append((caption, hint))

        # Warm sample prompts (including the CFG negative prompt — no
        # asymmetry for this family, module docstring).
        sample_texts = self._sample_prompt_texts()
        queued = {cap for cap, _ in need_encode}
        for sp in sample_texts:
            if sp not in self.text_cache and sp not in queued:
                need_encode.append((sp, ""))
                queued.add(sp)

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
                    # Trim each entry out of the sub-batch's own padding
                    # (RAGGED cache — see module docstring / qwen_image
                    # "W3-4" precedent).
                    emb_cpu, mask_cpu = self._trim_entry(
                        emb_batch[j].cpu(), mask_batch[j].cpu(),
                    )
                    self.text_cache[cap] = (emb_cpu, mask_cpu)
                    if te1_dir:
                        TextEmbeddingCache.save(
                            _disk_cache_key(cap), emb_cpu, te1_dir, hint,
                        )
                    if te2_dir:
                        TextEmbeddingCache.save(
                            _disk_cache_key(cap), mask_cpu, te2_dir, hint,
                        )

                pct = int(min(i + batch_size, encode_total) / encode_total * 100)
                if pct % 10 == 0 or (i + batch_size) >= encode_total:
                    if getattr(self, "_log_writer", None):
                        self._log_writer.status(
                            f"Caching Text Embeddings ({pct}%)",
                        )

        self.logger.info(
            "text_embedding_cache_complete",
            cached=len(self.text_cache), newly_encoded=encode_total,
        )

    # ── Text Encoding ────────────────────────────────────────────────────

    def encode_text(
        self,
        captions: list[str],
        dtype: torch.dtype,
        batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode captions through the Nucleus-Image Qwen3-VL text encoder.

        ``batch`` is accepted for hook compatibility and ignored here. Used
        identically for real training captions AND the CFG negative/uncond
        prompt (module docstring — no asymmetry for this family).

        Returns:
            (Qwen3-VL embeddings ``[B, L, 4096]``, attention mask
            ``[B, L]``) — ``L`` is the batch-max length after RAGGED
            re-padding (see ``_get_cached_text_embeddings``).
        """
        if self.config.get("cache_text_embeddings", True):
            return self._get_cached_text_embeddings(captions, dtype)
        return self._encode_text_direct(captions, dtype)

    def _encode_text_direct(
        self,
        captions: list[str],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode captions directly via the driver (no cache)."""
        out = self.driver.encode_text(captions, dtype)
        return out.embeddings, out.attention_mask

    @staticmethod
    def _trim_entry(
        emb: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Trim a single per-caption cache entry to its TRUE (mask) length.

        Qwen3-VL (via the processor's ``padding="longest"``) has no fixed
        crop, so an entry sliced out of a padded batch carries whatever
        padding THAT batch happened to have. Cache entries must be
        length-normalized (the qwen_image/kandinsky5/boogu_image precedent)
        so reassembly padding is well-defined and independent of the batch
        an entry was first encoded in — otherwise mixed-length batches raise
        a ragged ``torch.stack`` RuntimeError.

        Args:
            emb: Per-caption embeddings ``[L_padded, D]``.
            mask: Per-caption attention mask ``[L_padded]`` (right-padded,
                so valid positions are a prefix).

        Returns:
            ``(emb [L_true, D], mask [L_true])``.
        """
        true_len = int(mask.sum().item())
        return emb[:true_len], mask[:true_len]

    def _get_cached_text_embeddings(
        self,
        captions: list[str],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode on first encounter, reuse thereafter.

        Cache entries are per-caption CPU tuples ``(emb [L_true, D],
        mask [L_true])`` (trimmed to true length — RAGGED cache, see
        module docstring), re-padded to the batch max on retrieval —
        byte-equivalent to the direct encode path, which itself re-pads to
        ITS batch's own max (``NucleusImageDriver.forward_pass`` derives
        masking from whatever mask it is handed either way). A plain
        ``torch.stack`` over ragged entries would raise on any real
        mixed-length batch.
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
                self.text_cache[cap] = self._trim_entry(
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

        # Mask-aware padded reassembly (entries are ragged — see docstring).
        entries = [self.text_cache[cap] for cap in captions]
        max_len = max(e.shape[0] for e, _ in entries)

        emb_results: list[torch.Tensor] = []
        mask_results: list[torch.Tensor] = []
        for cached_emb, cached_mask in entries:
            emb = cached_emb.to(self.device, dtype=dtype)
            mask = cached_mask.to(self.device)
            pad_rows = max_len - emb.shape[0]
            if pad_rows > 0:
                emb = torch.cat(
                    [emb, emb.new_zeros(pad_rows, *emb.shape[1:])], dim=0,
                )
                mask = torch.cat([mask, mask.new_zeros(pad_rows)], dim=0)
            emb_results.append(emb)
            mask_results.append(mask)

        return torch.stack(emb_results, dim=0), torch.stack(mask_results, dim=0)
