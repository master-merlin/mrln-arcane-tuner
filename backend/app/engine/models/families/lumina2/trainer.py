"""Lumina2Trainer — family-specific trainer for Lumina-Image-2.0.

All shared logic (optimizer, EMA, gradient accumulation, noise offset,
checkpointing, signals, logging) lives in ``GenericTrainingPipeline``. This
module implements the Lumina2-specific behaviour:

- ``encode_text`` returns an ``(embeddings, attention_mask)`` TUPLE (ovis/
  chroma pattern) that ``driver.forward_pass`` unpacks — Lumina2's
  transformer DOES consume ``encoder_attention_mask``.
- ``_update_primary_model`` also syncs ``self.driver.transformer`` (flux1/
  chroma pattern — ``Lumina2Driver`` stores its primary model on
  ``.transformer``, not ``.model`` like ovis_image).
- TE DISK-CACHE KEY TEMPLATE VERSIONING (boogu_image pattern): the driver's
  ``encode_text`` prepends ``LUMINA2_SYSTEM_PROMPT + " <Prompt Start> "`` to
  every caption before tokenizing. If that template string ever changes, an
  on-disk embedding cached under the OLD template would silently keep being
  reused under the new one — same caption text, different actual encoded
  content. ``_disk_cache_key`` bakes a template identity
  (``_TE_TEMPLATE_ID``, itself deriving its fingerprint from the driver's
  ``te_template_fingerprint()`` so a prompt-text edit can never forget to
  bump it) into the string hashed for the on-disk filename — the IN-MEMORY
  ``self.text_cache`` dict stays keyed by the raw caption (matching every
  other family), only the disk path is template-versioned.
- UNCOND (negative-prompt) caching lives in a SEPARATE in-memory-only dict,
  ``self.uncond_text_cache`` — the real pipeline encodes the CFG negative
  prompt WITHOUT the system-prompt prefix (module docstring §1 in
  ``driver.py``), so it cannot share a cache keyed by raw caption text with
  the (prefixed) positive-prompt cache: the same string could otherwise
  collide under two different actual encodings. It is not disk-persisted
  (a single short negative-prompt string, cheap to re-encode; no template-
  fingerprint disk-key scheme needed for it).
"""

from __future__ import annotations

import os

import structlog
import torch

from app.engine.core.pipeline import GenericTrainingPipeline

from .driver import te_template_fingerprint

logger = structlog.get_logger(__name__)

# See module docstring. Version bump this manually for a non-prompt-text
# template change (e.g. a different padding/truncation policy); the
# fingerprint suffix auto-bumps on any system-prompt TEXT edit.
_TE_TEMPLATE_VERSION = "v1"
_TE_TEMPLATE_FINGERPRINT = te_template_fingerprint()
_TE_TEMPLATE_ID = (
    f"lumina2/system_prompt/{_TE_TEMPLATE_VERSION}/{_TE_TEMPLATE_FINGERPRINT}"
)


def _disk_cache_key(caption: str) -> str:
    """Compose the string hashed by ``TextEmbeddingCache.caption_to_filename``.

    Baking ``_TE_TEMPLATE_ID`` into the hashed string (instead of passing
    the raw caption) means a future system-prompt edit produces a DIFFERENT
    on-disk filename for the same caption text, instead of silently reusing
    a stale embedding encoded under the old template.
    """
    return f"{_TE_TEMPLATE_ID}::{caption}"


class Lumina2Trainer(GenericTrainingPipeline):
    """Lumina-Image-2.0 LoRA trainer.

    2.6B DiT (26 joint + 2 context-refiner + 2 noise-refiner blocks) with a
    single frozen Gemma-2-2B text encoder, the FLUX.1-dev AutoencoderKL VAE,
    and a REVERSED flow-matching timestep convention (see ``driver.py``
    module docstring §3). Supports true CFG with negative prompts at
    sample time.
    """

    # ── Setup ────────────────────────────────────────────────────────────

    def _setup_family(self) -> None:
        """Initialize Lumina2-specific loader, driver, and saver."""
        from .loader import Lumina2Loader  # noqa: PLC0415
        from .driver import Lumina2Driver  # noqa: PLC0415
        from .saver import Lumina2Saver  # noqa: PLC0415

        self.loader = Lumina2Loader(self.device)
        self.driver = Lumina2Driver(self.definition, self.device)
        self.saver = Lumina2Saver()
        self.uncond_text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _create_sampler(self):
        """Create a Lumina2Sampler if sampling is configured."""
        interval = int(self.config.get("sample_every_n_steps", 0))
        if interval > 0:
            from .sampler import Lumina2Sampler  # noqa: PLC0415

            return Lumina2Sampler(self)
        return None

    def _update_primary_model(self, new_model: torch.nn.Module) -> None:
        """Keep self.transformer + driver.transformer in sync after PEFT wrap."""
        self.transformer = new_model
        self.components["unet"] = new_model
        self.driver.transformer = new_model

    # ── Disk-backed TE Pre-caching ───────────────────────────────────────

    def _sample_prompt_texts(self) -> list[str]:
        """Expanded sample-prompt strings to pre-cache (mirrors ovis/chroma)."""
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
        return texts

    def _pre_cache_text_embeddings(self) -> None:
        """Warm the Gemma-2 embedding + attention-mask cache from disk and
        encode whatever is missing; also warms the UNCOND negative-prompt
        cache (in-memory only — see module docstring).

        Layout (mirrors ovis_image / chroma):
        - ``embeddings/{te_quant}/te1`` stores embeddings ``[L, 2304]``
        - ``embeddings/{te_quant}/te2`` stores attention masks ``[L]``
        Disk save/load calls use :func:`_disk_cache_key` (template-baked),
        not the raw caption — see module docstring.
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

        # Warm sample prompts (POSITIVE — system-prompt applied) so the TE
        # can stay offloaded during sampling (krea2/ovis VRAM-spike lesson).
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

        if need_encode:
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
                        batch_caps, dtype,
                    )

                    for j, (cap, hint) in enumerate(batch_items):
                        emb_cpu = emb_batch[j].cpu()
                        mask_cpu = mask_batch[j].cpu()
                        self.text_cache[cap] = (emb_cpu, mask_cpu)
                        if te1_dir:
                            TextEmbeddingCache.save(
                                _disk_cache_key(cap), emb_cpu, te1_dir, hint,
                            )
                        if te2_dir:
                            TextEmbeddingCache.save(
                                _disk_cache_key(cap), mask_cpu, te2_dir, hint,
                            )

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
        elif getattr(self, "_log_writer", None):
            self._log_writer.status("TE Cache Loaded from Disk")

        # UNCOND negative prompt — RAW (no system prompt), in-memory only.
        if sample_texts:
            neg = str(self.config.get("sample_negative_prompt", "") or "")
            if neg not in self.uncond_text_cache:
                dtype = self._resolve_loading_dtype()
                emb, mask = self._encode_text_direct(
                    [neg], dtype, apply_system_prompt=False,
                )
                self.uncond_text_cache[neg] = (emb.squeeze(0).cpu(), mask.squeeze(0).cpu())

    # ── Text Encoding ────────────────────────────────────────────────────

    def encode_text(
        self,
        captions: list[str],
        dtype: torch.dtype,
        batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode captions through the Lumina2 Gemma-2 text encoder.

        ``batch`` is accepted for hook compatibility and ignored here. The
        system prompt is ALWAYS applied here (this path is used for real
        training captions and the sampler's positive prompt — see
        ``encode_negative_text`` for the CFG uncond path).

        Returns:
            (Gemma-2 embeddings ``[B, 256, 2304]``, attention mask
            ``[B, 256]``).
        """
        if self.config.get("cache_text_embeddings", True):
            return self._get_cached_text_embeddings(captions, dtype)
        return self._encode_text_direct(captions, dtype)

    def encode_negative_text(
        self,
        captions: list[str],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the CFG negative/uncond prompt WITHOUT the system prompt
        (pipeline_lumina2.py's ``negative_prompt`` branch never applies it —
        see ``driver.py`` module docstring §1). Serves from
        ``self.uncond_text_cache`` (warmed in ``_pre_cache_text_embeddings``)
        when available, falling back to a direct (TE-must-be-on-device)
        encode otherwise.
        """
        cache = getattr(self, "uncond_text_cache", None) or {}
        emb_results: list[torch.Tensor] = []
        mask_results: list[torch.Tensor] = []
        uncached: list[str] = []

        for cap in captions:
            if cap not in cache:
                uncached.append(cap)

        if uncached:
            emb_batch, mask_batch = self._encode_text_direct(
                uncached, dtype, apply_system_prompt=False,
            )
            for j, cap in enumerate(uncached):
                cache[cap] = (emb_batch[j].cpu(), mask_batch[j].cpu())
            self.uncond_text_cache = cache

        for cap in captions:
            cached_emb, cached_mask = cache[cap]
            emb_results.append(cached_emb.to(self.device, dtype=dtype))
            mask_results.append(cached_mask.to(self.device))

        return torch.stack(emb_results, dim=0), torch.stack(mask_results, dim=0)

    def _encode_text_direct(
        self,
        captions: list[str],
        dtype: torch.dtype,
        apply_system_prompt: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode captions directly via the driver (no cache)."""
        out = self.driver.encode_text(
            captions, dtype, apply_system_prompt=apply_system_prompt,
        )
        return out.embeddings, out.attention_mask

    def _get_cached_text_embeddings(
        self,
        captions: list[str],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode on first encounter, reuse thereafter.

        Cache entries are per-caption CPU tuples ``(emb [L, D], mask [L])``,
        stacked back to ``([B, L, D], [B, L])`` on retrieval.
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
