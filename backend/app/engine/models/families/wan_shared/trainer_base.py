"""Shared WAN trainer behaviour.

Provides :class:`WanTextCacheMixin` — UMT5-XXL text encoding (with lazy
in-memory caching) AND the disk-cache warm-up, shared verbatim by WAN 2.1 and
WAN 2.2 (both trainers encode through the driver's UMT5-XXL with the same
caching contract).
"""

from __future__ import annotations

import os
from typing import Any

import torch


class WanTextCacheMixin:
    """Warm the text cache (disk + memory) before the UMT5 encoder is offloaded.

    ``run_trainer`` runs ``_pre_cache_text_embeddings`` → ``_offload_text_encoders``.
    WAN's ``encode_text`` caches lazily and re-encodes on a miss, but the base
    pre-cache step is a no-op — so without this override the cache is EMPTY when
    the TE is offloaded, and the first training step hits
    ``_get_cached_text_embeddings`` with no cache AND no encoder, which raises
    "Text encoder unavailable for uncached caption(s)". (Same class of bug LTX-2
    hit; LTX-2's fix is the sibling :class:`Ltx2Trainer._pre_cache_text_embeddings`.)

    Disk-backed cache (P1c): like the image families (see
    ``qwen_image``/``krea2``), each ``[1, L, D]`` embedding is persisted via the
    shared :class:`TextEmbeddingCache` under
    ``{ds}/.cache/{model}/{ver}/embeddings/{te_quant}/te1/`` keyed on the caption
    hash. A warm run loads the whole set from disk and NEVER re-encodes through
    the 12B-class UMT5-XXL encoder — closing the "re-encode every run" gap.
    UMT5 has no audio pair, so a single ``te1`` slot (matching
    ``_get_cached_text_embeddings``) suffices; the sample prompts AND the CFG
    negative prompt round-trip through the SAME disk cache as the training
    captions (mirroring the image families, where sample prompts ride in
    ``_build_caption_hints`` and therefore also persist to disk).
    """

    # ── Text Encoding (UMT5-XXL via driver, with lazy cache) ─────────────

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, batch: dict | None = None
    ) -> Any:
        """Encode captions through UMT5-XXL with in-memory caching.

        Returns a raw ``[B, L, D]`` tensor (the WAN transformer takes
        ``encoder_hidden_states`` directly).
        """
        if not self.config.get("cache_text_embeddings", True):
            out = self.driver.encode_text(captions, dtype)
            return out.embeddings if hasattr(out, "embeddings") else out
        return self._get_cached_text_embeddings(captions, dtype)

    def _get_cached_text_embeddings(
        self, captions: list[str], dtype: torch.dtype
    ) -> torch.Tensor:
        results: list[torch.Tensor | None] = []
        uncached: list[tuple[int, str]] = []

        for i, cap in enumerate(captions):
            if cap in self.text_cache:
                results.append(self.text_cache[cap])
            else:
                uncached.append((i, cap))
                results.append(None)

        if uncached and self.text_encoder is not None:
            for orig_idx, cap in uncached:
                out = self.driver.encode_text([cap], dtype)
                emb = out.embeddings if hasattr(out, "embeddings") else out
                self.text_cache[cap] = emb.cpu()
                results[orig_idx] = emb.cpu()
        elif uncached:
            raise RuntimeError(
                "Text encoder unavailable for uncached caption(s): "
                + ", ".join(cap[:50] for _, cap in uncached)
            )

        return torch.cat(
            [r.to(self.device, dtype=dtype) for r in results if r is not None], dim=0
        )

    def _pre_cache_text_embeddings(self) -> None:
        if not self.config.get("cache_text_embeddings", True):
            return
        # The driver does the encoding; warm only while its encoder is resident.
        if getattr(self.driver, "text_encoder", None) is None:
            return

        from app.engine.components.text_embeddings import TextEmbeddingCache

        # Disk cache dir — include the TE quantization scheme so FP8/bf16
        # embeddings never collide (same convention as the image families).
        te_cache_dirs = self._resolve_te_cache_dirs()
        te_quant = self.config.get("te_quantization", "none")
        te1_dir = (
            os.path.join(te_cache_dirs[0], "embeddings", te_quant, "te1")
            if te_cache_dirs
            else ""
        )

        dtype = self._resolve_loading_dtype()

        # ── Build the full ordered work set: training captions, then the
        # expanded SAMPLE prompts, then the CFG negative. Sampling runs AFTER the
        # UMT5 encoder is offloaded and serves prompts from self.text_cache via
        # encode_text, so every one of these must be warmed now (else the sampler
        # hits the offloaded (None) encoder → "'NoneType' object is not callable"
        # for a sample prompt, or a broken cond+uncond pass for the negative).
        work: list[tuple[str, str]] = []
        seen: set[str] = set()

        def _add(cap: str, hint: str) -> None:
            if cap in self.text_cache or cap in seen:
                return
            seen.add(cap)
            work.append((cap, hint))

        for cap, hint in self._build_caption_hints().items():
            _add(cap, hint)
        sample_texts = self._sample_prompt_texts()
        for sp in sample_texts:
            _add(sp, "")
        if sample_texts:
            # Default "" is the standard unconditional; a configured
            # sample_negative_prompt is warmed (and persisted) under its own key.
            _add(str(self.config.get("sample_negative_prompt", "") or ""), "")

        # ── Phase 1: load from disk (skip the encoder entirely on a hit) ──
        disk_loaded = 0
        need_encode: list[tuple[str, str]] = []
        for cap, hint in work:
            if te1_dir:
                emb = TextEmbeddingCache.load(cap, te1_dir, hint)
                if emb is not None:
                    self.text_cache[cap] = emb
                    disk_loaded += 1
                    continue
            need_encode.append((cap, hint))

        if not need_encode:
            if getattr(self, "_log_writer", None):
                self._log_writer.status("TE Cache Loaded from Disk")
            self.logger.info(
                "wan_text_cache_complete",
                cached=len(self.text_cache),
                from_disk=disk_loaded,
                source="disk",
            )
            return

        # ── Phase 2: encode the misses (batched) + persist to disk ──
        if getattr(self, "_log_writer", None):
            self._log_writer.status("Caching Text Embeddings (0%)")

        total = len(need_encode)
        batch_size = 4
        with torch.no_grad():
            for i in range(0, total, batch_size):
                batch_items = need_encode[i : i + batch_size]
                chunk = [cap for cap, _ in batch_items]
                out = self.driver.encode_text(chunk, dtype)
                emb = out.embeddings if hasattr(out, "embeddings") else out
                for j, (cap, hint) in enumerate(batch_items):
                    emb_cpu = emb[j : j + 1].cpu()
                    self.text_cache[cap] = emb_cpu
                    if te1_dir:
                        TextEmbeddingCache.save(cap, emb_cpu, te1_dir, hint)
                if getattr(self, "_log_writer", None):
                    pct = round(min(i + batch_size, total) / total * 100)
                    self._log_writer.status(f"Caching Text Embeddings ({pct}%)")

        self.logger.info(
            "wan_text_cache_complete",
            cached=len(self.text_cache),
            from_disk=disk_loaded,
            newly_encoded=total,
        )

    def _sample_prompt_texts(self) -> list[str]:
        """Expanded sample-prompt strings to pre-cache.

        The sampler requests the wildcard-EXPANDED prompt after the UMT5 encoder
        is offloaded, so warming the same expansion here (shared module helper,
        so the two can't drift) makes the cache key match exactly.
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
