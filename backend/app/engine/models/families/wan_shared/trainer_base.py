"""Shared WAN trainer behaviour.

Currently provides :class:`WanTextCacheMixin` — text-embedding cache warming
shared by WAN 2.1 and WAN 2.2 (both encode through UMT5-XXL with the same lazy
in-memory cache).
"""

from __future__ import annotations

import torch


class WanTextCacheMixin:
    """Warm the in-memory text cache before the UMT5 encoder is offloaded.

    ``run_trainer`` runs ``_pre_cache_text_embeddings`` → ``_offload_text_encoders``.
    WAN's ``encode_text`` caches lazily and re-encodes on a miss, but the base
    pre-cache step is a no-op — so without this override the cache is EMPTY when
    the TE is offloaded, and the first training step hits
    ``_get_cached_text_embeddings`` with no cache AND no encoder, which raises
    "Text encoder unavailable for uncached caption(s)". (Same class of bug LTX-2
    hit; LTX-2's fix is the sibling :class:`Ltx2Trainer._pre_cache_text_embeddings`.)

    Warming here encodes every unique caption the train loop will request — the
    exact trigger/prefix/dropout composites built by ``_build_caption_hints`` —
    and stores each ``[1, L, D]`` tensor on CPU under the same key the lazy path
    + train loop use, so encoding still works once the encoder is gone. UMT5 has
    no audio pair, so a plain tensor cache (matching ``_get_cached_text_embeddings``)
    suffices.
    """

    def _pre_cache_text_embeddings(self) -> None:
        if not self.config.get("cache_text_embeddings", True):
            return
        # The driver does the encoding; warm only while its encoder is resident.
        if getattr(self.driver, "text_encoder", None) is None:
            return

        dtype = self._resolve_loading_dtype()
        captions = [c for c in self._build_caption_hints() if c not in self.text_cache]
        # Warm the expanded SAMPLE prompts too: the sampler runs AFTER the UMT5
        # encoder is offloaded and serves prompts from self.text_cache via
        # encode_text, so without this it hits the offloaded (None) encoder and
        # crashes with "'NoneType' object is not callable".
        sample_texts = self._sample_prompt_texts()
        for sp in sample_texts:
            if sp not in self.text_cache and sp not in captions:
                captions.append(sp)
        # CFG preview sampling runs a cond + UNCONDITIONAL forward when
        # guidance_scale > 1 (the default 3.5), and the UMT5 encoder is offloaded
        # by sample time — so the negative prompt must be warmed now, exactly
        # like the sample prompts. Default "" is the standard unconditional; a
        # configured ``sample_negative_prompt`` is warmed under its own key.
        if sample_texts:
            neg = str(self.config.get("sample_negative_prompt", "") or "")
            if neg not in self.text_cache and neg not in captions:
                captions.append(neg)
        total = len(captions)
        if not total:
            self.logger.info("wan_text_cache_complete", cached=len(self.text_cache))
            return

        if getattr(self, "_log_writer", None):
            self._log_writer.status("Caching Text Embeddings (0%)")

        batch_size = 4
        with torch.no_grad():
            for i in range(0, total, batch_size):
                chunk = captions[i : i + batch_size]
                out = self.driver.encode_text(chunk, dtype)
                emb = out.embeddings if hasattr(out, "embeddings") else out
                for j, cap in enumerate(chunk):
                    self.text_cache[cap] = emb[j : j + 1].cpu()
                if getattr(self, "_log_writer", None):
                    pct = round(min(i + batch_size, total) / total * 100)
                    self._log_writer.status(f"Caching Text Embeddings ({pct}%)")

        self.logger.info(
            "wan_text_cache_complete",
            cached=len(self.text_cache),
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
