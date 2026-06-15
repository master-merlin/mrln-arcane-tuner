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
