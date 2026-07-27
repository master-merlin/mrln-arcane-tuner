"""FLUX.2 trainer text-cache tests — TE-unloaded cache-miss handling.

Regression: ``_get_cached_text_embeddings``'s ``elif uncached_caps:`` branch
used to silently fill a ``torch.zeros(...)`` dummy embedding and just log
``text_encoder_unavailable`` when the text encoder had been fully unloaded
and a caption wasn't pre-cached (e.g. a resumed/edited run with one new
caption). That trains that caption on NULL conditioning — silent quality
degradation. Every sibling (flux1, qwen_image, sdxl, ovis_image, ...) raises
a ``RuntimeError`` instead; flux2 must match that house convention.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from app.engine.models.families.flux2.trainer import Flux2Trainer

_L, _D = 8, 16 * 3  # te_max_length=8, 4096*concat_layers stand-in


class _FakeDriver:
    """Driver stand-in: encode_text returns a deterministic tiny embedding."""

    te_max_length = _L
    te_concat_layers = 3

    def __init__(self):
        self.encoded: list[str] = []

    def encode_text(self, captions, dtype):
        self.encoded.extend(captions)
        emb = torch.stack([torch.full((_L, _D), float(len(c))) for c in captions]).to(
            dtype
        )
        return SimpleNamespace(embeddings=emb)


def _trainer(text_encoder=None, text_cache: dict | None = None) -> Flux2Trainer:
    t = object.__new__(Flux2Trainer)
    t.device = torch.device("cpu")
    t.config = {"cache_text_embeddings": True}
    t.text_cache = dict(text_cache or {})
    t.text_encoder = text_encoder
    t.driver = _FakeDriver()
    t.logger = SimpleNamespace(
        debug=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        info=lambda *a, **k: None,
    )
    return t


def test_te_unloaded_cache_miss_raises():
    """THE fix: TE fully unloaded + uncached caption -> hard RuntimeError,
    matching the house convention (qwen_image trainer.py, flux1, sdxl, ...)."""
    t = _trainer(text_encoder=None)

    with pytest.raises(RuntimeError, match="uncached"):
        t.encode_text(["never seen"], torch.float32)


def test_te_unloaded_cache_hit_still_works():
    """A caption that WAS pre-cached must still resolve fine when the TE is
    unloaded — only genuine misses should raise."""
    cached_emb = torch.full((1, _L, _D), 9.0)
    t = _trainer(text_encoder=None, text_cache={"already cached": cached_emb})

    out = t.encode_text(["already cached"], torch.float32)
    assert out.shape == (1, _L, _D)
    assert torch.all(out == 9.0)


def test_te_resident_cache_miss_encodes_fresh():
    """Sanity: with a resident TE, a cache miss encodes instead of raising."""
    t = _trainer(text_encoder=torch.nn.Linear(1, 1))

    out = t.encode_text(["fresh caption"], torch.float32)
    assert out.shape[0] == 1
    assert "fresh caption" in t.text_cache
    assert t.driver.encoded == ["fresh caption"]
