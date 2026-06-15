"""LTX-2 text-embedding cache (no GPU).

Regression for the burn-in crash "'NoneType' object has no attribute 'dtype'":
``run_trainer`` warms then offloads the 12B Gemma3 text encoder, but the base
warm step is a no-op, so LTX-2 trained with an empty cache + no TE →
``encode_text`` returned ``None`` → ``video_emb`` ``None``. The trainer now
warms the cache (full video+audio triple) before offload and reassembles a
batched ``TextEncoderOutput`` from it — so encoding still works once the TE is
gone.
"""

from __future__ import annotations

import structlog
import torch

from app.engine.core.text_encoding import TextEncoderOutput
from app.engine.models.families.ltx2.trainer import Ltx2Trainer


class _FakeDriver:
    """Driver whose encode_text returns a (video, audio) TextEncoderOutput."""

    def __init__(self) -> None:
        self.text_encoder: object | None = object()  # non-None = resident
        self.calls = 0

    def encode_text(self, captions, dtype):
        self.calls += 1
        b = len(captions)
        return TextEncoderOutput(
            embeddings=torch.ones(b, 5, 8),
            attention_mask=torch.ones(b, 5),
            pooled=torch.full((b, 5, 8), 2.0),  # audio text emb
        )


def _trainer(cache: bool = True) -> Ltx2Trainer:
    t = object.__new__(Ltx2Trainer)
    t.logger = structlog.get_logger("test")
    t.device = torch.device("cpu")
    t.config = {"cache_text_embeddings": cache}
    t.text_cache = {}
    t.driver = _FakeDriver()
    t._log_writer = None
    t._build_caption_hints = lambda: {"a cat": "h", "a dog": "h", "": "d"}
    t._resolve_loading_dtype = lambda: torch.float32
    return t


def test_pre_cache_warms_full_video_audio_triples():
    t = _trainer()
    t._pre_cache_text_embeddings()
    assert set(t.text_cache) == {"a cat", "a dog", ""}
    emb, pooled, mask = t.text_cache["a cat"]
    assert emb.shape == (1, 5, 8) and emb.device.type == "cpu"
    assert pooled.shape == (1, 5, 8)  # audio pooled preserved for PR3b
    assert mask.shape == (1, 5)


def test_encode_text_works_after_te_offloaded():
    """THE regression: warm → offload the TE → encoding still succeeds."""
    t = _trainer()
    t._pre_cache_text_embeddings()
    t.driver.text_encoder = None  # offloaded
    calls_after_warm = t.driver.calls

    out = t.encode_text(["a cat", "a dog"], torch.float32)

    assert out.embeddings.shape == (2, 5, 8)
    assert out.pooled.shape == (2, 5, 8)
    assert out.attention_mask.shape == (2, 5)
    # Served purely from cache — the offloaded TE was not touched.
    assert t.driver.calls == calls_after_warm


def test_encode_text_miss_after_offload_raises():
    t = _trainer()
    t.driver.text_encoder = None  # offloaded, nothing pre-cached
    import pytest

    with pytest.raises(RuntimeError, match="not pre-cached"):
        t.encode_text(["never seen"], torch.float32)


def test_encode_text_miss_while_resident_encodes_and_caches():
    t = _trainer()  # TE resident, empty cache
    out = t.encode_text(["a cat"], torch.float32)
    assert out.embeddings.shape == (1, 5, 8)
    assert "a cat" in t.text_cache  # now cached for the next step


def test_cache_off_delegates_to_driver():
    t = _trainer(cache=False)
    out = t.encode_text(["a cat"], torch.float32)
    assert out.embeddings.shape == (1, 5, 8)
    assert t.text_cache == {}  # nothing cached when caching is off
