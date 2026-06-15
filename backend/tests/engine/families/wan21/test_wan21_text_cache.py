"""WAN text-embedding cache warming (no GPU).

Regression for the TE-offload crash shared by WAN 2.1/2.2: ``run_trainer``
warms then offloads the UMT5 encoder, but the base warm step is a no-op, so the
cache was EMPTY at train time and ``_get_cached_text_embeddings`` raised "Text
encoder unavailable for uncached caption(s)". The shared ``WanTextCacheMixin``
now warms the cache before offload so encoding still works once the TE is gone.
"""

from __future__ import annotations

import structlog
import torch

from app.engine.models.families.wan_shared.trainer_base import WanTextCacheMixin
from app.engine.models.families.wan21.trainer import Wan21Trainer
from app.engine.models.families.wan22.trainer import Wan22Trainer


class _FakeDriver:
    """WAN driver whose encode_text returns a bare ``[B, L, D]`` tensor."""

    def __init__(self) -> None:
        self.text_encoder: object | None = object()  # resident
        self.calls = 0

    def encode_text(self, captions, dtype):
        self.calls += 1
        return torch.ones(len(captions), 5, 8)


def _trainer() -> Wan21Trainer:
    t = object.__new__(Wan21Trainer)
    t.logger = structlog.get_logger("test")
    t.device = torch.device("cpu")
    t.config = {"cache_text_embeddings": True}
    t.text_cache = {}
    t.text_encoder = object()  # trainer alias (resident)
    t.driver = _FakeDriver()
    t._log_writer = None
    t._build_caption_hints = lambda: {"a cat": "h", "a dog": "h", "": "d"}
    t._resolve_loading_dtype = lambda: torch.float32
    return t


def test_both_wan_trainers_use_the_warm_mixin():
    # Not the base no-op — the shared mixin's warm.
    assert Wan21Trainer._pre_cache_text_embeddings is WanTextCacheMixin._pre_cache_text_embeddings
    assert Wan22Trainer._pre_cache_text_embeddings is WanTextCacheMixin._pre_cache_text_embeddings


def test_warm_populates_tensor_cache():
    t = _trainer()
    t._pre_cache_text_embeddings()
    assert set(t.text_cache) == {"a cat", "a dog", ""}
    cat = t.text_cache["a cat"]
    assert cat.shape == (1, 5, 8) and cat.device.type == "cpu"


def test_encode_text_works_after_te_offloaded():
    """THE regression: warm → offload the TE → encoding still succeeds."""
    t = _trainer()
    t._pre_cache_text_embeddings()
    calls_after_warm = t.driver.calls
    t.driver.text_encoder = None  # offloaded
    t.text_encoder = None

    out = t.encode_text(["a cat", "a dog"], torch.float32)

    assert out.shape == (2, 5, 8)
    assert t.driver.calls == calls_after_warm  # served from cache, TE untouched


def test_uncached_after_offload_still_raises():
    """Without a warm (empty cache) + offloaded TE → the original failure mode."""
    t = _trainer()
    t.driver.text_encoder = None
    t.text_encoder = None
    import pytest

    with pytest.raises(RuntimeError, match="Text encoder unavailable"):
        t.encode_text(["never seen"], torch.float32)
