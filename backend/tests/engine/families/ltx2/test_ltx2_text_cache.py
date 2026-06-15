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


# ── Sample-prompt warming (so sampling works after TE offload) ──────────────

class _FakeTransformer:
    """Minimal stand-in: only `next(parameters()).dtype` is read by the sampler."""

    def parameters(self):
        return iter([torch.zeros(1, dtype=torch.float32)])


def _sampling_trainer() -> Ltx2Trainer:
    """Trainer with a sample prompt + triggerword, no training captions."""
    t = _trainer()
    t.config = {
        "cache_text_embeddings": True,
        "sample_prompts": [{"prompt": "a [triggerword] flying over the desert"}],
        "global_triggerword": "sks",
        "datasets": [],
    }
    t._build_caption_hints = lambda: {}  # isolate sample-prompt warming
    return t


def test_pre_cache_warms_expanded_sample_prompts():
    """Sample prompts are encoded (wildcards expanded) during pre-cache."""
    t = _sampling_trainer()
    t._pre_cache_text_embeddings()
    assert "a sks flying over the desert" in t.text_cache  # [triggerword] → sks
    assert "a [triggerword] flying over the desert" not in t.text_cache


def test_sampler_encode_prompt_serves_from_cache_after_te_offload():
    """THE regression: sampler.encode_prompt must use the cache, not the (None)
    driver TE — the old code called driver.encode_text → 'NoneType' is not callable."""
    from app.engine.models.families.ltx2.sampler import Ltx2Sampler

    t = _sampling_trainer()
    t._pre_cache_text_embeddings()
    t.driver.text_encoder = None  # offloaded after pre-cache (real lifecycle)
    t.transformer = _FakeTransformer()
    calls_after_warm = t.driver.calls

    sampler = object.__new__(Ltx2Sampler)
    sampler.pipeline = t

    # The base sampler passes the ALREADY-expanded prompt.
    out = sampler.encode_prompt("a sks flying over the desert")

    assert out.embeddings.shape == (1, 5, 8)
    assert out.pooled.shape == (1, 5, 8)
    assert t.driver.calls == calls_after_warm  # served from cache, TE untouched


def test_expand_prompt_wildcards_helper():
    from app.engine.core.sampling import expand_prompt_wildcards

    cfg = {"global_triggerword": "sks", "datasets": [{"caption_prefix": "photo of"}]}
    assert expand_prompt_wildcards("a [triggerword]", cfg) == "a sks"
    assert expand_prompt_wildcards("[captionprefix] x", cfg) == "photo of x"
    assert expand_prompt_wildcards("no wildcards", cfg) == "no wildcards"
