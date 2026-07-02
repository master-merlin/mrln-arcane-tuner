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
    t._resolve_te_cache_dirs = lambda: []  # disk cache off by default (in-memory tests)
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


# ── CFG unconditional warming (so guidance_scale>1 works after TE offload) ──


def test_pre_cache_warms_default_unconditional_for_cfg():
    """CFG needs the unconditional ('' negative) prompt cached before TE offload."""
    t = _sampling_trainer()  # has a sample prompt, no training captions
    t._pre_cache_text_embeddings()
    assert "" in t.text_cache  # default negative warmed for the cond+uncond pass


def test_pre_cache_warms_configured_negative_prompt():
    t = _sampling_trainer()
    t.config["sample_negative_prompt"] = "blurry, low quality"
    t._pre_cache_text_embeddings()
    assert "blurry, low quality" in t.text_cache


def test_pre_cache_skips_unconditional_when_no_sample_prompts():
    """No sampling → no need to warm the unconditional (keeps the cache minimal)."""
    t = _trainer()  # captions {"a cat","a dog",""} but NO sample prompts
    t._build_caption_hints = lambda: {"a cat": "h", "a dog": "h"}  # drop the ""
    t._pre_cache_text_embeddings()
    assert "" not in t.text_cache  # not warmed when sampling is off


# ── Disk-backed cache (P1c — mirror the image families' TextEmbeddingCache) ──


class _FakeLogWriter:
    def __init__(self) -> None:
        self.statuses: list[str] = []

    def status(self, msg: str) -> None:
        self.statuses.append(msg)


def _disk_trainer(tmp: str, log_writer=None) -> Ltx2Trainer:
    t = _trainer()
    t._log_writer = log_writer
    t._resolve_te_cache_dirs = lambda: [tmp]
    return t


def test_cold_run_writes_full_triple_to_disk(tmp_path):
    """First run encodes via the stub TE and persists emb/pooled/mask to te1-3."""
    import os

    t = _disk_trainer(str(tmp_path))
    t._pre_cache_text_embeddings()

    base = os.path.join(str(tmp_path), "embeddings", "none")
    for slot in ("te1", "te2", "te3"):
        files = [f for f in os.listdir(os.path.join(base, slot)) if f.endswith(".safetensors")]
        assert len(files) == 3, f"{slot} should hold 3 caption files"
    assert t.driver.calls > 0


def test_warm_run_loads_triple_from_disk_without_re_encoding(tmp_path):
    """Second run over the same captions never touches the stub TE."""
    log = _FakeLogWriter()
    cold = _disk_trainer(str(tmp_path), log)
    cold._pre_cache_text_embeddings()

    warm = _disk_trainer(str(tmp_path), log)
    warm._pre_cache_text_embeddings()

    assert warm.driver.calls == 0  # every caption served from disk
    assert set(warm.text_cache) == {"a cat", "a dog", ""}
    emb, pooled, mask = warm.text_cache["a cat"]
    assert emb.shape == (1, 5, 8) and emb.device.type == "cpu"
    assert pooled.shape == (1, 5, 8)  # audio pooled preserved across disk round-trip
    assert mask.shape == (1, 5)
    assert "TE Cache Loaded from Disk" in log.statuses


def test_changed_caption_re_encodes_only_the_delta(tmp_path):
    cold = _disk_trainer(str(tmp_path))
    cold._build_caption_hints = lambda: {"a cat": "h", "": "d"}
    cold._pre_cache_text_embeddings()

    warm = _disk_trainer(str(tmp_path))
    warm._build_caption_hints = lambda: {"a dog": "h", "": "d"}  # cat → dog
    warm._pre_cache_text_embeddings()

    assert warm.driver.calls == 1  # only the new "a dog" re-encoded
    assert "a dog" in warm.text_cache
    assert "" in warm.text_cache  # unchanged → from disk


def test_disk_cache_persists_negative_prompt(tmp_path):
    """The P1a negative prompt round-trips through disk like sample prompts."""
    cfg = {
        "cache_text_embeddings": True,
        "sample_prompts": [{"prompt": "a cat"}],
        "sample_negative_prompt": "blurry, low quality",
        "datasets": [],
    }
    cold = _disk_trainer(str(tmp_path))
    cold.config = dict(cfg)
    cold._build_caption_hints = lambda: {}
    cold._pre_cache_text_embeddings()
    assert "blurry, low quality" in cold.text_cache  # P1a warming preserved

    warm = _disk_trainer(str(tmp_path))
    warm.config = dict(cfg)
    warm._build_caption_hints = lambda: {}
    warm._pre_cache_text_embeddings()
    assert "blurry, low quality" in warm.text_cache
    assert warm.driver.calls == 0  # negative + sample served from disk
