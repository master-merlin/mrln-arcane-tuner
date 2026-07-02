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
    t._resolve_te_cache_dirs = lambda: []  # disk cache off by default (in-memory tests)
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


def test_warm_includes_expanded_sample_prompts():
    """Sampling runs AFTER the TE offload and serves prompts from the cache, so
    the expanded sample prompts must be warmed (else 'NoneType' is not callable).
    """
    from app.engine.core.sampling import expand_prompt_wildcards

    t = _trainer()
    t.config = {
        "cache_text_embeddings": True,
        "global_triggerword": "airwolf",
        "sample_prompts": [{"prompt": "A [triggerword] flying over the desert"}],
    }
    t._pre_cache_text_embeddings()

    # The sampler will request the wildcard-EXPANDED string — it must be cached.
    expanded = expand_prompt_wildcards(
        "A [triggerword] flying over the desert", t.config
    )
    assert "[triggerword]" not in expanded  # expansion actually happened
    assert expanded in t.text_cache


# ── CFG unconditional warming (so guidance_scale>1 works after TE offload) ──


def _sampling_trainer() -> Wan21Trainer:
    """Trainer with a sample prompt + triggerword, no training captions."""
    t = _trainer()
    t.config = {
        "cache_text_embeddings": True,
        "sample_prompts": [{"prompt": "a [triggerword] flying over the desert"}],
        "global_triggerword": "sks",
    }
    t._build_caption_hints = lambda: {}  # isolate sample-prompt warming
    return t


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


def test_sampler_encodes_via_trainer_cache_not_the_offloaded_driver():
    """The WAN sampler must route prompt encoding through the trainer's cached
    encode_text (survives TE offload), not the driver's (None after offload).
    """
    from app.engine.models.families.wan_shared.sampler_base import WanVideoSamplerBase

    t = _trainer()
    t._pre_cache_text_embeddings()
    # Offload the encoder exactly as run_trainer does before sampling.
    t.driver.text_encoder = None
    t.text_encoder = None
    calls_before = t.driver.calls

    # get_primary_model() supplies the dtype probe the sampler needs.
    param = torch.nn.Parameter(torch.zeros(1))
    t.driver.get_primary_model = lambda: torch.nn.ParameterList([param])

    sampler = object.__new__(WanVideoSamplerBase)
    sampler.pipeline = t
    sampler.device = torch.device("cpu")

    emb = sampler.encode_prompt("a cat")  # a warmed training caption

    assert emb.shape == (1, 5, 8)
    assert t.driver.calls == calls_before  # served from cache; offloaded TE untouched


# ── Disk-backed cache (P1c — mirror the image families' TextEmbeddingCache) ──


class _FakeLogWriter:
    def __init__(self) -> None:
        self.statuses: list[str] = []

    def status(self, msg: str) -> None:
        self.statuses.append(msg)


def _disk_trainer(tmp: str, log_writer=None) -> Wan21Trainer:
    t = _trainer()
    t._log_writer = log_writer
    t._resolve_te_cache_dirs = lambda: [tmp]
    return t


def test_cold_run_writes_disk_cache(tmp_path):
    """First run encodes via the stub TE and persists .safetensors to te1/."""
    import os

    t = _disk_trainer(str(tmp_path))
    t._pre_cache_text_embeddings()

    te1 = os.path.join(str(tmp_path), "embeddings", "none", "te1")
    files = os.listdir(te1)
    assert files, "cold run must persist embeddings to disk"
    # {"a cat","a dog",""} → 3 caption files
    assert len([f for f in files if f.endswith(".safetensors")]) == 3
    assert t.driver.calls > 0  # the stub TE was exercised


def test_warm_run_loads_from_disk_without_re_encoding(tmp_path):
    """Second run over the same captions never touches the stub TE."""
    log = _FakeLogWriter()
    cold = _disk_trainer(str(tmp_path), log)
    cold._pre_cache_text_embeddings()

    warm = _disk_trainer(str(tmp_path), log)
    warm._pre_cache_text_embeddings()

    assert warm.driver.calls == 0  # zero encode calls — everything came from disk
    assert set(warm.text_cache) == {"a cat", "a dog", ""}
    cat = warm.text_cache["a cat"]
    assert cat.shape == (1, 5, 8) and cat.device.type == "cpu"
    assert "TE Cache Loaded from Disk" in log.statuses


def test_changed_caption_re_encodes_only_the_delta(tmp_path):
    """A changed caption invalidates (re-encodes); unchanged ones load from disk."""
    cold = _disk_trainer(str(tmp_path))
    cold._build_caption_hints = lambda: {"a cat": "h", "": "d"}
    cold._pre_cache_text_embeddings()

    warm = _disk_trainer(str(tmp_path))
    warm._build_caption_hints = lambda: {"a dog": "h", "": "d"}  # cat → dog
    warm._pre_cache_text_embeddings()

    assert warm.driver.calls == 1  # exactly one encode batch for the new "a dog"
    assert "a dog" in warm.text_cache
    assert "" in warm.text_cache  # unchanged → loaded from disk


def test_disk_cache_persists_negative_prompt(tmp_path):
    """The P1a negative prompt round-trips through disk like sample prompts."""
    cfg = {
        "cache_text_embeddings": True,
        "sample_prompts": [{"prompt": "a cat"}],
        "sample_negative_prompt": "blurry, low quality",
    }
    cold = _disk_trainer(str(tmp_path))
    cold.config = dict(cfg)
    cold._build_caption_hints = lambda: {}
    cold._pre_cache_text_embeddings()
    assert "blurry, low quality" in cold.text_cache  # P1a warming preserved

    # Warm run: the configured negative must load from disk, not re-encode.
    warm = _disk_trainer(str(tmp_path))
    warm.config = dict(cfg)
    warm._build_caption_hints = lambda: {}
    warm._pre_cache_text_embeddings()
    assert "blurry, low quality" in warm.text_cache
    assert warm.driver.calls == 0  # negative + sample both served from disk
