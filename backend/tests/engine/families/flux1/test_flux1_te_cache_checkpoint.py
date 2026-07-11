"""FLUX.1 CLIP-pooled text-embedding cache must survive a checkpoint
save/resume cycle.

Regression: ``Flux1Driver.get_te_cache``/``set_te_cache`` (driver.py) were
dead code — the real dispatch path calls ``self.get_te_cache()``/
``set_te_cache()`` on the TRAINER (``pipeline_train.py`` save sites + the
``pipeline_optimization.py`` resume merge), and ``PipelineBaseMixin``'s
default only persists ``{"te": self.text_cache}``. Flux1's
``_clip_pooled_cache`` (pooled CLIP embeds, set in ``trainer.py``) was
silently dropped on every checkpoint save. A resumed run with
offloaded/unloaded text encoders would then hit a missing pooled entry for
any caption not re-warmed this run.

Mirrors ``tests/engine/families/sdxl/test_sdxl_te_cache_checkpoint.py``.
"""

from __future__ import annotations

import structlog
import torch

from app.engine.components.checkpoints import CheckpointManager
from app.engine.core.pipeline.pipeline_optimization import _merge_te_caches
from app.engine.models.families.flux1.trainer import Flux1Trainer


def _trainer() -> Flux1Trainer:
    """Bare trainer instance with just the TE-cache state populated."""
    t = object.__new__(Flux1Trainer)
    t.logger = structlog.get_logger("test")
    t.text_cache = {}
    t._clip_pooled_cache = {}
    return t


def test_checkpoint_roundtrip_persists_pooled_cache(tmp_path):
    """Save via the trainer's real get_te_cache(), reload, and the CLIP
    pooled cache must survive — not just the T5 cache."""
    src = _trainer()
    src.text_cache = {"a cat": torch.randn(2, 8)}
    src._clip_pooled_cache = {"a cat": torch.randn(4)}

    mgr = CheckpointManager(str(tmp_path))
    mgr.save_checkpoint(step=10, components={}, config={}, te_cache=src.get_te_cache())

    state = mgr.load_checkpoint(str(tmp_path / "checkpoint-000010"))

    dst = _trainer()
    dst.set_te_cache(state.te_cache)

    assert "a cat" in dst.text_cache, "T5 cache must survive checkpoint roundtrip"
    assert "a cat" in dst._clip_pooled_cache, (
        "CLIP pooled cache must survive checkpoint roundtrip — dropped means "
        "a resumed run with offloaded TEs misses pooled entries"
    )
    assert torch.equal(dst._clip_pooled_cache["a cat"], src._clip_pooled_cache["a cat"])


def test_get_te_cache_returns_both_subcaches():
    t = _trainer()
    t.text_cache = {"x": torch.zeros(1)}
    t._clip_pooled_cache = {"x": torch.ones(1)}

    cache = t.get_te_cache()

    assert cache is not None
    assert set(cache) == {"t5", "clip_pooled"}
    assert cache["t5"] == {"x": torch.zeros(1)}
    assert cache["clip_pooled"] == {"x": torch.ones(1)}


def test_get_te_cache_none_when_empty():
    t = _trainer()
    assert t.get_te_cache() is None


def test_set_te_cache_restores_both_subcaches():
    t = _trainer()
    t.set_te_cache({"t5": {"a": torch.zeros(1)}, "clip_pooled": {"a": torch.ones(1)}})
    assert t.text_cache == {"a": torch.zeros(1)}
    assert t._clip_pooled_cache == {"a": torch.ones(1)}


def test_set_te_cache_tolerates_missing_clip_pooled_subcache():
    """Old (pre-fix) checkpoints only ever saved {"te": ...} — no pooled
    subcache at all. Restoring one must not crash; the CLIP pooled cache
    stays untouched and gets re-warmed on demand."""
    t = _trainer()
    t._clip_pooled_cache = {"stale": torch.ones(1)}

    t.set_te_cache({"te": {"a cat": torch.zeros(1)}})

    assert t.text_cache == {"a cat": torch.zeros(1)}
    assert t._clip_pooled_cache == {"stale": torch.ones(1)}


def test_resume_merge_preserves_clip_pooled_overlay_wins():
    """Mirrors the ltx2-precedent resume-merge test but for Flux1's second
    (CLIP pooled) subcache: the freshly-warmed pooled entries for THIS run
    must survive the merge with the checkpoint's restored pooled cache."""
    checkpoint_cache = {
        "t5": {"a cat": torch.tensor([1.0]), "a dog": torch.tensor([2.0])},
        "clip_pooled": {"a cat": torch.tensor([10.0]), "a dog": torch.tensor([20.0])},
    }
    this_run_warm = {
        "t5": {"a cat": torch.tensor([99.0]), "NEW prompt": torch.tensor([3.0])},
        "clip_pooled": {"a cat": torch.tensor([999.0]), "NEW prompt": torch.tensor([30.0])},
    }

    merged = _merge_te_caches(checkpoint_cache, this_run_warm)

    t = _trainer()
    t.set_te_cache(merged)

    assert torch.equal(t.text_cache["a cat"], torch.tensor([99.0]))  # overlay wins
    assert torch.equal(t.text_cache["a dog"], torch.tensor([2.0]))  # checkpoint-only kept
    assert "NEW prompt" in t.text_cache
    assert torch.equal(t._clip_pooled_cache["a cat"], torch.tensor([999.0]))  # overlay wins
    assert torch.equal(t._clip_pooled_cache["a dog"], torch.tensor([20.0]))  # checkpoint-only kept
    assert "NEW prompt" in t._clip_pooled_cache
