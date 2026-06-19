"""Resume must MERGE the checkpoint TE-cache under the freshly-warmed one.

Regression for: a CFG/sample run resumed from a checkpoint hit "caption not
pre-cached" at sample time. The trainer warms this run's prompts (incl. a NEW
sample prompt + the CFG unconditional), then resume restored the checkpoint's
te_cache and REPLACED the warmed one — evicting the new prompt. The merge keeps
the warmed entries (overlay-wins) so new/changed sample prompts survive.
"""

from __future__ import annotations

from app.engine.core.pipeline.pipeline_optimization import _merge_te_caches


def test_overlay_wins_and_unions_keys():
    base = {"te": {"a cat": 1, "a dog": 2}}
    overlay = {"te": {"a cat": 99, "NEW sample prompt": 3, "": 0}}
    merged = _merge_te_caches(base, overlay)
    assert merged["te"] == {
        "a cat": 99,            # overlay (this run) wins
        "a dog": 2,             # checkpoint-only entry kept
        "NEW sample prompt": 3,  # new sample prompt survives
        "": 0,                  # CFG unconditional survives
    }


def test_unions_distinct_subcaches():
    # Multi-cache families (flux1: t5 + clip_pooled) merge per subcache.
    base = {"t5": {"x": 1}, "clip_pooled": {"x": 1}}
    overlay = {"t5": {"y": 2}}
    merged = _merge_te_caches(base, overlay)
    assert merged["t5"] == {"x": 1, "y": 2}
    assert merged["clip_pooled"] == {"x": 1}


def test_handles_none_and_empty():
    assert _merge_te_caches(None, None) == {}
    assert _merge_te_caches({"te": {"a": 1}}, None) == {"te": {"a": 1}}
    assert _merge_te_caches(None, {"te": {"a": 1}}) == {"te": {"a": 1}}
