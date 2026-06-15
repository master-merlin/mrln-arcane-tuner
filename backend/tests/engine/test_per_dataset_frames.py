"""Per-dataset frame override — bucket-manager resolution (no GPU).

A run keeps a general ``num_frames``; a dataset may override it
(``DatasetItem.num_frames`` > 0) or inherit it (0). The override builds its own
frame ladder, capped at the override, and the result is cached per cap.
"""

from __future__ import annotations

from app.engine.core.pipeline.pipeline_data import PipelineDataMixin


class _Host(PipelineDataMixin):
    """Minimal carrier for the bucket-manager helper."""


def _host(frame_rule="8n+1"):
    h = _Host()
    h._video_frame_rule = frame_rule
    h._video_resolutions = [256]
    h._video_bm_cache = {}
    return h


def test_no_frame_rule_returns_none():
    h = _host(frame_rule=None)
    assert h._video_bucket_manager_for(81) is None


def test_caches_by_cap_and_distinct_caps_differ():
    h = _host()
    bm9a = h._video_bucket_manager_for(9)
    bm9b = h._video_bucket_manager_for(9)
    bm25 = h._video_bucket_manager_for(25)
    assert bm9a is bm9b          # cached by cap
    assert bm9a is not bm25      # different caps → different managers


def test_override_caps_the_frame_ladder():
    h = _host()
    # Abundant available frames → bucket frames == the ladder's max (the cap),
    # snapped to 8n+1: cap 9 → 9, cap 25 → 25.
    bm9 = h._video_bucket_manager_for(9)
    bm25 = h._video_bucket_manager_for(25)
    assert bm9.get_bucket_for_video(256, 256, 1000)["frames"] == 9
    assert bm25.get_bucket_for_video(256, 256, 1000)["frames"] == 25


def test_clip_shorter_than_cap_uses_available():
    h = _host()
    bm25 = h._video_bucket_manager_for(25)
    # Only 9 frames available → largest bucket ≤ 9 is 9 (not the cap 25).
    assert bm25.get_bucket_for_video(256, 256, 9)["frames"] == 9
