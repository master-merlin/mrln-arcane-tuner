"""No-GPU tests for temporal_coverage='sliding' (Phase 2, Option A)."""

import types
from unittest.mock import MagicMock

from app.engine.core.pipeline.pipeline_data import video_trim_extra_key


class TestSlideExtraKey:
    def test_image_returns_empty(self):
        assert video_trim_extra_key({"is_video": False}) == ""

    def test_first_mode_untrimmed_unchanged(self):
        # No temporal_mode → byte-identical to the pre-Phase-2 key.
        item = {"is_video": True, "trim_start_s": 0.0, "trim_end_s": None}
        assert video_trim_extra_key(item) == "t0.0-None"

    def test_sliding_appends_full_frame_count(self):
        item = {
            "is_video": True, "trim_start_s": 0.0, "trim_end_s": None,
            "temporal_mode": "sliding", "cache_frames": 81,
        }
        assert video_trim_extra_key(item) == "t0.0-None-slideF81"

    def test_sliding_distinct_from_first_mode(self):
        first = {"is_video": True, "trim_start_s": 0.0, "trim_end_s": None}
        slide = {**first, "temporal_mode": "sliding", "cache_frames": 81}
        assert video_trim_extra_key(first) != video_trim_extra_key(slide)


def _mixin_with(coverage, max_secs=0.0):
    """Bare object carrying just the attrs _emit_temporal_items reads."""
    from app.engine.core.pipeline.pipeline_data import PipelineDataMixin

    obj = types.SimpleNamespace()
    obj._temporal_coverage = coverage
    obj._sliding_max_clip_seconds = max_secs
    obj._window_overlap = 0.0
    obj._max_windows = 10
    obj.logger = MagicMock()  # structlog-style: accepts arbitrary kwargs
    obj._emit_temporal_items = types.MethodType(
        PipelineDataMixin._emit_temporal_items, obj
    )
    obj._compute_tiled_windows = types.MethodType(
        PipelineDataMixin._compute_tiled_windows, obj
    )
    return obj


VIDEO_ITEM = {
    "is_video": True, "target_frames": 25,
    "trim_start_s": 0.0, "trim_end_s": None, "id": "c",
}


class TestEmitSliding:
    def test_sliding_emits_one_fullclip_item_per_repeat(self):
        obj = _mixin_with("sliding")
        out = obj._emit_temporal_items(
            base_item=VIDEO_ITEM, trim_start_s=0.0, end_s=10.0,
            window_span_s=1.0, repeats=3, full_clip_frames=81,
        )
        assert len(out) == 3
        for it in out:
            assert it["temporal_mode"] == "sliding"
            assert it["cache_frames"] == 81
            assert it["target_frames"] == 25       # window length preserved
            assert it["trim_end_s"] is None         # full clip, original trim

    def test_no_slide_room_falls_back_to_first(self):
        # full_clip_frames == target_frames → no room → first (verbatim clone).
        obj = _mixin_with("sliding")
        out = obj._emit_temporal_items(
            base_item=VIDEO_ITEM, trim_start_s=0.0, end_s=10.0,
            window_span_s=1.0, repeats=2, full_clip_frames=25,
        )
        assert len(out) == 2
        assert all("temporal_mode" not in it for it in out)

    def test_over_long_clip_downgrades_to_tiled(self):
        obj = _mixin_with("sliding", max_secs=5.0)
        out = obj._emit_temporal_items(
            base_item=VIDEO_ITEM, trim_start_s=0.0, end_s=20.0,  # 20s > 5s guard
            window_span_s=2.0, repeats=1, full_clip_frames=81,
        )
        assert all(it.get("temporal_mode") != "sliding" for it in out)
        assert any(it.get("trim_end_s") is not None for it in out)

    def test_first_and_tiled_unchanged(self):
        first = _mixin_with("first")._emit_temporal_items(
            base_item=VIDEO_ITEM, trim_start_s=0.0, end_s=10.0,
            window_span_s=1.0, repeats=2, full_clip_frames=81,
        )
        assert len(first) == 2 and all("temporal_mode" not in it for it in first)
