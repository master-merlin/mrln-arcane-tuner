"""No-GPU tests for temporal_coverage='sliding' (Phase 2, Option A)."""

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
