"""PR5 — pure paired-control inventory helpers."""

from __future__ import annotations

import pytest

from app.engine.core.pipeline.edit_inventory import (
    CONTROL_VARIANT_ROOT,
    build_control_fields,
    control_target_dims,
    control_variant,
    is_video_control,
)


class TestControlVariant:
    def test_control_slot(self):
        assert control_variant("control/img1.jpg") == "control"

    def test_second_slot(self):
        assert control_variant("control_2/img1.png") == "control_2"

    def test_root_as_control(self):
        assert control_variant("img1.png") == CONTROL_VARIANT_ROOT

    def test_backslash_normalized(self):
        assert control_variant("control\\img1.jpg") == "control"


class TestControlTargetDims:
    def test_zero_follows_target(self):
        assert control_target_dims(768, 1024, 0) == (768, 1024)

    def test_nonzero_uses_bucket(self):
        def bucket_for(w, h, base):
            assert base == 1024
            return {"width": 1024, "height": 1024}
        assert control_target_dims(768, 768, 1024, bucket_for) == (1024, 1024)

    def test_nonzero_without_bucketfn_square(self):
        assert control_target_dims(768, 512, 1024) == (1024, 1024)


class TestIsVideoControl:
    def test_image_exts_are_not_video(self):
        assert is_video_control("control/img1.jpg") is False
        assert is_video_control("control/img1.png") is False
        assert is_video_control("control/img1.webp") is False

    def test_video_exts_are_video(self):
        assert is_video_control("control/clip.mp4") is True
        assert is_video_control("control/clip.webm") is True
        assert is_video_control("control/clip.mkv") is True

    def test_mov_is_video_despite_target_scanner_asymmetry(self):
        # .mov is legal as a CONTROL ext but NOT in the target scanner's
        # VIDEO_EXTENSIONS — the two ext-sets are not symmetric.
        assert is_video_control("control/clip.mov") is True

    def test_case_insensitive(self):
        assert is_video_control("control/CLIP.MP4") is True
        assert is_video_control("control/IMG.JPG") is False


class TestBuildControlFields:
    def _cache_dir_for(self, res_str, variant):
        return f"/cache/{variant}/{res_str}"

    def test_complete_pair(self):
        fields = build_control_fields(
            ["control/img1.jpg"], "/ds", 1, 512, 512, 0, self._cache_dir_for,
        )
        assert fields is not None
        assert fields["control_rel_paths"] == ["control/img1.jpg"]
        assert fields["control_paths"][0].replace("\\", "/") == "/ds/control/img1.jpg"
        assert fields["control_variants"] == ["control"]
        assert fields["control_dims"] == [(512, 512)]
        assert fields["control_cache_dirs"] == ["/cache/control/512x512"]
        assert fields["control_is_video"] == [False]

    def test_video_control_flagged(self):
        fields = build_control_fields(
            ["control/clip.mp4"], "/ds", 1, 512, 512, 0, self._cache_dir_for,
        )
        assert fields["control_is_video"] == [True]

    def test_mixed_slots_flagged_independently(self):
        fields = build_control_fields(
            ["control/a.jpg", "control_2/a.mp4"],
            "/ds", 2, 512, 512, 0, self._cache_dir_for,
        )
        assert fields["control_is_video"] == [False, True]

    def test_partial_pair_returns_none(self):
        # Model wants 1 control, pair resolved zero.
        assert build_control_fields([], "/ds", 1, 512, 512, 0, self._cache_dir_for) is None

    def test_multi_control_truncates_to_model_inputs(self):
        fields = build_control_fields(
            ["control/a.jpg", "control_2/a.png", "control_3/a.webp"],
            "/ds", 2, 512, 512, 0, self._cache_dir_for,
        )
        assert fields["control_variants"] == ["control", "control_2"]

    def test_multi_control_partial_when_fewer_than_inputs(self):
        # Model wants 3, only 2 resolved → partial.
        assert build_control_fields(
            ["control/a.jpg", "control_2/a.png"],
            "/ds", 3, 512, 512, 0, self._cache_dir_for,
        ) is None

    def test_control_resolution_changes_cache_res(self):
        def bucket_for(w, h, base):
            return {"width": 1024, "height": 1024}
        fields = build_control_fields(
            ["control/img1.jpg"], "/ds", 1, 512, 512, 1024,
            self._cache_dir_for, bucket_for,
        )
        assert fields["control_cache_dirs"] == ["/cache/control/1024x1024"]
        assert fields["control_dims"] == [(1024, 1024)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
