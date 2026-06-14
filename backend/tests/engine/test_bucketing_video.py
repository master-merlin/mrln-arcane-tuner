"""Tests for the temporal (frame) axis of BucketManager.

Frame ladder correctness, video bucket selection, and a regression assertion
that image mode (frame_buckets=None) is byte-identical to plain get_bucket.
"""

from app.engine.components.bucketing import BucketManager


# ── frame_ladder ─────────────────────────────────────────────────────────


class TestFrameLadder:
    def test_4n_plus_1(self):
        ladder = BucketManager.frame_ladder(81, "4n+1")
        assert ladder[0] == 1
        # 1, 5, 9, ... 81
        assert ladder == [1] + list(range(5, 82, 4))
        assert ladder[-1] == 81
        # Every entry satisfies (f - 1) % 4 == 0.
        assert all((f - 1) % 4 == 0 for f in ladder)

    def test_8n_plus_1(self):
        ladder = BucketManager.frame_ladder(121, "8n+1")
        assert ladder[0] == 1
        assert ladder == [1] + list(range(9, 122, 8))
        assert ladder[-1] == 121
        assert all((f - 1) % 8 == 0 for f in ladder)

    def test_respects_max_frames(self):
        assert BucketManager.frame_ladder(20, "4n+1") == [1, 5, 9, 13, 17]
        assert BucketManager.frame_ladder(20, "8n+1") == [1, 9, 17]

    def test_unknown_rule_single_frame(self):
        assert BucketManager.frame_ladder(81, "weird") == [1]
        assert BucketManager.frame_ladder(81, None) == [1]


# ── get_bucket_for_video ─────────────────────────────────────────────────


class TestGetBucketForVideo:
    def test_picks_largest_valid_frame_bucket(self):
        bm = BucketManager(base_resolutions=1024, frame_rule="4n+1")
        # available 30 → largest 4n+1 <= 30 is 29.
        b = bm.get_bucket_for_video(1024, 1024, available_frames=30)
        assert b["frames"] == 29
        assert b["width"] == 1024 and b["height"] == 1024

    def test_caps_at_available(self):
        bm = BucketManager(base_resolutions=1024, frame_rule="8n+1")
        b = bm.get_bucket_for_video(1024, 1024, available_frames=10)
        # 8n+1 ladder: [1, 9, 17, ...] → largest <= 10 is 9.
        assert b["frames"] == 9

    def test_explicit_frame_buckets(self):
        bm = BucketManager(base_resolutions=512, frame_buckets=[1, 8, 16, 32])
        b = bm.get_bucket_for_video(512, 512, available_frames=20)
        assert b["frames"] == 16

    def test_fewer_than_smallest_bucket_falls_back_to_min(self):
        bm = BucketManager(base_resolutions=512, frame_buckets=[5, 9, 13])
        # available 2 < smallest (5) → falls back to the minimum bucket.
        b = bm.get_bucket_for_video(512, 512, available_frames=2)
        assert b["frames"] == 5

    def test_spatial_matches_image_bucket(self):
        bm = BucketManager(base_resolutions=1024, frame_rule="4n+1")
        vb = bm.get_bucket_for_video(1920, 1080, available_frames=17)
        ib = bm.get_bucket(1920, 1080)
        assert (vb["width"], vb["height"]) == (ib["width"], ib["height"])


# ── Image-mode regression ────────────────────────────────────────────────


class TestImageModeRegression:
    def test_frame_buckets_none_is_image_mode(self):
        bm = BucketManager(base_resolutions=1024)
        assert bm.frame_buckets is None
        # frame_bucket_for always returns 1 in image mode.
        assert bm.frame_bucket_for(99) == 1

    def test_get_bucket_unchanged_dims(self):
        """A video and image manager produce identical spatial buckets."""
        img_bm = BucketManager(base_resolutions=[512, 1024])
        for w, h in [(1024, 1024), (1920, 1080), (768, 1280), (256, 256)]:
            b = img_bm.get_bucket(w, h)
            # frames key present and defaulted to 1, dims unchanged.
            assert b["frames"] == 1
            assert b["width"] % 32 == 0 and b["height"] % 32 == 0

    def test_generated_buckets_all_carry_frames_one(self):
        bm = BucketManager(base_resolutions=1024)
        assert all(b["frames"] == 1 for b in bm.buckets)
