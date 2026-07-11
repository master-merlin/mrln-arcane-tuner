"""still_resolutions: stills in a video job bucket at their OWN resolutions.

Phase-3 contract (original field description, recovered from 37d39889^):
empty list means INHERIT from `resolutions`. The field was deleted in W4-1
as a dead knob; this suite is its REAL consumer contract — see
test_config_field_consumers.py docstring for the circular-validator trap.
"""

from app.engine.core.pipeline.pipeline_data import resolve_still_resolutions
from app.engine.components.bucketing import BucketManager


class TestResolveStillResolutions:
    def test_unset_inherits_resolutions(self):
        cfg = {"resolutions": [768]}
        assert resolve_still_resolutions(cfg, is_video_family=True) == [768]

    def test_empty_inherits_resolutions(self):
        cfg = {"resolutions": [768], "still_resolutions": []}
        assert resolve_still_resolutions(cfg, is_video_family=True) == [768]

    def test_set_wins_on_video_family(self):
        cfg = {"resolutions": [768], "still_resolutions": [1024, 1536]}
        assert resolve_still_resolutions(cfg, is_video_family=True) == [1024, 1536]

    def test_image_family_always_inherits(self):
        # A stale key on an image-family job (old DB row predating the
        # W4-2 allowlist) must not change image bucketing.
        cfg = {"resolutions": [1024], "still_resolutions": [512]}
        assert resolve_still_resolutions(cfg, is_video_family=False) == [1024]


class TestStillBucketSelection:
    """The stills branch buckets via still_bucket_manager, video via the
    per-dataset video manager — proven at BucketManager level."""

    def test_still_buckets_scale_with_still_resolutions(self):
        video_mgr = BucketManager(base_resolutions=[768])
        still_mgr = BucketManager(base_resolutions=[1536])
        # A large square source: video manager caps at 768-scale,
        # still manager offers 1536-scale.
        vbucket = video_mgr.get_bucket(2048, 2048)
        sbucket = still_mgr.get_bucket(2048, 2048)
        vw, vh = vbucket["width"], vbucket["height"]
        sw, sh = sbucket["width"], sbucket["height"]
        assert max(sw, sh) > max(vw, vh)
        assert max(sw, sh) >= 1280  # 1536-scale bucket (32-snapped)

    def test_field_is_declared_with_inherit_default(self):
        from app.engine.models.base import BaseTrainingConfig

        f = BaseTrainingConfig.model_fields["still_resolutions"]
        assert f.default == []
        assert "inherit" in (f.description or "").lower()
