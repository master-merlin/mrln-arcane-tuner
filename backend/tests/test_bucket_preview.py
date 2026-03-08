"""Tests for the bucket_preview utility."""
import pytest
from app.core.bucket_preview import preview_buckets


# ---------- fixtures ----------

@pytest.fixture
def landscape_images() -> list[dict]:
    """A small set of landscape images at different sizes."""
    return [
        {"path": "a.png", "width": 1920, "height": 1080},
        {"path": "b.png", "width": 1280, "height": 720},
        {"path": "c.png", "width": 800, "height": 600},
    ]


@pytest.fixture
def portrait_images() -> list[dict]:
    return [
        {"path": "p1.png", "width": 768, "height": 1024},
        {"path": "p2.png", "width": 512, "height": 768},
    ]


# ---------- tests ----------

class TestPreviewBuckets:
    """Tests for preview_buckets()."""

    def test_empty_resolutions_returns_unmodified(self, landscape_images: list[dict]):
        """When resolutions list is empty, images pass through unchanged."""
        result = preview_buckets(landscape_images, [])
        assert result is landscape_images
        assert "buckets" not in result[0]

    def test_kohya_mode_assigns_single_bucket(self, landscape_images: list[dict]):
        """Kohya mode should assign exactly one bucket per image."""
        result = preview_buckets(landscape_images, [1024], "kohya")
        for img in result:
            assert "buckets" in img
            assert len(img["buckets"]) == 1
            bucket = img["buckets"][0]
            assert "width" in bucket
            assert "height" in bucket
            assert "target_resolution" in bucket
            assert bucket["target_resolution"] == 1024

    def test_multi_mode_can_assign_multiple_buckets(self, landscape_images: list[dict]):
        """Multi mode with multiple resolutions should assign ≥1 bucket per qualifying image."""
        result = preview_buckets(landscape_images, [768, 1024], "multi")
        # The 1920×1080 image should qualify for both 768 and 1024
        large_img = next(img for img in result if img["path"] == "a.png")
        assert len(large_img["buckets"]) >= 2

    def test_bucket_dimensions_are_divisible_by_32(self, landscape_images: list[dict]):
        """All assigned buckets must have dimensions divisible by 32."""
        result = preview_buckets(landscape_images, [1024], "kohya")
        for img in result:
            for bucket in img["buckets"]:
                assert bucket["width"] % 32 == 0, f"width {bucket['width']} not divisible by 32"
                assert bucket["height"] % 32 == 0, f"height {bucket['height']} not divisible by 32"

    def test_zero_dimension_images_get_empty_buckets(self):
        """Images with zero dimensions should get an empty bucket list."""
        images = [{"path": "bad.png", "width": 0, "height": 0}]
        result = preview_buckets(images, [1024])
        assert result[0]["buckets"] == []

    def test_portrait_images_get_portrait_buckets(self, portrait_images: list[dict]):
        """Portrait images should receive buckets where height > width."""
        result = preview_buckets(portrait_images, [1024], "kohya")
        for img in result:
            bucket = img["buckets"][0]
            assert bucket["height"] >= bucket["width"], (
                f"Portrait image {img['path']} got landscape bucket {bucket['width']}x{bucket['height']}"
            )

    def test_does_not_mutate_original_keys(self, landscape_images: list[dict]):
        """preview_buckets should only add 'buckets', not remove existing keys."""
        original_keys = set(landscape_images[0].keys())
        preview_buckets(landscape_images, [1024])
        new_keys = set(landscape_images[0].keys())
        assert original_keys.issubset(new_keys)
        assert "buckets" in new_keys
