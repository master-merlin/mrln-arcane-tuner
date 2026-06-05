"""
Tests for core/dataset/ helper modules — geometry, scan_helpers, media_helpers.

These are pure-function unit tests with no DB or event dependencies.
"""


from PIL import Image

from app.core.dataset.geometry import calculate_target_dims, ar_to_display
from app.core.dataset.scan_helpers import (
    extract_media_dimensions,
    classify_orientation,
    build_media_entry,
    compute_majority_ar,
    compute_harmonization_score,
    compute_caption_coverage,
)
from app.core.dataset.media_helpers import (
    invalidate_mask_files,
    refresh_media_metadata_after_change,
)


# ── geometry.py ──────────────────────────────────────────────────────────


class TestCalculateTargetDims:
    """Tests for calculate_target_dims."""

    def test_landscape_32_aligned(self):
        """Landscape: width=long_side, height derived, both multiples of 32."""
        w, h = calculate_target_dims(1024, 1.5, "landscape")
        assert w == 1024
        assert h % 32 == 0
        assert h > 0

    def test_portrait_32_aligned(self):
        """Portrait: height=long_side, width derived, both multiples of 32."""
        w, h = calculate_target_dims(1024, 0.75, "portrait")
        assert h == 1024
        assert w % 32 == 0
        assert w > 0

    def test_squared_equal(self):
        """AR=1.0 should give equal width and height."""
        w, h = calculate_target_dims(1024, 1.0, "squared")
        assert w == 1024
        assert h == 1024

    def test_minimum_32(self):
        """Very small long_side should still return at least 32."""
        w, h = calculate_target_dims(32, 2.0, "landscape")
        assert w >= 32
        assert h >= 32

    def test_large_ar(self):
        """Extreme AR (e.g. 4.0) should still produce valid aligned dims."""
        w, h = calculate_target_dims(1024, 4.0, "landscape")
        assert w == 1024
        assert h % 32 == 0
        assert 0 < h < w


class TestArToDisplay:
    """Tests for ar_to_display."""

    def test_square(self):
        assert ar_to_display(1.0, "squared") == "1:1"

    def test_near_square(self):
        """Slight deviation from 1.0 should still return 1:1."""
        assert ar_to_display(1.005, "squared") == "1:1"

    def test_landscape(self):
        result = ar_to_display(1.5, "landscape")
        assert ":" in result
        parts = result.split(":")
        assert int(parts[0]) > int(parts[1])  # W > H for landscape

    def test_portrait_swaps(self):
        """Portrait display should put H:W."""
        result = ar_to_display(0.75, "portrait")
        parts = result.split(":")
        assert int(parts[0]) > int(parts[1])  # Portrait convention: tall:wide


# ── scan_helpers.py ──────────────────────────────────────────────────────


class TestExtractMediaDimensions:
    """Tests for extract_media_dimensions."""

    def test_image_dimensions(self, tmp_path):
        """Should return (width, height) for a valid image."""
        img_path = str(tmp_path / "test.png")
        Image.new("RGB", (300, 200), "blue").save(img_path)
        w, h = extract_media_dimensions(img_path, ".png")
        assert (w, h) == (300, 200)

    def test_invalid_file_returns_zero(self, tmp_path):
        """Invalid file should return (0, 0)."""
        bad_path = str(tmp_path / "corrupt.png")
        with open(bad_path, "w") as f:
            f.write("not an image")
        w, h = extract_media_dimensions(bad_path, ".png")
        assert (w, h) == (0, 0)

    def test_nonexistent_file_returns_zero(self):
        """Missing file should return (0, 0)."""
        w, h = extract_media_dimensions("/nonexistent/img.png", ".png")
        assert (w, h) == (0, 0)


class TestClassifyOrientation:
    """Tests for classify_orientation."""

    def test_landscape(self):
        assert classify_orientation(200, 100) == "landscape"

    def test_portrait(self):
        assert classify_orientation(100, 200) == "portrait"

    def test_squared(self):
        assert classify_orientation(100, 100) == "squared"


class TestBuildMediaEntry:
    """Tests for build_media_entry."""

    def test_basic_entry(self, tmp_path):
        """Should produce a metadata dict with dimensions, ratio, orientation."""
        img_path = str(tmp_path / "photo.png")
        Image.new("RGB", (400, 200), "red").save(img_path)

        entry = build_media_entry(img_path, "photo", ".png", str(tmp_path), {}, 400, 200)
        assert entry["width"] == 400
        assert entry["height"] == 200
        assert entry["orientation"] == "landscape"
        assert entry["aspect_ratio"] == round(400 / 200, 5)
        assert entry["enabled"] is True
        assert entry["has_mask"] is False

    def test_preserves_enabled_false(self, tmp_path):
        """Should preserve enabled=False from existing metadata."""
        img_path = str(tmp_path / "disabled.png")
        Image.new("RGB", (100, 100), "red").save(img_path)

        existing = {"enabled": False}
        entry = build_media_entry(img_path, "disabled", ".png", str(tmp_path), existing, 100, 100)
        assert entry["enabled"] is False

    def test_detects_mask(self, tmp_path):
        """Should detect mask file in masks/ subdirectory."""
        img_path = str(tmp_path / "masked.png")
        Image.new("RGB", (100, 100), "red").save(img_path)

        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        mask_img = Image.new("L", (100, 100), 255)
        mask_img.save(str(masks_dir / "masked.png"))

        entry = build_media_entry(img_path, "masked", ".png", str(tmp_path), {}, 100, 100)
        assert entry["has_mask"] is True
        assert "mask_info" in entry
        assert entry["mask_info"]["width"] == 100


class TestComputeMajorityAr:
    """Tests for compute_majority_ar."""

    def test_normal(self):
        assert compute_majority_ar([1.5, 1.5, 1.0]) == 1.5

    def test_single(self):
        assert compute_majority_ar([2.0]) == 2.0

    def test_empty(self):
        assert compute_majority_ar([]) is None


class TestComputeHarmonizationScore:
    """Tests for compute_harmonization_score."""

    def test_uniform_dataset(self):
        """All same AR → score ≈ 1.0."""
        meta = {
            "a.png": {"width": 300, "height": 200, "aspect_ratio": 1.5, "orientation": "landscape"},
            "b.png": {"width": 300, "height": 200, "aspect_ratio": 1.5, "orientation": "landscape"},
        }
        score, updated = compute_harmonization_score(meta)
        assert score == 1.0
        assert updated["a.png"]["is_majority_ar"] is True

    def test_mixed_dataset(self):
        """Mixed ARs → score < 1.0."""
        meta = {
            "a.png": {"width": 300, "height": 200, "aspect_ratio": 1.5, "orientation": "landscape"},
            "b.png": {"width": 400, "height": 200, "aspect_ratio": 2.0, "orientation": "landscape"},
            "c.png": {"width": 300, "height": 200, "aspect_ratio": 1.5, "orientation": "landscape"},
        }
        score, updated = compute_harmonization_score(meta)
        assert 0 < score < 1.0
        assert updated["b.png"]["is_majority_ar"] is False

    def test_empty_metadata(self):
        """Empty metadata → score 0.0."""
        score, updated = compute_harmonization_score({})
        assert score == 0.0

    def test_annotates_target_dims(self):
        """Should add target_width/target_height fields."""
        meta = {
            "a.png": {"width": 1024, "height": 768, "aspect_ratio": round(1024/768, 5), "orientation": "landscape"},
        }
        _, updated = compute_harmonization_score(meta)
        assert "target_width" in updated["a.png"]
        assert "target_height" in updated["a.png"]
        assert updated["a.png"]["target_width"] % 32 == 0
        assert updated["a.png"]["target_height"] % 32 == 0


class TestComputeCaptionCoverage:
    """Tests for compute_caption_coverage."""

    def test_full_coverage(self):
        assert compute_caption_coverage({"a", "b"}, {"a", "b", "c"}, 2, 3) is True

    def test_partial_coverage(self):
        assert compute_caption_coverage({"a", "b"}, {"a"}, 2, 1) is False

    def test_no_media_no_captions(self):
        assert compute_caption_coverage(set(), set(), 0, 0) is False

    def test_no_media_with_captions(self):
        assert compute_caption_coverage(set(), {"a"}, 0, 1) is True


# ── media_helpers.py ─────────────────────────────────────────────────────


class TestInvalidateMaskFiles:
    """Tests for invalidate_mask_files."""

    def test_removes_existing_files(self, tmp_path):
        """Should delete mask, masked image, and masked caption."""
        masks_dir = tmp_path / "masks"
        masked_dir = tmp_path / "masked"
        masks_dir.mkdir()
        masked_dir.mkdir()

        (masks_dir / "photo.png").write_text("mask")
        (masked_dir / "photo.jpg").write_text("masked_img")
        (masked_dir / "photo.txt").write_text("masked_cap")

        invalidate_mask_files(str(tmp_path), "photo", reason="test")

        assert not (masks_dir / "photo.png").exists()
        assert not (masked_dir / "photo.jpg").exists()
        assert not (masked_dir / "photo.txt").exists()

    def test_no_crash_on_missing_dirs(self, tmp_path):
        """Should not crash if masks/ or masked/ directories don't exist."""
        invalidate_mask_files(str(tmp_path), "nonexistent", reason="test")


class TestRefreshMediaMetadataAfterChange:
    """Tests for refresh_media_metadata_after_change."""

    def test_clears_hash_and_mask(self, tmp_path):
        """Should clear solid_hash and nullify mask fields."""
        img_path = str(tmp_path / "img.png")
        Image.new("RGB", (100, 100), "red").save(img_path)

        metadata = {
            "img.png": {
                "width": 100, "height": 100,
                "solid_hash": "abc123",
                "has_mask": True,
                "has_masked": True,
                "has_masked_caption": True,
                "mask_info": {"width": 100},
                "size_bytes": 50,
            }
        }
        refresh_media_metadata_after_change(metadata, "img.png", img_path)

        entry = metadata["img.png"]
        assert "solid_hash" not in entry
        assert entry["has_mask"] is False
        assert entry["has_masked"] is False
        assert entry["has_masked_caption"] is False
        assert "mask_info" not in entry
        assert entry["size_bytes"] > 0

    def test_updates_dims_on_crop(self, tmp_path):
        """When new_dims is provided, should update width/height/AR/orientation."""
        img_path = str(tmp_path / "cropped.png")
        Image.new("RGB", (200, 100), "blue").save(img_path)

        metadata = {
            "cropped.png": {"width": 400, "height": 300, "size_bytes": 100}
        }
        refresh_media_metadata_after_change(metadata, "cropped.png", img_path, new_dims=(200, 100))

        entry = metadata["cropped.png"]
        assert entry["width"] == 200
        assert entry["height"] == 100
        assert entry["orientation"] == "landscape"
        assert entry["aspect_ratio"] == 2.0

    def test_missing_key_is_noop(self, tmp_path):
        """Calling with a non-existent key should do nothing."""
        img_path = str(tmp_path / "fake.png")
        metadata = {}
        refresh_media_metadata_after_change(metadata, "missing.png", img_path)
        assert metadata == {}
