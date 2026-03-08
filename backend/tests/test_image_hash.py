"""
Tests for the rotation-invariant image hashing module.
Covers: solide_hash_robust, measure_similarity, _hex_to_bool_grid,
edge cases (small images, RGBA, grayscale).
"""


import cv2
import numpy as np
import pytest

from app.core.image_hash import (
    solide_hash_robust,
    measure_similarity,
    _hex_to_bool_grid,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _write_test_image(path: str, width: int = 200, height: int = 200,
                      color: tuple = (128,), pattern: str = "gradient",
                      channels: int = 1):
    """Create a test image on disk and return the path."""
    if pattern == "gradient":
        if channels == 1:
            img = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
        else:
            base = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
            img = np.stack([base] * channels, axis=-1)
    elif pattern == "solid":
        if channels == 1:
            img = np.full((height, width), color[0], dtype=np.uint8)
        else:
            img = np.full((height, width, channels), color, dtype=np.uint8)
    elif pattern == "random":
        rng = np.random.RandomState(42)
        if channels == 1:
            img = rng.randint(0, 256, (height, width), dtype=np.uint8)
        else:
            img = rng.randint(0, 256, (height, width, channels), dtype=np.uint8)
    elif pattern == "circle":
        img = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(img, (width // 2, height // 2), min(width, height) // 3, 255, -1)
    else:
        img = np.zeros((height, width), dtype=np.uint8)

    cv2.imwrite(path, img)
    return path


# ── Basic Hash Generation ────────────────────────────────────────────────


class TestSolideHashRobust:
    """Tests for the main hash function."""

    def test_returns_hex_string(self, tmp_path):
        """Hash should be a non-empty hex string."""
        path = _write_test_image(str(tmp_path / "test.png"))
        h = solide_hash_robust(path)
        assert isinstance(h, str)
        assert len(h) > 0
        # Verify it's valid hex
        int(h, 16)

    def test_deterministic_output(self, tmp_path):
        """Same image should always produce the same hash."""
        path = _write_test_image(str(tmp_path / "test.png"))
        h1 = solide_hash_robust(path)
        h2 = solide_hash_robust(path)
        assert h1 == h2

    def test_different_images_different_hashes(self, tmp_path):
        """Visually distinct images should produce different hashes."""
        path_a = _write_test_image(str(tmp_path / "a.png"), pattern="gradient")
        path_b = _write_test_image(str(tmp_path / "b.png"), pattern="circle")
        assert solide_hash_robust(path_a) != solide_hash_robust(path_b)

    def test_custom_size_parameter(self, tmp_path):
        """Different hash sizes should produce different-length outputs."""
        path = _write_test_image(str(tmp_path / "test.png"))
        h_small = solide_hash_robust(path, size=16)
        h_large = solide_hash_robust(path, size=48)
        assert len(h_small) < len(h_large)

    def test_invalid_path_raises(self):
        """Non-existent image should raise."""
        with pytest.raises(Exception):
            solide_hash_robust("/nonexistent/image.png")


# ── Rotation Invariance ──────────────────────────────────────────────────


class TestRotationInvariance:
    """Hash should be similar for rotated versions of the same image."""

    def test_90_degree_rotation_similar(self, tmp_path):
        """Same image rotated 90° should have high similarity."""
        base = _write_test_image(str(tmp_path / "base.png"), pattern="circle", width=200, height=200)

        # Create a 90° rotated version
        img = cv2.imread(base, cv2.IMREAD_GRAYSCALE)
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        rot_path = str(tmp_path / "rot90.png")
        cv2.imwrite(rot_path, rotated)

        h1 = solide_hash_robust(base)
        h2 = solide_hash_robust(rot_path)
        sim = measure_similarity(h1, h2)
        # Should be high (algorithm aims for invariance)
        assert sim > 0.5

    def test_180_degree_rotation_similar(self, tmp_path):
        """Same image rotated 180° should have high similarity."""
        base = _write_test_image(str(tmp_path / "base.png"), pattern="circle", width=200, height=200)

        img = cv2.imread(base, cv2.IMREAD_GRAYSCALE)
        rotated = cv2.rotate(img, cv2.ROTATE_180)
        rot_path = str(tmp_path / "rot180.png")
        cv2.imwrite(rot_path, rotated)

        h1 = solide_hash_robust(base)
        h2 = solide_hash_robust(rot_path)
        sim = measure_similarity(h1, h2)
        assert sim > 0.5


# ── measure_similarity ───────────────────────────────────────────────────


class TestMeasureSimilarity:
    """Tests for the similarity comparison function."""

    def test_identical_hashes_return_one(self, tmp_path):
        """Comparing a hash with itself should give 1.0."""
        path = _write_test_image(str(tmp_path / "test.png"), pattern="circle")
        h = solide_hash_robust(path)
        assert measure_similarity(h, h) == 1.0

    def test_different_hashes_below_one(self, tmp_path):
        """Distinct images should have similarity < 1.0."""
        a = _write_test_image(str(tmp_path / "a.png"), pattern="gradient")
        b = _write_test_image(str(tmp_path / "b.png"), pattern="circle")
        ha = solide_hash_robust(a)
        hb = solide_hash_robust(b)
        sim = measure_similarity(ha, hb)
        assert 0.0 <= sim < 1.0

    def test_empty_hash_returns_zero(self):
        """Empty or None hashes should return 0.0."""
        assert measure_similarity("", "abc123") == 0.0
        assert measure_similarity(None, "abc123") == 0.0

    def test_error_hash_returns_zero(self):
        """Hashes starting with 'Error' should return 0.0."""
        assert measure_similarity("Error: decode failed", "abc123") == 0.0

    def test_mismatched_sizes_returns_zero(self, tmp_path):
        """Hashes of different sizes should return 0.0."""
        path = _write_test_image(str(tmp_path / "test.png"))
        h16 = solide_hash_robust(path, size=16)
        h48 = solide_hash_robust(path, size=48)
        assert measure_similarity(h16, h48) == 0.0


# ── _hex_to_bool_grid ────────────────────────────────────────────────────


class TestHexToBoolGrid:
    """Tests for the hex → boolean grid conversion."""

    def test_round_trip_preserves_data(self, tmp_path):
        """Hash → grid → verify shape is perfect square."""
        path = _write_test_image(str(tmp_path / "test.png"))
        h = solide_hash_robust(path, size=16)
        grid = _hex_to_bool_grid(h)
        # 16x16 hash → 256 bits → 16x16 grid
        assert grid.shape == (16, 16)
        assert grid.dtype == bool

    def test_invalid_hex_raises(self):
        """Non-square bit count should raise ValueError."""
        # 3 bytes = 24 bits, sqrt(24) is not integer
        with pytest.raises(ValueError, match="not a perfect square"):
            _hex_to_bool_grid("abcdef")


# ── Edge Cases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases: RGBA, very small images, colour images."""

    def test_rgb_image(self, tmp_path):
        """3-channel colour image should hash without error."""
        path = _write_test_image(str(tmp_path / "rgb.png"), channels=3, pattern="gradient")
        h = solide_hash_robust(path)
        assert isinstance(h, str) and len(h) > 0

    def test_rgba_image(self, tmp_path):
        """4-channel RGBA image should hash without error."""
        path = _write_test_image(str(tmp_path / "rgba.png"), channels=4, pattern="gradient")
        h = solide_hash_robust(path)
        assert isinstance(h, str) and len(h) > 0

    def test_very_small_image(self, tmp_path):
        """Tiny images (e.g. 8x8) should still produce a valid hash."""
        path = _write_test_image(str(tmp_path / "tiny.png"), width=8, height=8, pattern="random")
        h = solide_hash_robust(path)
        assert isinstance(h, str) and len(h) > 0
