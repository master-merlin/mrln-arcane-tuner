"""Unit tests for app.core.image_adjustments module."""

import numpy as np
import pytest
from PIL import Image

from app.core.image_adjustments import (
    CurvePoint,
    apply_all,
    apply_contrast,
    apply_curves,
    apply_hue_saturation,
    apply_lut_cube,
    apply_sharpening,
    compute_histogram,
    export_curves_as_cube,
    parse_cube_file,
)
from app.core.image_processing.curves import _build_lut_from_points


def _make_image(w: int = 64, h: int = 64, color: tuple = (128, 100, 80)) -> Image.Image:
    """Create a solid-color test image."""
    return Image.new("RGB", (w, h), color)


def _gradient_image(w: int = 256, h: int = 64) -> Image.Image:
    """Create a horizontal gradient image covering 0-255 in R channel."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        arr[:, x, :] = (x, x // 2, 255 - x)
    return Image.fromarray(arr, "RGB")


# ── Curves ───────────────────────────────────────────────────────────────


class TestCurves:
    def test_identity_curves_no_change(self):
        """Identity curve (diagonal) should not alter the image."""
        img = _gradient_image()
        identity = [CurvePoint(x=0, y=0), CurvePoint(x=255, y=255)]
        result = apply_curves(img, master=identity, r=identity, g=identity, b=identity)
        np.testing.assert_array_equal(np.array(img), np.array(result))

    def test_invert_curve(self):
        """Curve from (0,255) to (255,0) should invert."""
        img = _make_image(color=(50, 100, 200))
        invert = [CurvePoint(x=0, y=255), CurvePoint(x=255, y=0)]
        result = apply_curves(img, master=invert)
        arr = np.array(result)
        # All pixels inverted: 50→205, 100→155, 200→55 (PCHIP may differ slightly)
        assert arr[0, 0, 0] > 180  # Was 50, should be near 205
        assert arr[0, 0, 2] < 80   # Was 200, should be near 55

    def test_build_lut_length(self):
        """LUT should always produce exactly 256 entries."""
        pts = [CurvePoint(x=0, y=0), CurvePoint(x=128, y=200), CurvePoint(x=255, y=255)]
        lut = _build_lut_from_points(pts)
        assert lut.shape == (256,)
        assert lut.dtype == np.uint8


# ── CUBE LUT ─────────────────────────────────────────────────────────────


class TestCubeLUT:
    IDENTITY_CUBE_SIZE = 2

    @staticmethod
    def _identity_cube_str(size: int = 2) -> str:
        """Build a minimal identity .cube file string."""
        lines = [
            'TITLE "Identity"',
            f"LUT_3D_SIZE {size}",
            "DOMAIN_MIN 0.0 0.0 0.0",
            "DOMAIN_MAX 1.0 1.0 1.0",
        ]
        for b in range(size):
            for g in range(size):
                for r in range(size):
                    rv = r / (size - 1)
                    gv = g / (size - 1)
                    bv = b / (size - 1)
                    lines.append(f"{rv:.6f} {gv:.6f} {bv:.6f}")
        return "\n".join(lines)

    def test_parse_cube_file(self):
        cube_str = self._identity_cube_str(2)
        lut = parse_cube_file(cube_str)
        assert lut.title == "Identity"
        assert lut.size == 2
        assert lut.table.shape == (2, 2, 2, 3)

    def test_apply_lut_cube_identity(self):
        """Identity 3D LUT should not alter the image (within rounding)."""
        img = _gradient_image()
        lut = parse_cube_file(self._identity_cube_str(17))
        result = apply_lut_cube(img, lut)
        orig = np.array(img, dtype=np.int16)
        res = np.array(result, dtype=np.int16)
        # Allow ±2 rounding tolerance from trilinear interpolation
        assert np.abs(orig - res).max() <= 2

    def test_export_curves_as_cube_roundtrip(self):
        """Export identity curves → parse back → verify it's a valid cube."""
        identity = [CurvePoint(x=0, y=0), CurvePoint(x=255, y=255)]
        cube_str = export_curves_as_cube(master=identity, size=5)
        lut = parse_cube_file(cube_str)
        assert lut.size == 5
        assert lut.table.shape == (5, 5, 5, 3)

    def test_parse_invalid_1d_raises(self):
        with pytest.raises(ValueError, match="1D LUT"):
            parse_cube_file("LUT_1D_SIZE 256\n")

    def test_parse_missing_size_raises(self):
        with pytest.raises(ValueError, match="LUT_3D_SIZE not found"):
            parse_cube_file("0.0 0.0 0.0\n1.0 1.0 1.0\n")


# ── Hue/Saturation ──────────────────────────────────────────────────────


class TestHueSaturation:
    def test_no_change(self):
        """Default params should not alter image."""
        img = _make_image()
        result = apply_hue_saturation(img, 0.0, 1.0)
        assert result is img  # Should return same object (early exit)

    def test_hue_shift_wraparound(self):
        """+180 and -180 should produce the same result."""
        img = _make_image(color=(200, 100, 50))
        r1 = apply_hue_saturation(img, 180.0, 1.0)
        r2 = apply_hue_saturation(img, -180.0, 1.0)
        arr1 = np.array(r1, dtype=np.int16)
        arr2 = np.array(r2, dtype=np.int16)
        assert np.abs(arr1 - arr2).max() <= 1  # Allow ±1 rounding

    def test_desaturate(self):
        """Saturation=0 should produce a grayscale image."""
        img = _make_image(color=(255, 0, 0))
        result = apply_hue_saturation(img, 0.0, 0.0)
        arr = np.array(result)
        # All channels should be equal (grayscale)
        assert np.allclose(arr[:, :, 0], arr[:, :, 1], atol=1)
        assert np.allclose(arr[:, :, 1], arr[:, :, 2], atol=1)


# ── Contrast ─────────────────────────────────────────────────────────────


class TestContrast:
    def test_no_change(self):
        img = _make_image()
        result = apply_contrast(img, 1.0)
        assert result is img

    def test_zero_contrast_flat_gray(self):
        """Factor=0 should yield a flat mid-gray image."""
        img = _gradient_image()
        result = apply_contrast(img, 0.0)
        arr = np.array(result)
        # All pixels should be the same value (mean gray)
        unique_per_channel = [len(np.unique(arr[:, :, c])) for c in range(3)]
        assert all(u == 1 for u in unique_per_channel)


# ── Sharpening ───────────────────────────────────────────────────────────


class TestSharpening:
    @pytest.mark.parametrize("method", ["unsharp_mask", "kernel", "high_pass"])
    def test_sharpening_methods_valid(self, method: str):
        """Each sharpening method should run without error."""
        img = _gradient_image()
        result = apply_sharpening(img, method=method)
        assert result.size == img.size

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            apply_sharpening(_make_image(), method="nonexistent")


# ── Histogram ────────────────────────────────────────────────────────────


class TestHistogram:
    def test_compute_histogram_shape(self):
        """Should return 4 arrays of length 256."""
        img = _gradient_image()
        hist = compute_histogram(img)
        assert set(hist.keys()) == {"r", "g", "b", "luminance"}
        for key in hist:
            assert len(hist[key]) == 256

    def test_solid_color_histogram(self):
        """Solid color image should have one spike per channel."""
        img = _make_image(64, 64, (100, 150, 200))
        hist = compute_histogram(img)
        assert hist["r"][100] == 64 * 64
        assert hist["g"][150] == 64 * 64
        assert hist["b"][200] == 64 * 64


# ── Pipeline ─────────────────────────────────────────────────────────────


class TestApplyAll:
    def test_empty_adjustments(self):
        """Empty dict should not alter image."""
        img = _make_image()
        result = apply_all(img, {})
        np.testing.assert_array_equal(np.array(img.convert("RGB")), np.array(result))

    def test_pipeline_order(self):
        """Applying multiple adjustments should produce a different image."""
        img = _gradient_image()
        result = apply_all(img, {
            "curves": {
                "master": [{"x": 0, "y": 20}, {"x": 255, "y": 235}],
                "r": [], "g": [], "b": [],
            },
            "contrast": 1.5,
            "hue_shift": 30.0,
            "saturation": 0.8,
            "sharpening": {"method": "unsharp_mask", "params": {"radius": 1, "percent": 50, "threshold": 0}},
        })
        orig = np.array(img.convert("RGB"))
        res = np.array(result)
        assert not np.array_equal(orig, res)
