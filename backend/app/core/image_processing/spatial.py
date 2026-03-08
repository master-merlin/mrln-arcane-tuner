"""Spatial transformations — sharpening, vignette, lens correction."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


# ── Sharpening ───────────────────────────────────────────────────────────


def apply_sharpening(
    img: Image.Image,
    method: str = "unsharp_mask",
    params: dict | None = None,
) -> Image.Image:
    """Apply sharpening with selectable method."""
    params = params or {}
    rgb = img.convert("RGB")

    if method == "unsharp_mask":
        radius = params.get("radius", 2.0)
        percent = params.get("percent", 150)
        threshold = params.get("threshold", 3)
        return rgb.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

    if method == "kernel":
        strength = params.get("strength", 1.0)
        sharpened = rgb.filter(ImageFilter.SHARPEN)
        if strength == 1.0:
            return sharpened
        # Blend: original * (1 - strength) + sharpened * strength
        arr_o = np.array(rgb, dtype=np.float32)
        arr_s = np.array(sharpened, dtype=np.float32)
        blended = arr_o * (1 - strength) + arr_s * strength
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), "RGB")

    if method == "high_pass":
        radius = params.get("radius", 3.0)
        strength = params.get("strength", 0.5)
        arr = np.array(rgb, dtype=np.float32)
        blurred = np.array(rgb.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)
        high_pass = arr - blurred + 128.0
        # Overlay blend: combine high-pass details back onto original
        result = arr + (high_pass - 128.0) * strength
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), "RGB")

    raise ValueError(f"Unknown sharpening method: {method}")


# ── Vignette ─────────────────────────────────────────────────────────────


def apply_vignette(
    img: Image.Image,
    amount: float = 0.0,
    midpoint: float = 0.5,
    feather: float = 0.5,
) -> Image.Image:
    """Apply or remove vignette (radial brightness correction).

    Args:
        amount: -1.0 (remove/brighten corners) to +1.0 (darken corners). 0 = no change.
        midpoint: Radial distance (0-1) where effect begins. 0.5 = halfway from center.
        feather: Transition smoothness (0-1). 0 = hard, 1 = very gradual.
    """
    if amount == 0.0:
        return img

    arr = np.array(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]

    # Build radial distance map (0 at center, 1 at corners)
    y_coords = np.linspace(-1, 1, h)[:, np.newaxis]
    x_coords = np.linspace(-1, 1, w)[np.newaxis, :]
    # Account for aspect ratio
    aspect = w / max(h, 1)
    radius = np.sqrt((x_coords / max(aspect, 1.0)) ** 2 + (y_coords * min(aspect, 1.0)) ** 2)
    # Normalize so max dist = 1
    radius = radius / (radius.max() + 1e-6)

    # Apply midpoint and feather to create the mask
    feather_val = max(feather, 0.01)  # avoid division by zero
    mask = np.clip((radius - midpoint) / feather_val, 0.0, 1.0)

    # Amount: positive darkens, negative brightens
    if amount > 0:
        # Darken: multiply by (1 - amount * mask)
        multiplier = 1.0 - amount * mask
    else:
        # Brighten corners: boost by (1 + |amount| * mask)
        multiplier = 1.0 + abs(amount) * mask

    arr *= multiplier[:, :, np.newaxis]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


# ── Lens Correction ──────────────────────────────────────────────────────


def _find_perspective_coeffs(
    source_coords: list[tuple[float, float]],
    target_coords: list[tuple[float, float]],
) -> tuple[float, ...]:
    """Compute 8 perspective transform coefficients for PIL.Image.transform()."""
    matrix = []
    for s, t in zip(source_coords, target_coords):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0] * t[0], -s[0] * t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1] * t[0], -s[1] * t[1]])
    A = np.array(matrix, dtype=np.float64)
    B = np.array([c for s in source_coords for c in s], dtype=np.float64)
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return tuple(res.tolist())


def apply_lens_correction(
    img: Image.Image,
    barrel: float = 0.0,
    vertical_keystone: float = 0.0,
    horizontal_keystone: float = 0.0,
) -> Image.Image:
    """Apply barrel/pincushion distortion and perspective keystone correction.

    Args:
        barrel: -1.0 (pincushion) to +1.0 (barrel correction). 0 = no change.
        vertical_keystone: Vertical perspective tilt in degrees (-45..+45).
        horizontal_keystone: Horizontal perspective tilt in degrees (-45..+45).
    """
    if barrel == 0.0 and vertical_keystone == 0.0 and horizontal_keystone == 0.0:
        return img

    from scipy.ndimage import map_coordinates

    arr = np.array(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    result = arr.copy()

    if barrel != 0.0:
        # Barrel/pincushion via radial distortion
        cy, cx = h / 2, w / 2
        # Create coordinate grid
        y_idx, x_idx = np.mgrid[0:h, 0:w].astype(np.float64)
        # Normalize to center
        xn = (x_idx - cx) / cx
        yn = (y_idx - cy) / cy
        r = np.sqrt(xn ** 2 + yn ** 2)
        # Distortion: r_new = r * (1 + k * r^2)
        k = barrel * 0.5  # scale to reasonable range
        r_new = r * (1 + k * r ** 2)
        # Map back
        factor = np.where(r > 0, r_new / (r + 1e-10), 1.0)
        src_x = cx + (x_idx - cx) * factor
        src_y = cy + (y_idx - cy) * factor

        # Remap each channel
        for ch in range(3):
            result[:, :, ch] = map_coordinates(
                arr[:, :, ch], [src_y, src_x], order=1, mode="reflect"
            ).astype(np.float32)

    # Keystone correction via PIL perspective transform
    if vertical_keystone != 0.0 or horizontal_keystone != 0.0:
        pil_img = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), "RGB")
        # Build perspective coefficients
        vk = np.tan(np.radians(vertical_keystone * 0.5))
        hk = np.tan(np.radians(horizontal_keystone * 0.5))

        # Four corner mapping (source -> destination)
        x0 = hk * w * 0.5
        y0 = vk * h * 0.5

        coeffs = _find_perspective_coeffs(
            [(x0, y0), (w - x0, -y0), (w + x0, h + y0), (-x0, h - y0)],
            [(0, 0), (w, 0), (w, h), (0, h)],
        )
        pil_img = pil_img.transform((w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)
        result = np.array(pil_img, dtype=np.float32)

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), "RGB")
