"""Color adjustments — hue/saturation, contrast, white balance."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance


# ── Hue / Saturation ────────────────────────────────────────────────────


def apply_hue_saturation(
    img: Image.Image,
    hue_shift: float = 0.0,
    saturation_factor: float = 1.0,
) -> Image.Image:
    """Adjust hue (degrees, -180..180) and saturation (0..3) in HSV space."""
    if hue_shift == 0.0 and saturation_factor == 1.0:
        return img

    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0

    # RGB → HSV (manual, faster than converting to/from PIL HSV mode)
    r_ch, g_ch, b_ch = arr[..., 0], arr[..., 1], arr[..., 2]
    cmax = np.maximum(np.maximum(r_ch, g_ch), b_ch)
    cmin = np.minimum(np.minimum(r_ch, g_ch), b_ch)
    delta = cmax - cmin

    # Hue
    hue = np.zeros_like(delta)
    mask_r = (cmax == r_ch) & (delta > 0)
    mask_g = (cmax == g_ch) & (delta > 0)
    mask_b = (cmax == b_ch) & (delta > 0)
    hue[mask_r] = 60.0 * (((g_ch[mask_r] - b_ch[mask_r]) / delta[mask_r]) % 6)
    hue[mask_g] = 60.0 * (((b_ch[mask_g] - r_ch[mask_g]) / delta[mask_g]) + 2)
    hue[mask_b] = 60.0 * (((r_ch[mask_b] - g_ch[mask_b]) / delta[mask_b]) + 4)

    # Saturation
    sat = np.where(cmax > 0, delta / cmax, 0.0)
    val = cmax

    # Apply adjustments
    hue = (hue + hue_shift) % 360.0
    sat = np.clip(sat * saturation_factor, 0.0, 1.0)

    # HSV → RGB
    c = val * sat
    x = c * (1 - np.abs((hue / 60.0) % 2 - 1))
    m = val - c

    out = np.zeros_like(arr)
    h_seg = (hue / 60.0).astype(np.int32) % 6

    for i, (c1, c2) in enumerate([(c, x), (x, c), (0, c), (0, x), (x, 0), (c, 0)]):
        mask = h_seg == i
        if isinstance(c1, (int, float)):
            out[..., 0][mask] = c1 + m[mask]
        else:
            out[..., 0][mask] = c1[mask] + m[mask]
        if isinstance(c2, (int, float)):
            out[..., 1][mask] = c2 + m[mask]
        else:
            out[..., 1][mask] = c2[mask] + m[mask]

    # Fill the third channel based on segment
    for i, c3 in enumerate([0, 0, x, c, c, x]):
        mask = h_seg == i
        if isinstance(c3, (int, float)):
            out[..., 2][mask] = c3 + m[mask]
        else:
            out[..., 2][mask] = c3[mask] + m[mask]

    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB")


# ── Contrast ─────────────────────────────────────────────────────────────


def apply_contrast(img: Image.Image, factor: float = 1.0) -> Image.Image:
    """Adjust contrast.  factor=1.0 → no change, 0.0 → solid gray."""
    if factor == 1.0:
        return img
    return ImageEnhance.Contrast(img.convert("RGB")).enhance(factor)


# ── White Balance ────────────────────────────────────────────────────────


def _kelvin_to_rgb(temperature: float) -> tuple[float, float, float]:
    """Convert color temperature (Kelvin) to RGB multipliers (0-1 scale).

    Approximation based on Tanner Helland's algorithm.
    """
    temp = max(1000, min(40000, temperature)) / 100.0

    # Red
    if temp <= 66:
        red = 1.0
    else:
        red = 329.698727446 * ((temp - 60) ** -0.1332047592) / 255.0

    # Green
    if temp <= 66:
        green = (99.4708025861 * np.log(temp) - 161.1195681661) / 255.0
    else:
        green = 288.1221695283 * ((temp - 60) ** -0.0755148492) / 255.0

    # Blue
    if temp >= 66:
        blue = 1.0
    elif temp <= 19:
        blue = 0.0
    else:
        blue = (138.5177312231 * np.log(temp - 10) - 305.0447927307) / 255.0

    return (
        float(np.clip(red, 0.0, 1.0)),
        float(np.clip(green, 0.0, 1.0)),
        float(np.clip(blue, 0.0, 1.0)),
    )


def apply_white_balance(
    img: Image.Image,
    temperature: float = 6500.0,
    tint: float = 0.0,
) -> Image.Image:
    """Adjust white balance via color temperature and tint.

    Args:
        temperature: Color temperature in Kelvin (2000-12000, default 6500 = daylight).
        tint: Green-magenta tint shift (-100..+100, default 0).
    """
    if temperature == 6500.0 and tint == 0.0:
        return img

    # Get RGB scale factors for both the target and neutral (6500K) temperatures,
    # then compute the correction ratio.
    target_rgb = _kelvin_to_rgb(temperature)
    neutral_rgb = _kelvin_to_rgb(6500.0)

    r_scale = neutral_rgb[0] / max(target_rgb[0], 0.001)
    g_scale = neutral_rgb[1] / max(target_rgb[1], 0.001)
    b_scale = neutral_rgb[2] / max(target_rgb[2], 0.001)

    # Tint: shift green-magenta axis
    tint_factor = tint / 100.0
    g_scale *= (1.0 + tint_factor * 0.3)
    r_scale *= (1.0 - tint_factor * 0.1)
    b_scale *= (1.0 - tint_factor * 0.1)

    arr = np.array(img.convert("RGB"), dtype=np.float32)
    arr[:, :, 0] *= r_scale
    arr[:, :, 1] *= g_scale
    arr[:, :, 2] *= b_scale

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
