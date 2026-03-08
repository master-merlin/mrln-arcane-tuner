"""HSL selective color adjustments — per-hue-range HSL shifts."""

from __future__ import annotations

import numpy as np
from PIL import Image


# 8 hue ranges with center hue and width (degrees)
HSL_RANGES: dict[str, tuple[float, float]] = {
    "reds":     (0.0,   30.0),
    "oranges":  (30.0,  30.0),
    "yellows":  (60.0,  30.0),
    "greens":   (120.0, 40.0),
    "cyans":    (180.0, 30.0),
    "blues":    (240.0, 40.0),
    "purples":  (285.0, 30.0),
    "magentas": (330.0, 30.0),
}


def apply_hsl_selective(
    img: Image.Image,
    adjustments: dict[str, dict[str, float]],
) -> Image.Image:
    """Apply per-hue-range HSL adjustments.

    Args:
        adjustments: Dict mapping range name to {hue_shift, saturation, luminance}.
            Example: {"reds": {"hue_shift": 10, "saturation": 20, "luminance": -10}}
            hue_shift: degrees (-30..+30), saturation/luminance: percent (-100..+100)
    """
    if not adjustments:
        return img

    # Check if all adjustments are zero
    has_changes = False
    for adj in adjustments.values():
        if any(abs(v) > 0.001 for v in adj.values()):
            has_changes = True
            break
    if not has_changes:
        return img

    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0

    # Convert to HLS (using colorsys convention: H=0-360, L=0-1, S=0-1)
    r_ch, g_ch, b_ch = arr[..., 0], arr[..., 1], arr[..., 2]
    cmax = np.maximum(np.maximum(r_ch, g_ch), b_ch)
    cmin = np.minimum(np.minimum(r_ch, g_ch), b_ch)
    delta = cmax - cmin
    lum = (cmax + cmin) / 2.0

    # Hue (0-360)
    hue = np.zeros_like(delta)
    mask_r = (cmax == r_ch) & (delta > 0)
    mask_g = (cmax == g_ch) & (delta > 0)
    mask_b = (cmax == b_ch) & (delta > 0)
    hue[mask_r] = 60.0 * (((g_ch[mask_r] - b_ch[mask_r]) / delta[mask_r]) % 6)
    hue[mask_g] = 60.0 * (((b_ch[mask_g] - r_ch[mask_g]) / delta[mask_g]) + 2)
    hue[mask_b] = 60.0 * (((r_ch[mask_b] - g_ch[mask_b]) / delta[mask_b]) + 4)

    # Saturation
    sat = np.where(delta > 0, delta / (1.0 - np.abs(2.0 * lum - 1.0) + 1e-10), 0.0)

    # Apply per-range adjustments with smooth falloff
    for range_name, adj in adjustments.items():
        if range_name not in HSL_RANGES:
            continue
        center, width = HSL_RANGES[range_name]
        h_shift = adj.get("hue_shift", 0.0)
        s_adj = adj.get("saturation", 0.0) / 100.0  # -1 to +1
        l_adj = adj.get("luminance", 0.0) / 100.0   # -1 to +1

        if abs(h_shift) < 0.001 and abs(s_adj) < 0.001 and abs(l_adj) < 0.001:
            continue

        # Compute angular distance with wrapping
        d = np.abs(hue - center)
        d = np.minimum(d, 360.0 - d)
        # Smooth falloff: 1.0 inside range, cosine taper to 0 at 1.5× width
        falloff = np.clip(1.0 - (d - width) / (width * 0.5 + 1e-6), 0.0, 1.0)
        # Smooth with cosine
        weight = 0.5 * (1.0 + np.cos(np.pi * (1.0 - falloff)))
        weight *= (sat > 0.01).astype(np.float32)  # Skip achromatic pixels

        hue += h_shift * weight
        sat += s_adj * weight
        lum += l_adj * 0.5 * weight  # Scale luminance adjustment

    hue = hue % 360.0
    sat = np.clip(sat, 0.0, 1.0)
    lum = np.clip(lum, 0.0, 1.0)

    # HLS → RGB
    c = (1.0 - np.abs(2.0 * lum - 1.0)) * sat
    x = c * (1.0 - np.abs((hue / 60.0) % 2 - 1.0))
    m = lum - c / 2.0

    out = np.zeros_like(arr)
    h_seg = (hue / 60.0).astype(np.int32) % 6

    # Build output channels
    for i, (c1, c2, c3) in enumerate([
        (c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)
    ]):
        mask = h_seg == i
        for ch_idx, val in enumerate([c1, c2, c3]):
            if isinstance(val, (int, float)):
                out[..., ch_idx][mask] = val + m[mask]
            else:
                out[..., ch_idx][mask] = val[mask] + m[mask]

    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB")
