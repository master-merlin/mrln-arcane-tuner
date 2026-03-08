"""Adjustment pipeline orchestrator — applies all adjustments in canonical order.

Order: Color Match → White Balance → Vignette → Lens Correction →
       Curves → CUBE LUT → HSL Selective → Hue/Saturation →
       Contrast → Sharpening.
"""

from __future__ import annotations

import os

from PIL import Image

from app.core.image_processing.curves import (
    CurvePoint,
    apply_curves,
    parse_cube_file,
    apply_lut_cube,
)
from app.core.image_processing.color import (
    apply_hue_saturation,
    apply_contrast,
    apply_white_balance,
)
from app.core.image_processing.spatial import (
    apply_sharpening,
    apply_vignette,
    apply_lens_correction,
)
from app.core.image_processing.hsl import apply_hsl_selective
from app.core.image_processing.color_match import apply_color_match


def apply_all(img: Image.Image, adjustments: dict) -> Image.Image:
    """Apply all adjustments in the canonical order.

    Order: Color Match → White Balance → Vignette → Lens Correction →
           Curves → CUBE LUT → HSL Selective → Hue/Saturation →
           Contrast → Sharpening.
    """
    result = img.convert("RGB")

    # 0. Color Match (applies reference color distribution first)
    cm = adjustments.get("color_match")
    if cm:
        ref_path = cm.get("reference_path")
        if ref_path and os.path.exists(ref_path):
            with Image.open(ref_path) as ref_img:
                result = apply_color_match(
                    result,
                    ref_img.convert("RGB"),
                    strength=cm.get("strength", 1.0),
                    method=cm.get("method", "cdf"),
                )

    # 1. White Balance
    wb = adjustments.get("white_balance")
    if wb:
        temperature = wb.get("temperature", 6500.0)
        tint = wb.get("tint", 0.0)
        if temperature != 6500.0 or tint != 0.0:
            result = apply_white_balance(result, temperature, tint)

    # 2. Vignette
    vig = adjustments.get("vignette")
    if vig:
        amount = vig.get("amount", 0.0)
        if amount != 0.0:
            result = apply_vignette(
                result, amount,
                midpoint=vig.get("midpoint", 0.5),
                feather=vig.get("feather", 0.5),
            )

    # 3. Lens Correction
    lens = adjustments.get("lens_correction")
    if lens:
        barrel = lens.get("barrel", 0.0)
        vk = lens.get("vertical_keystone", 0.0)
        hk = lens.get("horizontal_keystone", 0.0)
        if barrel != 0.0 or vk != 0.0 or hk != 0.0:
            result = apply_lens_correction(result, barrel, vk, hk)

    # 4. Color grading: CUBE LUT and curves can coexist (curves first, then LUT)
    curves = adjustments.get("curves")
    if curves:
        def _to_points(pts: list[dict] | None) -> list[CurvePoint] | None:
            if not pts:
                return None
            return [CurvePoint(x=p["x"], y=p["y"]) for p in pts]

        result = apply_curves(
            result,
            master=_to_points(curves.get("master")),
            r=_to_points(curves.get("r")),
            g=_to_points(curves.get("g")),
            b=_to_points(curves.get("b")),
        )

    cube_lut_str: str | None = adjustments.get("cube_lut")
    if cube_lut_str:
        lut_data = parse_cube_file(cube_lut_str)
        lut_strength = adjustments.get("cube_lut_strength", 1.0)
        result = apply_lut_cube(result, lut_data, strength=lut_strength)

    # 5. HSL Selective Color
    hsl = adjustments.get("hsl_selective")
    if hsl:
        result = apply_hsl_selective(result, hsl)

    # 6. Hue / Saturation
    hue_shift = adjustments.get("hue_shift", 0.0)
    saturation = adjustments.get("saturation", 1.0)
    if hue_shift != 0.0 or saturation != 1.0:
        result = apply_hue_saturation(result, hue_shift, saturation)

    # 7. Contrast
    contrast = adjustments.get("contrast", 1.0)
    if contrast != 1.0:
        result = apply_contrast(result, contrast)

    # 8. Sharpening
    sharpening = adjustments.get("sharpening")
    if sharpening:
        result = apply_sharpening(
            result,
            method=sharpening.get("method", "unsharp_mask"),
            params=sharpening.get("params"),
        )

    return result
