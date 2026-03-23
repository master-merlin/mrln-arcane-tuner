"""Adjustment pipeline orchestrator — block-based, composable, user-reorderable.

Supports both the legacy `apply_all()` interface (fixed-order dict-based) and
the new `execute_pipeline()` interface (ordered list of typed blocks).

Each block has a type, enabled flag, and params dict. The executor dispatches
to the appropriate handler in sequence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

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


# ---------------------------------------------------------------------------
# Block type constants
# ---------------------------------------------------------------------------

BlockType = Literal[
    "denoise",
    "face_restore",
    "deartifact",
    "dehaze",
    "white_balance",
    "curves",
    "cube_lut",
    "hsl_selective",
    "hue_saturation",
    "contrast",
    "vignette",
    "lens_correction",
    "sharpening",
    "color_match",
    "upscale",
]


@dataclass
class PipelineBlock:
    """A single operation block in the editing pipeline."""

    type: BlockType
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Block handlers — each takes (img, params) → img
# ---------------------------------------------------------------------------


def _handle_white_balance(img: Image.Image, params: dict) -> Image.Image:
    temperature = params.get("temperature", 6500.0)
    tint = params.get("tint", 0.0)
    if temperature != 6500.0 or tint != 0.0:
        return apply_white_balance(img, temperature, tint)
    return img


def _handle_vignette(img: Image.Image, params: dict) -> Image.Image:
    amount = params.get("amount", 0.0)
    if amount != 0.0:
        return apply_vignette(
            img,
            amount,
            midpoint=params.get("midpoint", 0.5),
            feather=params.get("feather", 0.5),
        )
    return img


def _handle_lens_correction(img: Image.Image, params: dict) -> Image.Image:
    barrel = params.get("barrel", 0.0)
    vk = params.get("vertical_keystone", 0.0)
    hk = params.get("horizontal_keystone", 0.0)
    if barrel != 0.0 or vk != 0.0 or hk != 0.0:
        return apply_lens_correction(img, barrel, vk, hk)
    return img


def _handle_curves(img: Image.Image, params: dict) -> Image.Image:
    def _to_points(pts: list[dict] | None) -> list[CurvePoint] | None:
        if not pts:
            return None
        return [CurvePoint(x=p["x"], y=p["y"]) for p in pts]

    return apply_curves(
        img,
        master=_to_points(params.get("master")),
        r=_to_points(params.get("r")),
        g=_to_points(params.get("g")),
        b=_to_points(params.get("b")),
    )


def _handle_cube_lut(img: Image.Image, params: dict) -> Image.Image:
    cube_lut_str: str | None = params.get("cube_lut")
    if cube_lut_str:
        lut_data = parse_cube_file(cube_lut_str)
        strength = params.get("cube_lut_strength", 1.0)
        return apply_lut_cube(img, lut_data, strength=strength)
    return img


def _handle_hsl_selective(img: Image.Image, params: dict) -> Image.Image:
    hsl = params.get("hsl_config")
    if hsl:
        return apply_hsl_selective(img, hsl)
    return img


def _handle_hue_saturation(img: Image.Image, params: dict) -> Image.Image:
    hue_shift = params.get("hue_shift", 0.0)
    saturation = params.get("saturation", 1.0)
    if hue_shift != 0.0 or saturation != 1.0:
        return apply_hue_saturation(img, hue_shift, saturation)
    return img


def _handle_contrast(img: Image.Image, params: dict) -> Image.Image:
    contrast = params.get("contrast", 1.0)
    if contrast != 1.0:
        return apply_contrast(img, contrast)
    return img


def _handle_sharpening(img: Image.Image, params: dict) -> Image.Image:
    method = params.get("method", "none")
    if method != "none":
        return apply_sharpening(img, method=method, params=params.get("params"))
    return img


def _handle_color_match(img: Image.Image, params: dict) -> Image.Image:
    ref_path = params.get("reference_path")
    if ref_path and os.path.exists(ref_path):
        with Image.open(ref_path) as ref_img:
            return apply_color_match(
                img,
                ref_img.convert("RGB"),
                strength=params.get("strength", 1.0),
                method=params.get("method", "cdf"),
            )
    return img


def _handle_restoration(img: Image.Image, params: dict) -> Image.Image:
    """Handle any GPU restoration block (denoise, face_restore, etc.)."""
    from app.core.image_processing.restoration import apply_restoration

    model_path = params.get("model_path")
    if not model_path:
        return img
    return apply_restoration(
        img,
        model_path=model_path,
        strength=params.get("strength", 1.0),
        tile_size=params.get("tile_size", 512),
        tile_pad=params.get("tile_pad", 32),
    )


def _handle_upscale(img: Image.Image, params: dict) -> Image.Image:
    """Handle upscale block — uses tiled inference with rescale support."""
    model_path = params.get("model_path")
    if not model_path:
        return img

    from app.core.image_processing.tiled_inference import (
        cleanup_vram,
        image_to_tensor,
        load_spandrel_model,
        run_tiled_inference,
        tensor_to_image,
    )

    model, scale, device = load_spandrel_model(model_path)
    img_tensor = image_to_tensor(img, device)

    output_tensor = run_tiled_inference(
        model,
        img_tensor,
        tile_size=params.get("tile_size", 512),
        tile_pad=params.get("tile_pad", 32),
        scale=scale,
    )
    result = tensor_to_image(output_tensor)
    cleanup_vram(model, img_tensor, output_tensor)

    # Post-process rescale if target differs from model native
    target = params.get("target_scale", 0)
    if target > 0 and abs(target - scale) > 0.01:
        w_orig, h_orig = img.size
        final_w = round(w_orig * target)
        final_h = round(h_orig * target)
        resample_map = {
            "lanczos": Image.LANCZOS,
            "bicubic": Image.BICUBIC,
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
        }
        resample = resample_map.get(
            params.get("resize_method", "lanczos"), Image.LANCZOS
        )
        result = result.resize((final_w, final_h), resample)

    return result


# ---------------------------------------------------------------------------
# Block handler registry
# ---------------------------------------------------------------------------

BLOCK_HANDLERS: dict[str, callable] = {
    "white_balance": _handle_white_balance,
    "vignette": _handle_vignette,
    "lens_correction": _handle_lens_correction,
    "curves": _handle_curves,
    "cube_lut": _handle_cube_lut,
    "hsl_selective": _handle_hsl_selective,
    "hue_saturation": _handle_hue_saturation,
    "contrast": _handle_contrast,
    "sharpening": _handle_sharpening,
    "color_match": _handle_color_match,
    "denoise": _handle_restoration,
    "face_restore": _handle_restoration,
    "deartifact": _handle_restoration,
    "dehaze": _handle_restoration,
    "upscale": _handle_upscale,
}


# ---------------------------------------------------------------------------
# Pipeline executors
# ---------------------------------------------------------------------------


def execute_pipeline(img: Image.Image, blocks: list[PipelineBlock]) -> Image.Image:
    """Execute an ordered list of pipeline blocks on an image.

    Blocks are applied in the order they appear. Disabled blocks are skipped.
    """
    result = img.convert("RGB")
    for block in blocks:
        if not block.enabled:
            continue
        handler = BLOCK_HANDLERS.get(block.type)
        if handler:
            result = handler(result, block.params)
    return result


def apply_all(img: Image.Image, adjustments: dict) -> Image.Image:
    """Legacy interface — applies all adjustments in canonical order.

    Backward-compatible wrapper that converts the flat adjustments dict
    into ordered pipeline blocks and executes them.
    """
    blocks: list[PipelineBlock] = []

    # 0. Color Match
    cm = adjustments.get("color_match")
    if cm:
        blocks.append(PipelineBlock(type="color_match", params=cm))

    # 1. White Balance
    wb = adjustments.get("white_balance")
    if wb:
        blocks.append(PipelineBlock(type="white_balance", params=wb))

    # 2. Vignette
    vig = adjustments.get("vignette")
    if vig:
        blocks.append(PipelineBlock(type="vignette", params=vig))

    # 3. Lens Correction
    lens = adjustments.get("lens_correction")
    if lens:
        blocks.append(PipelineBlock(type="lens_correction", params=lens))

    # 4. Curves
    curves = adjustments.get("curves")
    if curves:
        blocks.append(PipelineBlock(type="curves", params=curves))

    # 5. CUBE LUT
    cube_lut_str = adjustments.get("cube_lut")
    if cube_lut_str:
        blocks.append(
            PipelineBlock(
                type="cube_lut",
                params={
                    "cube_lut": cube_lut_str,
                    "cube_lut_strength": adjustments.get("cube_lut_strength", 1.0),
                },
            )
        )

    # 6. HSL Selective
    hsl = adjustments.get("hsl_selective")
    if hsl:
        blocks.append(
            PipelineBlock(type="hsl_selective", params={"hsl_config": hsl})
        )

    # 7. Hue / Saturation
    hue_shift = adjustments.get("hue_shift", 0.0)
    saturation = adjustments.get("saturation", 1.0)
    if hue_shift != 0.0 or saturation != 1.0:
        blocks.append(
            PipelineBlock(
                type="hue_saturation",
                params={"hue_shift": hue_shift, "saturation": saturation},
            )
        )

    # 8. Contrast
    contrast = adjustments.get("contrast", 1.0)
    if contrast != 1.0:
        blocks.append(
            PipelineBlock(type="contrast", params={"contrast": contrast})
        )

    # 9. Sharpening
    sharpening = adjustments.get("sharpening")
    if sharpening:
        blocks.append(PipelineBlock(type="sharpening", params=sharpening))

    return execute_pipeline(img, blocks)
