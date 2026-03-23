"""GPU-based image restoration operations using Spandrel model inference.

Supports: denoise, face restore, JPEG deartifact, dehaze.
All operations use tiled inference for VRAM safety and support
strength-based alpha blending with the original.
"""

from __future__ import annotations

from PIL import Image

from app.core.image_processing.tiled_inference import (
    cleanup_vram,
    image_to_tensor,
    load_spandrel_model,
    run_tiled_inference,
    tensor_to_image,
)


def _blend_with_original(
    original: Image.Image, restored: Image.Image, strength: float
) -> Image.Image:
    """Alpha-blend restored result with original at given strength.

    strength=1.0 → fully restored, strength=0.0 → fully original.
    """
    if strength >= 1.0:
        return restored
    if strength <= 0.0:
        return original
    return Image.blend(original, restored, strength)


def apply_restoration(
    img: Image.Image,
    model_path: str,
    strength: float = 1.0,
    tile_size: int = 512,
    tile_pad: int = 32,
) -> Image.Image:
    """Apply a restoration model (denoise, face restore, deartifact, dehaze).

    Uses Spandrel to load the model (auto-detects architecture) and runs
    tiled inference at scale=1 (same dimensions as input). The result is
    blended with the original at the given strength.

    Args:
        img: Input PIL Image.
        model_path: Path to the model file (.pth, .safetensors, etc.).
        strength: Blending factor 0.0 (original) to 1.0 (fully restored).
        tile_size: Tile size for tiled inference.
        tile_pad: Overlap padding to prevent seams.

    Returns:
        Restored PIL Image at original resolution.
    """
    model, scale, device = load_spandrel_model(model_path)

    # For restoration models, scale should be 1. If Spandrel detects a
    # different scale (some architectures report scale even for 1:1 models),
    # we still use scale=1 for the output allocation and resize back.
    img_tensor = image_to_tensor(img, device)

    output_tensor = run_tiled_inference(
        model, img_tensor, tile_size=tile_size, tile_pad=tile_pad, scale=scale
    )

    restored = tensor_to_image(output_tensor)
    cleanup_vram(model, img_tensor, output_tensor)

    # If model produced a scaled output (unexpected for denoise), resize back
    if scale != 1:
        restored = restored.resize(img.size, Image.LANCZOS)

    return _blend_with_original(img, restored, strength)
