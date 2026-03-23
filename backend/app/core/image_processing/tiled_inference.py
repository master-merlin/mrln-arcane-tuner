"""Reusable tiled inference utility for GPU-based model inference.

Supports both 1:1 (denoise, restore) and scaled (upscale) models.
Handles VRAM-safe tiling, padding removal, and cleanup.
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch
from PIL import Image

import numpy as np


def load_spandrel_model(model_path: str | Path) -> tuple:
    """Load a model via Spandrel and return (model, scale, device).

    Returns:
        Tuple of (model_on_device, scale_factor, device).
    """
    from spandrel import ModelLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelLoader().load_from_file(str(model_path))
    model = model.to(device).eval()

    # Detect model scale — spandrel stores it on the descriptor
    scale = getattr(model, "scale", None)
    if scale is None:
        with torch.inference_mode():
            test_out = model(torch.zeros(1, 3, 8, 8, device=device))
            scale = test_out.shape[2] // 8
        if scale < 1:
            scale = 1

    return model, scale, device


def image_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """Convert a PIL Image to a [1, 3, H, W] float32 tensor on device."""
    img_rgb = img.convert("RGB")
    img_np = np.array(img_rgb).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a [1, 3, H, W] tensor to a PIL Image."""
    result_np = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return Image.fromarray((result_np * 255).astype(np.uint8))


def run_tiled_inference(
    model: object,
    img_tensor: torch.Tensor,
    tile_size: int = 512,
    tile_pad: int = 32,
    scale: int = 1,
) -> torch.Tensor:
    """Run model inference on an image tensor using VRAM-safe tiling.

    Args:
        model: A Spandrel-loaded model (callable with __call__).
        img_tensor: Input tensor of shape [1, 3, H, W].
        tile_size: Size of each tile in pixels.
        tile_pad: Overlap padding for seam prevention.
        scale: Output scale factor (1 for denoise, 2/4/8 for upscale).

    Returns:
        Output tensor of shape [1, 3, H*scale, W*scale].
    """
    h, w = img_tensor.shape[2], img_tensor.shape[3]
    out_h, out_w = h * scale, w * scale
    output = torch.zeros(
        1, 3, out_h, out_w, device=img_tensor.device, dtype=img_tensor.dtype
    )

    with torch.inference_mode():
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                # Input tile with padding
                y1 = max(0, y - tile_pad)
                x1 = max(0, x - tile_pad)
                y2 = min(h, y + tile_size + tile_pad)
                x2 = min(w, x + tile_size + tile_pad)

                tile_in = img_tensor[:, :, y1:y2, x1:x2]
                tile_out = model(tile_in)

                # Calculate output region (remove padding)
                oy1 = (y - y1) * scale
                ox1 = (x - x1) * scale
                oy2 = oy1 + min(tile_size, h - y) * scale
                ox2 = ox1 + min(tile_size, w - x) * scale

                out_y1 = y * scale
                out_x1 = x * scale
                out_y2 = out_y1 + min(tile_size, h - y) * scale
                out_x2 = out_x1 + min(tile_size, w - x) * scale

                output[:, :, out_y1:out_y2, out_x1:out_x2] = tile_out[
                    :, :, oy1:oy2, ox1:ox2
                ]

    return output


def cleanup_vram(*tensors_and_models: object) -> None:
    """Delete GPU objects and reclaim VRAM."""
    for obj in tensors_and_models:
        del obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
