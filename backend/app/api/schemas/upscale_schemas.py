"""Upscale-related schemas."""

from __future__ import annotations

from pydantic import BaseModel


class UpscaleListRequest(BaseModel):
    """Request body for listing upscale models."""
    folder: str = ""


class UpscaleApplyRequest(BaseModel):
    """Request body for applying neural upscaling to an image."""
    model_path: str
    image_path: str
    tile_size: int = 512
    tile_pad: int = 32
    target_scale: float = 0     # 0 = use model native scale
    resize_method: str = "lanczos"  # lanczos, bicubic, bilinear, nearest
