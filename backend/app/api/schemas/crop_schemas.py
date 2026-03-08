"""Crop-related schemas."""

from __future__ import annotations

from pydantic import BaseModel


class CropRequest(BaseModel):
    """Request body for image cropping."""
    path: str
    target_width: int
    target_height: int
    origin: str = "center"
    crop_x: int | None = None       # explicit left offset (px) for freeform crop
    crop_y: int | None = None       # explicit top offset (px) for freeform crop


class CalcCropTargetRequest(BaseModel):
    """Request body for crop target calculation."""
    width: int
    height: int
    aspect_ratio: float
