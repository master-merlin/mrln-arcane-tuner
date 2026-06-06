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


class CropBatchItem(BaseModel):
    """One image queued for batch crop, with its own analysis-derived target."""
    path: str
    target_width: int
    target_height: int


class CropBatchRequest(BaseModel):
    """Request body for a batch crop. `origin` (9-position anchor) is shared
    across all items; each item carries its own target dimensions."""
    items: list[CropBatchItem]
    origin: str = "center"


# ── Response models ──────────────────────────────────────────────────────


class CropResponse(BaseModel):
    """Ack for a single-image crop."""
    status: str = "cropped"
    file: str


class CalcCropTargetResponse(BaseModel):
    """Computed crop target dimensions for a given image + aspect ratio."""
    target_width: int
    target_height: int
