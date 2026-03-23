"""Overlay and pipeline-related Pydantic schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class PipelineBlockSchema(BaseModel):
    """A single operation block in the editing pipeline."""

    type: Literal[
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
    enabled: bool = True
    params: dict[str, Any] = {}


class RenderPipelineRequest(BaseModel):
    """Request body for executing the non-destructive pipeline."""

    image_path: str
    blocks: list[PipelineBlockSchema]
    tile_size: int = 512
    tile_pad: int = 32
    replace_recipe: bool = False


class OverlayCommitRequest(BaseModel):
    """Request body for committing (flattening) an overlay into the original."""

    image_path: str


class RestoreModelListRequest(BaseModel):
    """Request body for listing restoration models."""

    folder: str = ""


class ModelDownloadRequest(BaseModel):
    """Request body for downloading a model from the registry."""

    category: Literal["restore", "upscale"]
    filename: str
    target_folder: str = ""
