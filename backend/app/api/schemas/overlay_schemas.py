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


class RenderPipelineBatchRequest(BaseModel):
    """Request body for applying one pipeline recipe to MANY images (mass-edit).
    Mirrors RenderPipelineRequest but takes a list of image paths; the blocks +
    tile config + replace_recipe are shared across all targets."""

    image_paths: list[str]
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


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class RenderPipelineResponse(BaseModel):
    """Ack for a synchronous (non-destructive) pipeline render that saved an overlay."""

    status: str = "overlay_saved"
    file: str
    overlay: str
    dimensions: list[int]
    hash: str


class OverlayRecipeResponse(BaseModel):
    """The pipeline recipe that produced an overlay.

    ``recipe`` is the raw overlays.json entry (overlay_file / created_at /
    operations); kept as a free-form dict so no recipe fields are dropped.
    """

    image_path: str
    recipe: dict[str, Any]


class OverlayActionResponse(BaseModel):
    """Ack for overlay revert/commit actions ({"status", "file"})."""

    status: str
    file: str


class RestoreModelItem(BaseModel):
    """One restoration model file found on disk."""

    name: str
    path: str
    size_mb: float


class RestoreModelListResponse(BaseModel):
    """Listing of restoration model files in a folder."""

    models: list[RestoreModelItem]
    folder: str


class ModelRegistryItem(BaseModel):
    """One curated registry model with its download status + metadata."""

    filename: str
    downloaded: bool
    local_size_mb: float | None = None
    url: str
    size_mb: float
    description: str


class ModelRegistryResponse(BaseModel):
    """All known models for a category with download status."""

    category: str
    folder: str
    models: list[ModelRegistryItem]


class ModelDownloadResponse(BaseModel):
    """Ack for a completed model download."""

    status: str = "downloaded"
    filename: str
    path: str
    size_mb: float
