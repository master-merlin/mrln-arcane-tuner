"""Dataset CRUD and management schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CreateDatasetRequest(BaseModel):
    """Request body for dataset creation."""
    name: str
    description: str = ""
    classifier: str = ""
    trigger_word: str = ""
    tags: list[str] = Field(default_factory=list)
    notes: str = ""


class UpdateDatasetRequest(BaseModel):
    """Request body for dataset update."""
    name: str
    description: str
    classifier: str = ""
    trigger_word: str = ""
    tags: list[str] = Field(default_factory=list)
    notes: str = ""


class CaptionRequest(BaseModel):
    """Request body for caption save."""
    content: str


class ToggleEnabledRequest(BaseModel):
    """Request body for image enable/disable toggle."""
    enabled: bool


class ImportPathRequest(BaseModel):
    archive_path: str
    # Constrained so an unexpected value 422s instead of silently becoming a 409.
    on_conflict: Literal["rename", "overwrite"] | None = None
    new_name: str | None = None


# ── Response models ──────────────────────────────────────────────────────
# Replace the raw-dict returns the CRUD routes used to emit, per the
# API_CONVENTIONS "every response is a Pydantic model" rule. Field sets mirror
# exactly what the handlers returned so wiring ``response_model=`` is a pure
# typing change with no payload drift.


class DatasetDeletedResponse(BaseModel):
    """Ack for unregistering a dataset."""
    status: str = "deleted"
    name: str


class MediaPairDeletedResponse(BaseModel):
    """Ack for deleting a media file + its caption sidecar."""
    status: str = "deleted"
    file: str


class UploadResponse(BaseModel):
    """Ack for a single-file upload into a dataset."""
    filename: str
    status: str = "uploaded"


class CaptionContentResponse(BaseModel):
    """A caption file's contents."""
    content: str


class CaptionSavedResponse(BaseModel):
    """Ack for a caption save."""
    status: str = "saved"


class ToggleEnabledResponse(BaseModel):
    """New enabled state for a single image."""
    media_file: str
    enabled: bool


class EnableAllResponse(BaseModel):
    """Count of images flipped back to enabled."""
    reset_count: int


class DatasetPairResponse(BaseModel):
    """One image/caption pair row. Canonical contract mirrored by the frontend
    ``DatasetPair`` interface (services/dataset.ts). Results are filtered to rows
    that have a media file, so ``media_file``/``media_type`` are always set."""
    stem: str
    media_file: str
    media_type: Literal["image", "video"]
    # null (not absent) when the image has no caption sidecar.
    caption_file: str | None = None
    # Present only for media rows (which is all returned rows).
    size_bytes: int | None = None
    caption_content: str = ""
    masked_caption_content: str | None = None
    # Free-form per-item media_metadata; shape varies by enrichment state.
    metadata: dict | None = None
