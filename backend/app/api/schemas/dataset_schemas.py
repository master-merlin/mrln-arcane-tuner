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
