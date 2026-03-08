"""Dataset CRUD and management schemas."""

from __future__ import annotations

from pydantic import BaseModel


class CreateDatasetRequest(BaseModel):
    """Request body for dataset creation."""
    name: str
    description: str = ""
    classifier: str = ""


class UpdateDatasetRequest(BaseModel):
    """Request body for dataset update."""
    name: str
    description: str
    classifier: str = ""


class CaptionRequest(BaseModel):
    """Request body for caption save."""
    content: str


class ToggleEnabledRequest(BaseModel):
    """Request body for image enable/disable toggle."""
    enabled: bool
