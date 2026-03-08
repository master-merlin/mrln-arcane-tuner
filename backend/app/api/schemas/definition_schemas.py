"""Model definition schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class CreateDefinitionRequest(BaseModel):
    """Request body for creating a new model definition."""
    id: str
    family: str
    name: str
    version: str = "1.0"
    defaults: dict[str, Any] = {}
    components: dict[str, dict[str, Any]] = {}


class UpdateDefinitionRequest(BaseModel):
    """Request body for updating an existing model definition."""
    name: str | None = None
    version: str | None = None
    defaults: dict[str, Any] | None = None
    components: dict[str, dict[str, Any]] | None = None


class VRAMEstimateRequest(BaseModel):
    """Request body for VRAM estimation."""
    definition_id: str
    config: dict[str, Any]
