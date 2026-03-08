"""Template CRUD schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class CreateTemplateRequest(BaseModel):
    """Request body for creating a template."""
    category: str
    name: str
    definition_id: str | None = None
    model_id: str | None = None
    is_default: bool = False
    readonly: bool = False
    system_prompt: str | None = None
    config: dict[str, Any] = {}


class UpdateTemplateRequest(BaseModel):
    """Request body for updating a template."""
    name: str | None = None
    is_default: bool | None = None
    system_prompt: str | None = None
    config: dict[str, Any] | None = None
