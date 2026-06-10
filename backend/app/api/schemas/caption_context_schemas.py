# backend/app/api/schemas/caption_context_schemas.py
from __future__ import annotations

from pydantic import BaseModel, Field


class DefinitionRef(BaseModel):
    """Lightweight definition entry for the top-bar selector."""
    id: str
    family: str
    name: str


class TokenCountRequest(BaseModel):
    text: str = Field(default="")
    definition_id: str


class TokenCountResponse(BaseModel):
    tokens: int
    limit: int
    will_truncate: bool
    cutoff_char_index: int | None = None
