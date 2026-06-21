# backend/app/api/schemas/caption_context_schemas.py
from __future__ import annotations

from pydantic import BaseModel, Field


class DefinitionRef(BaseModel):
    """Lightweight definition entry for the top-bar selector."""

    id: str
    family: str
    name: str
    # Structured caption format key for this definition's family ("plain" =
    # flat text/tags; "ideogram4_json" = structured). Drives the frontend's
    # model-aware structured editor swap, so it MUST be served on this
    # selector route (not only the training definition list).
    caption_format: str = "plain"


class TokenCountRequest(BaseModel):
    text: str = Field(default="")
    definition_id: str


class TokenCountResponse(BaseModel):
    tokens: int
    limit: int
    will_truncate: bool
    cutoff_char_index: int | None = None
