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
    # The definition's ``video.frame_rule`` ("Nn+M", e.g. "4n+1", "17n+5"), or
    # None for a family that states none. The SPA derives its frame guidance
    # from this string (video_contract.frame_predicate is the authority); a
    # field not declared HERE is silently dropped by the response model.
    frame_rule: str | None = None


class TokenCountRequest(BaseModel):
    text: str = Field(default="")
    definition_id: str


class TokenCountResponse(BaseModel):
    tokens: int
    limit: int
    will_truncate: bool
    cutoff_char_index: int | None = None
