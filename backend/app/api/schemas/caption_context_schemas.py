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
    # The fps every clip is resampled to at ingestion, or None when a clip
    # keeps its own fps (``VideoProfile.ingest_fps`` — the value the trainer's
    # ingestion resolves, from the definition's ``video.ingest_at_native_fps``).
    # ``frame_rule`` counts TRAINING frames: with a clock stated here a clip of
    # N source frames at S fps trains ``whole_frames(N / S, ingest_fps)`` frames
    # (``video_contract.py``), and the SPA must apply the rule to THAT number. Declared here for the same
    # reason as ``frame_rule``.
    ingest_fps: float | None = None


class TokenCountRequest(BaseModel):
    text: str = Field(default="")
    definition_id: str


class TokenCountResponse(BaseModel):
    tokens: int
    limit: int
    will_truncate: bool
    cutoff_char_index: int | None = None
