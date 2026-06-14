"""Video-curation API schemas (cutlist parse, split, scene-detect, trim, health).

The :class:`Segment` shape is the canonical clip-span model shared between the
cutlist parser (``app.core.video.cutlist.Segment``) and the API — re-exported
here so the frontend and backend agree on one definition.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Re-export the canonical Segment from the core parser so there is exactly one
# definition of the clip-span shape across the codebase.
from app.core.video.cutlist import Segment

__all__ = [
    "Segment",
    "CutlistParseResponse",
    "VideoSplitRequest",
    "SceneDetectRequest",
    "SceneProposalsResponse",
    "TrimRequest",
    "TrimResponse",
    "ClipHealthFamily",
    "ClipHealthSummaryResponse",
]


# ── Cutlist parse ────────────────────────────────────────────────────────────


class CutlistParseResponse(BaseModel):
    """Synchronous result of parsing an uploaded cutlist file."""

    segments: list[Segment]
    format: str
    warnings: list[str] = Field(default_factory=list)


# ── Split ────────────────────────────────────────────────────────────────────


class VideoSplitRequest(BaseModel):
    """Body for ``POST /datasets/{name}/video/split`` (enqueues a cpu task)."""

    source_rel_path: str
    segments: list[Segment]
    mode: Literal["auto", "copy", "reencode"] = "auto"
    output_prefix: str | None = None
    archive_source: bool = True


# ── Scene detection ──────────────────────────────────────────────────────────


class SceneDetectRequest(BaseModel):
    """Body for ``POST /datasets/{name}/video/scene-detect`` (enqueues a cpu task)."""

    source_rel_path: str
    threshold: float = 27.0
    min_scene_len_s: float = 1.0


class SceneProposalsResponse(BaseModel):
    """Reviewed-proposals payload for ``GET /video/scene-proposals``.

    ``ready`` is ``False`` (and ``segments`` empty) until the detect task has
    written the proposals file.
    """

    segments: list[Segment] = Field(default_factory=list)
    ready: bool = False


# ── Trim ─────────────────────────────────────────────────────────────────────


class TrimRequest(BaseModel):
    """Body for ``PATCH /datasets/{name}/video/trim`` — non-destructive trim.

    ``None`` for a bound clears it (trim back to the clip edge).
    """

    media_file: str
    trim_start_s: float | None = None
    trim_end_s: float | None = None


class TrimResponse(BaseModel):
    """Ack for a trim update, carrying recomputed per-family clip warnings."""

    status: str
    clip_warnings: dict[str, list[str]] = Field(default_factory=dict)


# ── Health ───────────────────────────────────────────────────────────────────


class ClipHealthOffender(BaseModel):
    """One clip that violates a family's rules, with its warning list."""

    media_file: str
    warnings: list[str]


class ClipHealthFamily(BaseModel):
    """Per-family health rollup."""

    healthy: int = 0
    warning: int = 0
    offenders: list[ClipHealthOffender] = Field(default_factory=list)


class ClipHealthSummaryResponse(BaseModel):
    """Dataset-wide clip-health summary (per-family counts + offenders)."""

    total: int = 0
    families: dict[str, ClipHealthFamily] = Field(default_factory=dict)
