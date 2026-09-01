"""Cross-cutting response schemas shared by multiple route modules.

These model the small action/acknowledgement payloads (task-enqueue acks,
status acks) that recur across nearly every router. Keeping them here lets each
domain wire ``response_model=`` without re-declaring the same shape, and gives
the frontend a single canonical contract per shape.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TaskEnqueuedResponse(BaseModel):
    """Ack for an endpoint that enqueues a backend task and returns immediately.

    The ubiquitous ``{"task_id": ...}`` shape returned by every GPU/background
    lane enqueue endpoint (rescan, captioning, masking, crop, harmonize, ...).
    """

    task_id: str


class ErrorResponse(BaseModel):
    """Standard error envelope (``_docs/API_CONVENTIONS.md`` → error_responses).

    Every error response — HTTPExceptions, validation failures, and the
    unhandled-500 fallback — is serialized through this shape so clients can
    rely on one contract. ``detail`` carries the human-readable message (kept
    verbatim from the raised ``HTTPException`` so existing consumers that read
    structured ``detail`` payloads keep working); ``error_code`` is a stable
    machine-readable token; ``context`` holds optional structured extras.
    """

    detail: Any = Field(..., description="Human-readable message (or structured detail payload).")
    error_code: str = Field(..., description="Stable machine-readable error code.")
    context: dict[str, Any] = Field(default_factory=dict, description="Optional structured context.")
