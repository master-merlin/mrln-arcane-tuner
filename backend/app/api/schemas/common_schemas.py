"""Cross-cutting response schemas shared by multiple route modules.

These model the small action/acknowledgement payloads (task-enqueue acks,
status acks) that recur across nearly every router. Keeping them here lets each
domain wire ``response_model=`` without re-declaring the same shape, and gives
the frontend a single canonical contract per shape.
"""

from __future__ import annotations

from pydantic import BaseModel


class TaskEnqueuedResponse(BaseModel):
    """Ack for an endpoint that enqueues a backend task and returns immediately.

    The ubiquitous ``{"task_id": ...}`` shape returned by every GPU/background
    lane enqueue endpoint (rescan, captioning, masking, crop, harmonize, ...).
    """

    task_id: str
