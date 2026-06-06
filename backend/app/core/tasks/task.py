from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(BaseModel):
    id: str
    type: str
    title: str
    status: TaskStatus = TaskStatus.PENDING
    dataset_name: str | None = None
    # Caption-specific discriminator ("original" | "masked"); None for task
    # types that don't have a target. Lets a reopened modal re-hook to the
    # right run (mass-caption = original, mass-mask caption tab = masked).
    target: str | None = None
    total: int = 0
    current: int = 0
    current_item: str | None = None
    ok: int = 0
    failed: int = 0
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    # When False the task runs + broadcasts as usual but is hidden from the
    # user-facing Task Center (internal/background jobs, e.g. cache-stats warmup).
    user_visible: bool = True
