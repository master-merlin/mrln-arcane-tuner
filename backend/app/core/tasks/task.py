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
    total: int = 0
    current: int = 0
    current_item: str | None = None
    ok: int = 0
    failed: int = 0
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
