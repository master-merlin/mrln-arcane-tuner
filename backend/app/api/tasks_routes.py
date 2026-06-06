"""Background-task monitoring + control endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.tasks.task_manager import task_manager

router = APIRouter()


# ── Response Models ──────────────────────────────────────────────────────

class CancelTaskResponse(BaseModel):
    """Ack for a task-cancellation request."""

    status: str
    task_id: str


@router.get("/tasks")
async def list_tasks():
    """All known tasks (active + recent), for client re-sync on load/reconnect."""
    return [t.model_dump(mode="json") for t in task_manager.list()]


@router.post("/tasks/{task_id}/cancel", response_model=CancelTaskResponse)
async def cancel_task(task_id: str):
    if task_manager.get(task_id) is None:
        raise HTTPException(status_code=404, detail="Task not found")
    task_manager.cancel(task_id)
    return {"status": "cancelling", "task_id": task_id}
