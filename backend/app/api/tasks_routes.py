"""Background-task monitoring + control endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.tasks.task_manager import task_manager

router = APIRouter()


@router.get("/tasks")
async def list_tasks():
    """All known tasks (active + recent), for client re-sync on load/reconnect."""
    return [t.model_dump(mode="json") for t in task_manager.list()]


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    if task_manager.get(task_id) is None:
        raise HTTPException(status_code=404, detail="Task not found")
    task_manager.cancel(task_id)
    return {"status": "cancelling", "task_id": task_id}
