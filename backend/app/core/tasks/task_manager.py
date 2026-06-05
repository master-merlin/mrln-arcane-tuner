from __future__ import annotations

import asyncio
import threading
import time
import uuid

from app.core.events import event_manager
from app.core.logger import get_logger
from app.core.tasks.task import Task, TaskStatus

logger = get_logger(__name__)

# Min interval / delta between *progress* broadcasts for one task (anti-flood).
_PROGRESS_MIN_INTERVAL_S = 0.2
_PROGRESS_MIN_DELTA = 0.02  # fraction of total


class TaskManager:
    """In-memory registry + lifecycle for background tasks. Singleton via the
    module-level `task_manager`. Survives client reload (server-side); a server
    restart drops everything (by design)."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._cancels: dict[str, threading.Event] = {}
        self._last_emit: dict[str, float] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def set_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        self._loop = loop

    def create(self, *, type: str, title: str, total: int = 0,
               dataset_name: str | None = None) -> Task:
        task = Task(
            id=uuid.uuid4().hex,
            type=type,
            title=title,
            total=total,
            dataset_name=dataset_name,
            created_at=time.time(),
        )
        with self._lock:
            self._tasks[task.id] = task
            self._cancels[task.id] = threading.Event()
        self._broadcast(task)
        return task

    def start(self, task_id: str) -> None:
        t = self._tasks.get(task_id)
        if not t:
            return
        t.status = TaskStatus.RUNNING
        t.started_at = time.time()
        self._broadcast(t)

    def update(self, task_id: str, *, current: int | None = None,
               item: str | None = None, ok: int | None = None,
               failed: int | None = None) -> None:
        t = self._tasks.get(task_id)
        if not t:
            return
        if current is not None:
            t.current = current
        if item is not None:
            t.current_item = item
        if ok is not None:
            t.ok = ok
        if failed is not None:
            t.failed = failed
        self._broadcast(t, throttle=True)

    def complete(self, task_id: str) -> None:
        self._finish(task_id, TaskStatus.COMPLETED)

    def fail(self, task_id: str, error: str) -> None:
        t = self._tasks.get(task_id)
        if t:
            t.error = error
        self._finish(task_id, TaskStatus.FAILED)

    def cancel(self, task_id: str) -> None:
        ev = self._cancels.get(task_id)
        if ev:
            ev.set()

    def is_cancelled(self, task_id: str) -> bool:
        ev = self._cancels.get(task_id)
        return bool(ev and ev.is_set())

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def list(self) -> list[Task]:
        return list(self._tasks.values())

    def _finish(self, task_id: str, status: TaskStatus) -> None:
        t = self._tasks.get(task_id)
        if not t:
            return
        t.status = status
        t.finished_at = time.time()
        self._broadcast(t)

    def _broadcast(self, task: Task, *, throttle: bool = False) -> None:
        if throttle:
            now = time.time()
            last = self._last_emit.get(task.id, 0.0)
            frac_step = task.total or 1
            min_step = max(1, int(_PROGRESS_MIN_DELTA * frac_step))
            recent = (now - last) < _PROGRESS_MIN_INTERVAL_S
            small = (task.current % min_step != 0) and task.current < task.total
            if recent and small:
                return
            self._last_emit[task.id] = now
        if self._loop is None:
            return  # tests / pre-startup: no-op
        try:
            asyncio.run_coroutine_threadsafe(
                event_manager.broadcast("task_update", task.model_dump(mode="json")),
                self._loop,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("task_broadcast_failed", task_id=task.id, error=str(e))


task_manager = TaskManager()
