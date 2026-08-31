from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import deque

from app.core.drain import DrainActive, is_draining
from app.core.events import event_manager
from app.core.logger import get_logger
from app.core.tasks.task import Task, TaskStatus

logger = get_logger(__name__)

# Min interval / delta between *progress* broadcasts for one task (anti-flood).
_PROGRESS_MIN_INTERVAL_S = 0.2
_PROGRESS_MIN_DELTA = 0.02  # fraction of total

# Cap on retained *terminal* (completed/failed/cancelled) task records. Without
# this, _tasks/_cancels/_last_emit only ever grew for the life of the process —
# a long session of batch ops (caption batches, rescan batches, ...) leaks
# memory and bloats every GET /tasks poll unboundedly. Active (pending/
# running) tasks are never pruned regardless of this cap.
_MAX_TERMINAL = 500


class TaskManager:
    """In-memory registry + lifecycle for background tasks. Singleton via the
    module-level `task_manager`. Survives client reload (server-side); a server
    restart drops everything (by design).

    Concurrency model: at most ONE writer per task id at a time — guaranteed by
    the FIFO lane worker (a task is only ever advanced by the single lane thread
    running it). `_lock` therefore only guards the registry mutation in
    `create()`; the per-task lifecycle mutations are race-free under that
    single-writer assumption, not because they hold the lock."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._cancels: dict[str, threading.Event] = {}
        self._last_emit: dict[str, float] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self._lanes: dict[str, list[tuple[str, object]]] = {}
        self._lane_threads: dict[str, threading.Thread] = {}
        # FIFO of terminal task ids, oldest first — drives the prune in
        # _finish(). Guarded by _lock (multiple lanes can finish different
        # tasks concurrently on their own worker threads).
        self._terminal_order: deque[str] = deque()

    def set_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        self._loop = loop

    def create(self, *, type: str, title: str, total: int = 0,
               dataset_name: str | None = None,
               target: str | None = None,
               user_visible: bool = True) -> Task:
        task = Task(
            id=uuid.uuid4().hex,
            type=type,
            title=title,
            total=total,
            dataset_name=dataset_name,
            target=target,
            user_visible=user_visible,
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

    def finish_batch(self, task_id: str, *, ok: int, failed: int,
                     error: str | None = None) -> None:
        """Terminal state for a COUNTED batch, derived from the batch's own tally
        instead of from "the worker returned without raising".

        Every counted adopter used to end in a bare ``complete(task_id)``: the
        per-item ``except`` swallowed the exception into a ``failed`` counter and
        nothing downstream ever read that counter. A refine batch in which the
        ONE item timed out was therefore recorded ``status=completed, ok=0,
        failed=1, error=None`` — the Task Center showed a finished task, nothing
        had changed on disk, and the user had no reason given (UAT round 4,
        LANE-52). That is the silent-failure shape ARCHITECTURE D10 forbids.

        The rule, in one place so all adopters agree (RULE-21, one producer):

        * ``failed == 0``           → COMPLETED, no error.
        * ``failed > 0, ok == 0``   → FAILED. Nothing succeeded, so the task did
          not do its job and must not present as success. ``error`` carries the
          reason (the last item's exception text when the caller passes one).
        * ``failed > 0, ok > 0``    → COMPLETED — real work landed and the
          partial result is on disk — but ``error`` is set to the same summary
          so the failure is still *stated*. A completed task carrying an error
          is deliberate: the Task Center renders it whenever it is present.

        Cancellation is NOT this method's business — a cancelled batch is
        terminal via :meth:`finish_cancelled` before it gets here.
        """
        if failed <= 0:
            self.complete(task_id)
            return
        total = ok + failed
        summary = f"{failed} of {total} items failed"
        if error:
            summary = f"{summary} (last: {error})"
        if ok <= 0:
            self.fail(task_id, summary)
            return
        t = self._tasks.get(task_id)
        if t:
            t.error = summary
        self.complete(task_id)

    def cancel(self, task_id: str) -> None:
        ev = self._cancels.get(task_id)
        if ev:
            ev.set()

    def finish_cancelled(self, task_id: str) -> None:
        """Public counterpart to complete()/fail() for the CANCELLED terminal
        state — batch workers call this from their own finally/epilogue when
        they observe a mid-run cancellation, instead of reaching into the
        private _finish()."""
        self._finish(task_id, TaskStatus.CANCELLED)

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
        # A dataset-scoped task just ended — it may have written/renamed files
        # (caption batch, and any future mass op that becomes a task). Tell
        # clients to reconcile that dataset's caches. Fires for completed AND
        # failed/cancelled: partial runs still change files on disk, and the
        # client-side reconcile is idempotent.
        if t.dataset_name:
            self._emit_dataset_invalidated(t.dataset_name)
        self._prune_terminal(task_id)

    def _prune_terminal(self, task_id: str) -> None:
        """Record *task_id* as newly-terminal and evict the oldest terminal
        record(s) once the retained set exceeds _MAX_TERMINAL. Active tasks
        are never touched — only ids that passed through _finish() land in
        _terminal_order."""
        with self._lock:
            self._terminal_order.append(task_id)
            while len(self._terminal_order) > _MAX_TERMINAL:
                oldest = self._terminal_order.popleft()
                self._tasks.pop(oldest, None)
                self._cancels.pop(oldest, None)
                self._last_emit.pop(oldest, None)

    def _emit_dataset_invalidated(self, name: str) -> None:
        """Broadcast the coarse ``dataset.invalidated`` signal (see
        DatasetManager._emit_dataset_invalidated). Threadsafe; no-op pre-loop."""
        if self._loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                event_manager.broadcast("dataset.invalidated", {"name": name}),
                self._loop,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("dataset_invalidated_broadcast_failed", name=name, error=str(e))

    def enqueue(self, task_id: str, worker_fn, *, lane: str = "gpu") -> None:
        """Append (task_id, worker_fn) to a lane FIFO and ensure the lane's
        single worker thread is running. worker_fn is called as worker_fn(task_id)
        on the lane thread when this task reaches the front of the queue."""
        if lane == "gpu" and is_draining():
            raise DrainActive("Update pending — new GPU tasks are paused until restart.")
        with self._lock:
            self._lanes.setdefault(lane, []).append((task_id, worker_fn))
            # Presence in `_lane_threads` IS the "lane has a live worker" flag —
            # added here and removed by `_run_lane` when it drains, both under
            # `_lock`. This closes the is_alive() race where a thread caught
            # between "saw empty / returning" and actually dying would leave a
            # newly-enqueued task orphaned in `pending`.
            if lane not in self._lane_threads:
                th = threading.Thread(target=self._run_lane, args=(lane,), daemon=True)
                self._lane_threads[lane] = th
                th.start()

    def _run_lane(self, lane: str) -> None:
        while True:
            with self._lock:
                queue = self._lanes.get(lane, [])
                if not queue:
                    self._lane_threads.pop(lane, None)  # release the lane under lock
                    return
                task_id, worker_fn = queue.pop(0)

            # Cancelled before it ran → finalize as cancelled, skip the worker.
            if self.is_cancelled(task_id):
                self._finish(task_id, TaskStatus.CANCELLED)
                continue

            self.start(task_id)
            try:
                worker_fn(task_id)
            except Exception as e:  # worker should normally finalize itself
                logger.warning("task_worker_crashed", task_id=task_id, error=str(e))
                t = self.get(task_id)
                if t and t.status == TaskStatus.RUNNING:
                    self.fail(task_id, str(e))
            else:
                t = self.get(task_id)
                if t and t.status == TaskStatus.RUNNING:
                    # Worker returned without finalizing → infer from cancel flag.
                    if not self.is_cancelled(task_id):
                        self.complete(task_id)
                    else:
                        self._finish(task_id, TaskStatus.CANCELLED)

    def join_lane(self, lane: str, timeout: float | None = None) -> None:
        """Test helper — block until the lane's worker thread drains."""
        th = self._lane_threads.get(lane)
        if th:
            th.join(timeout)

    def _broadcast(self, task: Task, *, throttle: bool = False) -> None:
        if self._loop is None:
            return  # tests / pre-startup: no-op (don't advance the throttle clock)
        if throttle:
            now = time.time()
            last = self._last_emit.get(task.id, 0.0)
            recent = (now - last) < _PROGRESS_MIN_INTERVAL_S
            if task.total > 0:
                min_step = max(1, int(_PROGRESS_MIN_DELTA * task.total))
                small = (task.current % min_step != 0) and task.current < task.total
            else:
                # Unknown total: there is no percentage to step on, so the
                # delta half of the rule cannot apply and time alone throttles.
                # Previously ``frac_step = total or 1`` made min_step 1, so
                # ``current % 1 == 0`` always held, ``small`` was always False,
                # and an unknown-total task broadcast on EVERY update.
                small = True
            if recent and small:
                return
            self._last_emit[task.id] = now
        try:
            asyncio.run_coroutine_threadsafe(
                event_manager.broadcast("task_update", task.model_dump(mode="json")),
                self._loop,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("task_broadcast_failed", task_id=task.id, error=str(e))


task_manager = TaskManager()
