"""Background harmonize worker — convert+rename+rescan a whole dataset.

Wraps the (otherwise synchronous, multi-minute) ``dataset_manager.harmonize_files``
with per-file progress. CPU/IO-bound; runs on the shared ``gpu`` lane because it
renames/rewrites EVERY file and must serialize against all other dataset mutators
(caption/mask/crop/adjust/rescan). No model/unload.

Cancellation: Pass-1 temp-renaming is not safely interruptible mid-run, so
harmonize is cancellable only WHILE QUEUED (the lane skips a cancelled-before-run
task). Documented limitation.

Module-level seams (monkeypatchable in tests):
  _harmonize(name, progress_cb)   → dataset_manager.harmonize_files(...)
  _emit_harmonize_summary(**kw)   → broadcast harmonize.summary
  run_harmonize_batch(...)        → the worker
"""

from __future__ import annotations

import asyncio

from app.core.logger import get_logger
from app.core.tasks.task_manager import task_manager

logger = get_logger(__name__)


def _harmonize(dataset_name: str, progress_cb) -> dict:
    from app.core.dataset_manager import dataset_manager as dm

    return dm.harmonize_files(dataset_name, progress_cb=progress_cb)


def _emit_harmonize_summary(*, dataset_name: str, processed: int, converted: int,
                            renamed: int) -> None:
    """Broadcast a one-off harmonize.summary event cross-thread (no-op pre-loop)."""
    loop = task_manager._loop
    if loop is None:
        return
    from app.core.events import event_manager

    payload = {
        "dataset_name": dataset_name,
        "processed": processed,
        "converted": converted,
        "renamed": renamed,
    }
    asyncio.run_coroutine_threadsafe(
        event_manager.broadcast("harmonize.summary", payload), loop,
    )


def run_harmonize_batch(task_id: str, *, dataset_name: str) -> None:
    """Synchronous worker — gpu lane thread. Drives task progress from
    harmonize_files' per-file callback, then emits a summary and completes."""
    # Mirror `current` into `ok` so the Task Center's "<n> done" tracks live —
    # harmonize has no failure-per-file branch that completes the task, so every
    # processed pair is a success. (Without this, `ok` stayed 0 and the row read
    # "0 done" even after all files were harmonized.)
    def progress_cb(current, total, fname):
        task_manager.update(task_id, current=current, item=fname, ok=current)

    try:
        result = _harmonize(dataset_name, progress_cb)
    except Exception as exc:  # noqa: BLE001
        task_manager.fail(task_id, str(exc))
        return

    # Reconcile to the authoritative processed count: progress counts every pair
    # visited (a conversion error skips a pair without appending it), whereas
    # `processed` is the number actually harmonized.
    task_manager.update(task_id, ok=result["processed"])

    _emit_harmonize_summary(
        dataset_name=dataset_name,
        processed=result["processed"],
        converted=result["converted"],
        renamed=result["renamed"],
    )
    task_manager.complete(task_id)
