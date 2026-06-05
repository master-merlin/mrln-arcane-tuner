"""Background batch-crop worker (Analyze modal "Crop all").

Loops the candidate list, cropping each image to its own analysis-derived
target via the unchanged ``dataset_manager.crop_media``. CPU-bound (PIL); runs
on the shared serial ``gpu`` lane so it never races a caption/mask/rescan task
on the same dataset (those overwrite/read the same files). No model is loaded,
so there is no VRAM unload step.

Module-level seam (monkeypatchable in tests):
  _crop(dataset_name, path, target_w, target_h, origin)
        → dataset_manager.crop_media(...)   (origin-based; crop_x/crop_y omitted)
  run_crop_batch(...) → the worker function
"""

from __future__ import annotations

from app.core.logger import get_logger
from app.core.tasks.task import TaskStatus
from app.core.tasks.task_manager import task_manager

logger = get_logger(__name__)


# ── Seam ──────────────────────────────────────────────────────────────────


def _crop(dataset_name: str, path: str, target_w: int, target_h: int, origin: str) -> None:
    """Crop one image in place. Persists the item + invalidates masks (handled
    inside crop_media)."""
    from app.core.dataset_manager import dataset_manager as dm

    dm.crop_media(dataset_name, path, target_w, target_h, origin)


# ── Worker ────────────────────────────────────────────────────────────────


def run_crop_batch(
    task_id: str,
    *,
    dataset_name: str,
    items: list,
    origin: str,
) -> None:
    """Synchronous worker — runs on the lane thread. Crops each item to its own
    target dims; isolates per-item failures; checks cancellation per item. Each
    crop_media call persists its image (entity.changed repaints the grid live),
    so a mid-batch cancel leaves cropped images persisted and the rest untouched.
    """
    ok = 0
    failed = 0
    cancelled = False

    try:
        for i, item in enumerate(items):
            if task_manager.is_cancelled(task_id):
                cancelled = True
                break

            path = item["path"] if isinstance(item, dict) else item.path
            tw = item["target_width"] if isinstance(item, dict) else item.target_width
            th = item["target_height"] if isinstance(item, dict) else item.target_height

            try:
                _crop(dataset_name, path, tw, th, origin)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.warning(
                    "crop_item_failed",
                    task_id=task_id, rel_path=path, error=str(exc),
                )

            task_manager.update(task_id, current=i + 1, item=path, ok=ok, failed=failed)

    except Exception as exc:  # unrecoverable setup error
        task_manager.fail(task_id, str(exc))
        return

    if cancelled:
        task_manager._finish(task_id, TaskStatus.CANCELLED)
    else:
        task_manager.complete(task_id)
