"""Background batch mask-generation worker (CREATE section).

Module-level seams (monkeypatchable in tests):
  _get_service()        → MaskingService.get_instance()
  _full_path(ds, rel)   → absolute source-image path on disk (path-guarded)
  _save_mask(ds, rel, mask) → save masks/{stem}.png + set has_mask + persist
  _unload()             → MaskingService.unload_models()
  run_mask_generate_batch(...) → the worker function
"""

from __future__ import annotations

from pathlib import Path

from app.core.logger import get_logger
from app.core.masking.masking_service import MaskingService
from app.core.tasks.task import TaskStatus
from app.core.tasks.task_manager import task_manager

logger = get_logger(__name__)


# ── Seams ─────────────────────────────────────────────────────────────────


def _get_service() -> MaskingService:
    return MaskingService.get_instance()


def _full_path(dataset_name: str, rel: str) -> str:
    """Resolve the absolute source-image path, guarding against traversal."""
    from app.api._path_guard import validate_path_within
    from app.core.dataset_manager import dataset_manager as dm

    dataset = dm.get_dataset(dataset_name)
    if dataset is None:
        raise ValueError(f"Dataset '{dataset_name}' not found")
    root = Path(dataset.path)
    return str(validate_path_within(root / rel, root))


def _save_mask(dataset_name: str, rel: str, mask_image) -> None:
    """Save the mask to masks/{stem}.png and flip has_mask on the media item.

    Mirrors the single-image generate route's metadata update so the grid
    repaints live as masks land. Uses the sync persist (worker is on a thread)."""
    from app.core.dataset_manager import dataset_manager as dm

    dataset = dm.get_dataset(dataset_name)
    if dataset is None:
        raise ValueError(f"Dataset '{dataset_name}' not found")
    masks_dir = Path(dataset.path) / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(rel).stem
    mask_image.save(str(masks_dir / f"{stem}.png"))

    lookup_key = rel.replace("\\", "/")
    if lookup_key in dataset.media_metadata:
        dataset.media_metadata[lookup_key]["has_mask"] = True
        dm._persist_media_item(dataset, lookup_key)


def _unload() -> None:
    """Free VRAM held by the segmentation model. Best-effort; never raises."""
    try:
        MaskingService.get_instance().unload_models()
    except Exception:  # noqa: BLE001
        pass


# ── Worker ────────────────────────────────────────────────────────────────


def run_mask_generate_batch(
    task_id: str,
    *,
    dataset_name: str,
    image_rel_paths: list[str],
    model_id: str,
    params: dict,
) -> None:
    """Synchronous worker — runs on the GPU lane thread. Generates a mask per
    image, saves it, updates progress; isolates per-item failures; unloads the
    model in finally."""
    ok = 0
    failed = 0
    cancelled = False

    try:
        service = _get_service()

        for i, rel in enumerate(image_rel_paths):
            if task_manager.is_cancelled(task_id):
                cancelled = True
                break

            try:
                src = _full_path(dataset_name, rel)
                mask = service.generate_mask(src, model_id, params)
                _save_mask(dataset_name, rel, mask)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.warning(
                    "mask_generate_item_failed",
                    task_id=task_id, rel_path=rel, error=str(exc),
                )

            task_manager.update(task_id, current=i + 1, item=rel, ok=ok, failed=failed)

    except Exception as exc:  # unrecoverable setup error
        task_manager.fail(task_id, str(exc))
        return
    finally:
        _unload()

    if cancelled:
        task_manager._finish(task_id, TaskStatus.CANCELLED)
    else:
        task_manager.complete(task_id)
