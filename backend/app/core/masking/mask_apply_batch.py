"""Background batch mask-apply worker (APPLY section).

Wraps the SHARED ``MaskingService.mass_apply`` primitive (also called directly
by the training pipeline for ``recreate_masks`` — do not change its contract).
On completion it reconciles ``has_masked`` flags from the masked/ directory
(no re-scan / no re-score) and emits a one-off ``mask.apply_summary`` event so
the UI can surface skipped/missing-mask counts.

Module-level seams (monkeypatchable in tests):
  _dataset_path(ds)                              → dataset.path
  _mass_apply(path, opacity, overwrite, cb)      → masking_service.mass_apply(...)
  _reconcile_has_masked(ds)                      → flag sync from masked/ dir
  _emit_apply_summary(**kw)                       → broadcast mask.apply_summary
  run_mask_apply_batch(...)                      → the worker function

Cancellation: mass_apply has no internal cancel hook, so APPLY is cancellable
only while queued (the lane skips a cancelled-before-run task). Documented
limitation.
"""

from __future__ import annotations

import asyncio
import os

from app.core.logger import get_logger
from app.core.masking.masking_service import MaskingService
from app.core.tasks.task_manager import task_manager

logger = get_logger(__name__)


# ── Seams ─────────────────────────────────────────────────────────────────


def _dataset_path(dataset_name: str) -> str:
    from app.core.dataset_manager import dataset_manager as dm

    dataset = dm.get_dataset(dataset_name)
    if dataset is None:
        raise ValueError(f"Dataset '{dataset_name}' not found")
    return dataset.path


def _mass_apply(dataset_path: str, opacity: float, overwrite: bool, progress_callback) -> dict:
    """Call the shared primitive — signature unchanged (training also calls it)."""
    return MaskingService.get_instance().mass_apply(
        dataset_path, opacity, overwrite, progress_callback=progress_callback,
    )


def _reconcile_has_masked(dataset_name: str) -> None:
    """Set each media item's ``has_masked`` to reflect whether a masked/{stem}.jpg
    exists. Cheap directory listing — no scan, no re-score. Persists once."""
    from app.core.dataset_manager import dataset_manager as dm

    dataset = dm.get_dataset(dataset_name)
    if dataset is None:
        # Unreachable in normal flow (_dataset_path already verified existence);
        # logs a race where the dataset is removed mid-apply.
        logger.warning("mask_apply_reconcile_dataset_not_found", dataset=dataset_name)
        return
    masked_dir = os.path.join(dataset.path, "masked")
    masked_stems: set[str] = set()
    if os.path.isdir(masked_dir):
        for f in os.listdir(masked_dir):
            if f.lower().endswith(".jpg"):
                masked_stems.add(os.path.splitext(f)[0])

    changed = False
    for key, meta in dataset.media_metadata.items():
        stem = os.path.splitext(os.path.basename(key))[0]
        new_val = stem in masked_stems
        if meta.get("has_masked", False) != new_val:
            meta["has_masked"] = new_val
            changed = True
    if changed:
        dm._persist_dataset(dataset)


def _emit_apply_summary(*, dataset_name: str, applied: int, skipped: int,
                        missing_masks_count: int) -> None:
    """Broadcast a one-off mask.apply_summary event cross-thread (no-op if no loop)."""
    loop = task_manager._loop
    if loop is None:
        return
    from app.core.events import event_manager

    payload = {
        "dataset_name": dataset_name,
        "applied": applied,
        "skipped": skipped,
        "missing_masks_count": missing_masks_count,
    }
    asyncio.run_coroutine_threadsafe(
        event_manager.broadcast("mask.apply_summary", payload), loop,
    )


# ── Worker ────────────────────────────────────────────────────────────────


def run_mask_apply_batch(
    task_id: str,
    *,
    dataset_name: str,
    opacity: float,
    overwrite: bool,
) -> None:
    """Synchronous worker — runs on the GPU lane thread. Composites masks via the
    shared mass_apply (driving task progress), reconciles has_masked flags, and
    emits a summary event. No re-scan."""
    try:
        path = _dataset_path(dataset_name)

        def progress_cb(current, total, stem):
            task_manager.update(task_id, current=current, item=stem)

        result = _mass_apply(path, opacity, overwrite, progress_cb)
        _reconcile_has_masked(dataset_name)
        _emit_apply_summary(
            dataset_name=dataset_name,
            applied=result["applied"],
            skipped=result["skipped"],
            missing_masks_count=len(result["missing_masks"]),
        )
    except Exception as exc:  # noqa: BLE001
        task_manager.fail(task_id, str(exc))
        return

    task_manager.complete(task_id)
