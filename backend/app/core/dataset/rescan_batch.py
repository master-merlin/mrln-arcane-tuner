"""Background batch-rescan worker.

Module-level seams (all monkeypatchable in tests):
  _scan(name, force_full, progress_cb) → dataset_manager.scan_dataset(...)
  _dataset_file_count(name)            → multimedia file count on disk
  _unload()                            → ScoringService.unload_models()
  count_multimedia(names)              → sum of file counts (seeds task total)
  run_rescan_batch(...)                → the worker function itself

Cancellation is checked at dataset boundaries. Persistence is atomic per
dataset (scan_dataset writes only in its finalize stage), so a cancel between
datasets leaves already-scanned datasets fully persisted and the rest untouched
— no in-memory restore needed.
"""

from __future__ import annotations

from app.core.logger import get_logger
from app.core.tasks.task_manager import task_manager

logger = get_logger(__name__)


# ── Seams ─────────────────────────────────────────────────────────────────


def _scan(name: str, force_full: bool, progress_cb) -> None:
    """Run the staged scan for one dataset, forwarding the progress callback."""
    from app.core.dataset_manager import dataset_manager as dm

    dm.scan_dataset(name, force_full, progress_cb=progress_cb)


def _dataset_file_count(name: str) -> int:
    """Multimedia file count for a registered dataset (0 if unknown/missing)."""
    from app.core.dataset_manager import dataset_manager as dm

    ds = dm.get_dataset(name)
    if ds is None:
        return 0
    return dm.count_multimedia_files(ds.path)


def _unload() -> None:
    """Free VRAM held by the scoring model. Best-effort; never raises."""
    try:
        from app.core.scoring.scoring_service import ScoringService

        ScoringService.unload_models()
    except Exception:  # noqa: BLE001
        pass


def count_multimedia(dataset_names: list[str]) -> int:
    """Sum of multimedia file counts across *dataset_names* — seeds task total."""
    return sum(_dataset_file_count(n) for n in dataset_names)


# ── Worker ────────────────────────────────────────────────────────────────


def run_rescan_batch(
    task_id: str,
    *,
    dataset_names: list[str],
    force_full: bool,
    total: int,
) -> None:
    """Synchronous worker — runs on the GPU lane thread.

    Scans each dataset in turn, mapping each dataset's per-file progress into a
    global running counter so a library sweep shows one file-granular bar.
    Honours cancellation between datasets; isolates per-dataset failures; always
    unloads the scoring model in the finally block.
    """
    ok = 0
    failed = 0
    done = 0
    cancelled = False

    try:
        for name in dataset_names:
            if task_manager.is_cancelled(task_id):
                cancelled = True
                break

            n = _dataset_file_count(name)
            base = done

            # scan_dataset calls progress_cb(cur, tot, fname) per file; we offset
            # the dataset-local cur by `base` to keep one continuous global bar.
            def progress_cb(cur, tot, fname, _base=base, _name=name):
                task_manager.update(
                    task_id,
                    current=_base + cur,
                    item=f"{_name} → {fname}",
                    ok=ok,
                    failed=failed,
                )

            try:
                _scan(name, force_full, progress_cb)
                ok += n
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.warning(
                    "rescan_dataset_failed",
                    task_id=task_id, dataset=name, error=str(exc),
                )

            # Advance even on failure so the bar doesn't stall. Note the
            # asymmetry: `ok` counts files (in succeeded datasets), `failed`
            # counts datasets — so on a failed dataset `current` still moves by n.
            done += n
            task_manager.update(task_id, current=done, ok=ok, failed=failed)

    except Exception as exc:  # unrecoverable setup error
        task_manager.fail(task_id, str(exc))
        return
    finally:
        _unload()

    if cancelled:
        task_manager.finish_cancelled(task_id)
    else:
        task_manager.complete(task_id)
