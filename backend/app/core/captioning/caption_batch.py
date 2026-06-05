"""Background batch-captioning worker.

Module-level seams (all monkeypatchable in tests):
  _get_service()        → CaptionService.get_instance()
  _full_path(ds, rel)   → absolute image path on disk
  _write_caption(...)   → persist caption + update metadata flag
  _emit_caption_written(**kw) → fire caption.written SSE event
  run_caption_batch(...)      → the worker function itself
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.captioning.caption_service import CaptionService
from app.core.logger import get_logger
from app.core.tasks.task import TaskStatus
from app.core.tasks.task_manager import task_manager

logger = get_logger(__name__)


# ── Seams ─────────────────────────────────────────────────────────────────


def _get_service() -> CaptionService:
    """Return the CaptionService singleton."""
    return CaptionService.get_instance()


def _full_path(dataset_name: str, rel: str) -> str:
    """Resolve the absolute image path for *rel* inside *dataset_name*."""
    from app.core.dataset_manager import dataset_manager as dm

    dataset = dm.get_dataset(dataset_name)
    if dataset is None:
        raise ValueError(f"Dataset '{dataset_name}' not found")
    return str(Path(dataset.path) / rel)


def _write_caption(dataset_name: str, rel: str, text: str, target: str) -> None:
    """Persist *text* to disk and update the media-item metadata flag.

    Verified storage locations (dataset_manager.py):
    - original: ``{dataset.path}/{stem}.txt``  → sets ``has_caption = True``
      (mirrors save_caption, lines 1364-1410)
    - masked:   ``{dataset.path}/masked/{stem}.txt`` → sets
      ``has_masked_caption = True``  (mirrors generate_caption route, lines
      81-96 of caption_routes.py)
    """
    from app.core.dataset_manager import dataset_manager as dm

    dataset = dm.get_dataset(dataset_name)
    if dataset is None:
        raise ValueError(f"Dataset '{dataset_name}' not found")

    stem = Path(rel).stem
    dataset_root = Path(dataset.path)

    if target == "masked":
        masked_dir = dataset_root / "masked"
        masked_dir.mkdir(parents=True, exist_ok=True)
        caption_path = masked_dir / f"{stem}.txt"
        caption_path.write_text(text, encoding="utf-8")

        lookup_key = rel.replace("\\", "/")
        if lookup_key in dataset.media_metadata:
            dataset.media_metadata[lookup_key]["has_masked_caption"] = True
            dm._persist_media_item(dataset, lookup_key)
    else:
        # original: sibling .txt next to the image
        caption_path = dataset_root / f"{stem}.txt"
        caption_path.write_text(text, encoding="utf-8")

        lookup_key = rel.replace("\\", "/")
        for key, meta in dataset.media_metadata.items():
            media_stem = Path(key).stem
            if media_stem == stem:
                meta["has_caption"] = True
                dm._persist_media_item(dataset, key)
                break


def _emit_caption_written(
    *,
    dataset_name: str,
    media_file: str,
    caption: str,
    target: str,
) -> None:
    """Broadcast a ``caption.written`` SSE event cross-thread (no-op if no loop)."""
    loop = task_manager._loop
    if loop is None:
        return
    from app.core.events import event_manager

    payload = {
        "dataset_name": dataset_name,
        "media_file": media_file,
        "caption": caption,
        "target": target,
    }
    asyncio.run_coroutine_threadsafe(
        event_manager.broadcast("caption.written", payload),
        loop,
    )


# ── Worker ────────────────────────────────────────────────────────────────


def run_caption_batch(
    task_id: str,
    *,
    dataset_name: str,
    image_rel_paths: list[str],
    model_id: str,
    params: dict,
    system_prompt: str | None,
    target: str,
) -> None:
    """Synchronous worker — runs on the GPU lane thread.

    Iterates over *image_rel_paths*, calls the caption service for each image,
    persists the result, and updates task progress.  Honours cancellation
    before each image.  Always calls ``CaptionService.unload_models()`` in
    the finally block to free VRAM.
    """
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
                call_params = params.copy()
                if system_prompt:
                    call_params["system_prompt"] = system_prompt

                caption = service.generate_caption(
                    image_path=_full_path(dataset_name, rel),
                    model_id=model_id,
                    params=call_params,
                )
                _write_caption(dataset_name, rel, caption, target)
                ok += 1
                _emit_caption_written(
                    dataset_name=dataset_name,
                    media_file=rel,
                    caption=caption,
                    target=target,
                )
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.warning(
                    "caption_batch_item_failed",
                    task_id=task_id,
                    rel_path=rel,
                    error=str(exc),
                )

            task_manager.update(task_id, current=i + 1, item=rel, ok=ok, failed=failed)

    except Exception as exc:  # unrecoverable setup error
        task_manager.fail(task_id, str(exc))
        return
    finally:
        CaptionService.unload_models()

    if cancelled:
        task_manager._finish(task_id, TaskStatus.CANCELLED)
    else:
        task_manager.complete(task_id)
