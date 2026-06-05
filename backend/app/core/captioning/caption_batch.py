"""Background batch-captioning worker.

Module-level seams (all monkeypatchable in tests):
  _get_service()        → CaptionService.get_instance()
  _full_path(ds, rel)   → absolute image path on disk
  _masked_path(ds, rel) → masked composite path (masked/{stem}.jpg)
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
    """Resolve the absolute image path for *rel* inside *dataset_name*.

    Guards against path traversal (``rel`` comes from the request body) the
    same way the single-image caption route does.
    """
    from app.api._path_guard import validate_path_within
    from app.core.dataset_manager import dataset_manager as dm

    dataset = dm.get_dataset(dataset_name)
    if dataset is None:
        raise ValueError(f"Dataset '{dataset_name}' not found")
    root = Path(dataset.path)
    return str(validate_path_within(root / rel, root))


def _masked_path(dataset_name: str, rel: str) -> str:
    """Resolve the masked composite (``masked/{stem}.jpg``) used as the caption
    source for ``target="masked"`` — mirrors the single-image route."""
    from app.api._path_guard import validate_path_within
    from app.core.dataset_manager import dataset_manager as dm

    dataset = dm.get_dataset(dataset_name)
    if dataset is None:
        raise ValueError(f"Dataset '{dataset_name}' not found")
    root = Path(dataset.path)
    masked = root / "masked" / f"{Path(rel).stem}.jpg"
    if not masked.exists():
        raise FileNotFoundError(f"Masked image not found: masked/{Path(rel).stem}.jpg")
    return str(validate_path_within(masked, root))


def _write_caption(dataset_name: str, rel: str, text: str, target: str) -> None:
    """Persist *text* to disk and update the media-item metadata flag.

    Storage:
    - original: delegates to ``dataset_manager.save_caption`` — writes
      ``{dataset.path}/{stem}.txt``, flips ``has_caption``, recomputes
      ``caption_count`` and broadcasts ``entity.changed`` so the dataset card
      updates live during the batch (don't reimplement — that path drifts the
      count and skips the broadcast).
    - masked: ``{dataset.path}/masked/{stem}.txt`` → sets
      ``has_masked_caption = True`` (no save_caption equivalent for masked).
    """
    from app.core.dataset_manager import dataset_manager as dm

    stem = Path(rel).stem

    if target != "masked":
        dm.save_caption(dataset_name, f"{stem}.txt", text)
        return

    dataset = dm.get_dataset(dataset_name)
    if dataset is None:
        raise ValueError(f"Dataset '{dataset_name}' not found")
    masked_dir = Path(dataset.path) / "masked"
    masked_dir.mkdir(parents=True, exist_ok=True)
    (masked_dir / f"{stem}.txt").write_text(text, encoding="utf-8")

    lookup_key = rel.replace("\\", "/")
    if lookup_key in dataset.media_metadata:
        dataset.media_metadata[lookup_key]["has_masked_caption"] = True
        dm._persist_media_item(dataset, lookup_key)


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

                src = (_masked_path(dataset_name, rel) if target == "masked"
                       else _full_path(dataset_name, rel))
                caption = service.generate_caption(
                    image_path=src,
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
