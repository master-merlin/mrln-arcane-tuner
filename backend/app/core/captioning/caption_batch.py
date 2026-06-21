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

from app.core.captioning import caption_variants
from app.core.captioning.caption_service import CaptionService
from app.core.captioning.models.base import (
    VIDEO_MOTION_INSTRUCTION as VIDEO_CAPTION_PROMPT,
)
from app.core.logger import get_logger
from app.core.tasks.task import TaskStatus
from app.core.tasks.task_manager import task_manager

logger = get_logger(__name__)

# ``VIDEO_CAPTION_PROMPT`` is the default "Video (motion-aware)" prompt, used
# when a video-caption batch supplies no user prompt. It is re-exported from the
# model base so the on-device VLM, the API lane, and the batch worker all share
# one motion-aware instruction (single source of truth — there is no template
# seeder/repo for caption prompts, so a module constant is the right home).


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


def _control_paths(dataset_name: str, rel: str) -> list[str]:
    """Resolve a target's control ("before") image absolute paths for two-image
    edit captioning. Returns the stem-matched control slot files in slot order
    (control, control_2, control_3); empty when the target has no controls."""
    from app.core.dataset.control_helpers import detect_control_slots
    from app.core.dataset_manager import dataset_manager as dm

    dataset = dm.get_dataset(dataset_name)
    if dataset is None:
        return []
    root = Path(dataset.path)
    stem = Path(rel).stem
    slots = detect_control_slots(str(root), stem)
    return [
        str(root / info["rel_path"]) for info in slots.values() if info.get("rel_path")
    ]


def _video_meta(dataset_name: str, rel: str) -> dict:
    """Return a video-captioning param overlay for *rel*.

    Reads the media item's metadata and surfaces the fields the video caption
    path consumes: ``is_video`` plus the user trim bounds ``trim_start_s`` /
    ``trim_end_s``. Falls back to extension detection (``is_probeable_video``)
    so a video clip is routed correctly even if its metadata row is sparse.
    Returns ``{}`` for plain images.
    """
    from app.core.dataset.media_types import is_probeable_video
    from app.core.dataset_manager import dataset_manager as dm

    is_video = is_probeable_video(Path(rel).suffix)
    overlay: dict = {}
    dataset = dm.get_dataset(dataset_name)
    if dataset is not None:
        meta = dataset.media_metadata.get(rel.replace("\\", "/"), {})
        if meta.get("is_video"):
            is_video = True
        for key in ("trim_start_s", "trim_end_s"):
            if meta.get(key) is not None:
                overlay[key] = meta[key]
    if is_video:
        overlay["is_video"] = True
    return overlay


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


def _write_caption(
    dataset_name: str,
    rel: str,
    text: str,
    target: str,
    definition_id: str | None = None,
) -> None:
    """Persist *text* to disk and update the media-item metadata flag.

    Storage:
    - per-definition variant: when *definition_id* is given and ``target ==
      "original"``, the caption is written to
      ``{dataset.path}/captions/{definition_id}/{stem}.txt`` via
      ``caption_variants.write_variant`` instead of the general caption.
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

    if definition_id and target == "original":
        dataset = dm.get_dataset(dataset_name)
        if dataset is not None:
            caption_variants.write_variant(dataset.path, definition_id, stem, text)
        return

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


def _get_caption_format(definition_id: str | None):
    """Resolve the caption format for a definition (PlainFormat when None/unknown)."""
    from app.core.captioning.formats import (
        get_caption_format_for_definition,
        PlainFormat,
    )

    if not definition_id:
        return PlainFormat()
    return get_caption_format_for_definition(definition_id)


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
    definition_id: str | None = None,
    include_control: bool = False,
) -> None:
    """Synchronous worker — runs on the GPU lane thread.

    Iterates over *image_rel_paths*, calls the caption service for each image,
    persists the result, and updates task progress.  Honours cancellation
    before each image.  For local models, always calls
    ``CaptionService.unload_models()`` in the finally block to free VRAM;
    api-* models are stateless and must never trigger the global unload
    (it would rip a local model out from under the GPU lane).
    """
    ok = 0
    failed = 0
    cancelled = False
    consecutive_failures = 0

    if definition_id is None:
        definition_id = params.get("definition_id")

    caption_format = _get_caption_format(definition_id)

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

                # Video items: surface is_video + trim bounds so the service
                # samples frames and routes to model.generate_video().
                video_overlay = _video_meta(dataset_name, rel)
                is_video = bool(video_overlay.get("is_video"))
                if is_video:
                    call_params.update(video_overlay)
                    # Masks are out of scope for video — skip the masked variant
                    # with a per-item warning rather than failing the item.
                    if target == "masked":
                        logger.warning(
                            "caption_batch_skip_masked_video",
                            task_id=task_id,
                            rel_path=rel,
                        )
                        task_manager.update(
                            task_id, current=i + 1, item=rel, ok=ok, failed=failed
                        )
                        continue
                    # Motion-aware default prompt when the user supplied none.
                    if not call_params.get("system_prompt"):
                        call_params["system_prompt"] = VIDEO_CAPTION_PROMPT

                if caption_format.is_structured:
                    if not call_params.get("system_prompt"):
                        call_params["system_prompt"] = (
                            caption_format.build_generation_prompt(
                                call_params.get("caption_instructions")
                            )
                        )
                    call_params.update(caption_format.generation_overrides())
                    if model_id.startswith("api-") and caption_format.json_schema():
                        call_params["response_format"] = {"type": "json_object"}

                if model_id.startswith("api-"):
                    # Let the HTTP client's retry/backoff loop bail out as soon
                    # as the task is cancelled (instead of sleeping through the
                    # full backoff schedule).
                    call_params["_should_abort"] = lambda: task_manager.is_cancelled(
                        task_id
                    )

                src = (
                    _masked_path(dataset_name, rel)
                    if target == "masked"
                    else _full_path(dataset_name, rel)
                )
                extra = _control_paths(dataset_name, rel) if include_control else None
                caption = service.generate_caption(
                    image_path=src,
                    model_id=model_id,
                    params=call_params,
                    extra_image_paths=extra,
                )
                if caption_format.is_structured:
                    caption = caption_format.serialize(
                        caption_format.parse_and_normalize(caption)
                    )
                if definition_id:
                    _write_caption(dataset_name, rel, caption, target, definition_id)
                else:
                    _write_caption(dataset_name, rel, caption, target)
                ok += 1
                consecutive_failures = 0
                _emit_caption_written(
                    dataset_name=dataset_name,
                    media_file=rel,
                    caption=caption,
                    target=target,
                )
            except Exception as exc:  # noqa: BLE001
                failed += 1
                consecutive_failures += 1
                logger.warning(
                    "caption_batch_item_failed",
                    task_id=task_id,
                    rel_path=rel,
                    error=str(exc),
                )
                # API batches: a long run of consecutive failures means the
                # provider/key/model is broken — fail fast instead of burning
                # through the whole batch (each item retries with backoff).
                if model_id.startswith("api-") and consecutive_failures >= 5:
                    task_manager.update(
                        task_id, current=i + 1, item=rel, ok=ok, failed=failed
                    )
                    task_manager.fail(
                        task_id,
                        f"Aborted after {consecutive_failures} consecutive "
                        f"API failures (last: {exc})",
                    )
                    return

            task_manager.update(task_id, current=i + 1, item=rel, ok=ok, failed=failed)

    except Exception as exc:  # unrecoverable setup error
        task_manager.fail(task_id, str(exc))
        return
    finally:
        # api-* plugins hold no VRAM; skipping the global unload keeps a
        # background-lane API batch from unloading the GPU lane's model.
        if not model_id.startswith("api-"):
            CaptionService.unload_models()

    if cancelled:
        task_manager._finish(task_id, TaskStatus.CANCELLED)
    else:
        task_manager.complete(task_id)
