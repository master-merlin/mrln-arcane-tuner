"""Background batch pipeline-render worker (Mass-Edit: apply one overlay recipe
to many targets).

The genuinely-shared image work is ``execute_pipeline``; the surrounding overlay
bookkeeping (save PNG, update overlays.json, persist metadata, broadcast) is
mirrored from the single-image ``render_pipeline`` route synchronously — the live
route is left untouched (the per-image edit workspace depends on it). Same
approach ``mask_generate_batch`` took with the single-image generate route.

Module-level seams (monkeypatchable in tests):
  _render_one(dataset_name, image_path, blocks, tile_size, tile_pad, replace_recipe)
  run_pipeline_batch(...) → the worker function

GPU VRAM is freed inside the op handlers (cleanup_vram in pipeline.py), so there
is no separate unload step — identical to the single route.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from app.core.logger import get_logger
from app.core.tasks.task_manager import task_manager

logger = get_logger(__name__)


# ── Per-image render (mirrors the single-image render_pipeline route) ───────


def _render_one(
    dataset_name: str,
    image_path: str,
    blocks: list,
    tile_size: int,
    tile_pad: int,
    replace_recipe: bool,
) -> None:
    """Render one image's overlay synchronously, mirroring the single route."""
    from PIL import Image

    from app.api.dataset.overlay_routes import (
        _compute_file_hash,
        _overlays_dir,
        _read_overlays_json,
        _write_overlays_json,
    )
    from app.core.dataset_manager import dataset_manager as dm
    from app.core.image_processing.pipeline import PipelineBlock, execute_pipeline

    dataset = dm.get_dataset(dataset_name)
    if dataset is None:
        raise ValueError(f"Dataset '{dataset_name}' not found")
    dataset_root = Path(dataset.path)
    # Containment lives HERE, co-located with the IO, not only in the callers.
    # `image_path` is client-supplied; both current callers (render_pipeline_batch
    # / render_pipeline_task) already validate pre-enqueue, but a third caller —
    # a resume, a retry, a replay from a persisted task record — would otherwise
    # silently reopen an arbitrary-file-read primitive with no test failing.
    # Mirrors mask_generate_batch._full_path / caption_batch._full_path.
    from app.api._path_guard import validate_path_within

    img_path = validate_path_within(dataset_root / image_path, dataset_root)
    stem = Path(image_path).stem

    existing_overlay = _overlays_dir(dataset_root) / f"{stem}.png"
    source_path = (
        img_path
        if replace_recipe
        else (existing_overlay if existing_overlay.exists() else img_path)
    )

    with Image.open(source_path) as img:
        img_rgb = img.convert("RGB")

    dataclass_blocks = []
    for b in blocks:
        params = dict(b["params"])
        if b["type"] in ("denoise", "face_restore", "deartifact", "dehaze", "upscale"):
            params.setdefault("tile_size", tile_size)
            params.setdefault("tile_pad", tile_pad)
        dataclass_blocks.append(
            PipelineBlock(type=b["type"], enabled=b["enabled"], params=params)
        )

    result = execute_pipeline(img_rgb, dataclass_blocks)
    overlays_dir = _overlays_dir(dataset_root)
    overlay_path = overlays_dir / f"{stem}.png"
    result.save(str(overlay_path), quality=95)
    dimensions = result.size

    overlays_data = _read_overlays_json(dataset_root)
    if replace_recipe:
        merged_ops = [b for b in blocks if b["enabled"]]
    else:
        existing_entry = overlays_data.get(image_path, {})
        existing_ops = existing_entry.get("operations", [])
        ops_by_type = {op["type"]: op for op in existing_ops}
        for b in blocks:
            if b["enabled"]:
                ops_by_type[b["type"]] = b
        merged_ops = list(ops_by_type.values())
    overlays_data[image_path] = {
        "overlay_file": f"overlays/{stem}.png",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "operations": merged_ops,
    }
    _write_overlays_json(dataset_root, overlays_data)

    overlay_hash = _compute_file_hash(overlay_path)
    lookup_key = image_path.replace("\\", "/")
    if lookup_key in dataset.media_metadata:
        meta = dataset.media_metadata[lookup_key]
        meta["has_overlay"] = True
        meta["overlay_hash"] = overlay_hash
        meta["overlay_score_stale"] = True
        meta["overlay_dimensions"] = list(dimensions)
        dm._persist_media_item(dataset, lookup_key)

    _emit_overlay_updated(dataset_name, image_path, dimensions, overlay_hash, merged_ops)


def _emit_overlay_updated(
    dataset_name: str,
    image_path: str,
    dimensions: tuple,
    overlay_hash: str,
    operations: list,
) -> None:
    """Broadcast entity.changed (overlay/updated) cross-thread (no-op pre-loop)."""
    loop = task_manager._loop
    if loop is None:
        return
    from app.api.dataset.overlay_routes import _overlay_id
    from app.core.events import emit_entity_change, event_manager

    overlay_id = _overlay_id(dataset_name, image_path)
    stem = Path(image_path).stem
    payload = {
        "id": overlay_id,
        "dataset_name": dataset_name,
        "media_file": image_path,
        "overlay_file": f"overlays/{stem}.png",
        "dimensions": list(dimensions),
        "hash": overlay_hash,
        "operations": operations,
    }
    asyncio.run_coroutine_threadsafe(
        emit_entity_change(
            event_manager.broadcast,
            entity="overlay",
            op="updated",
            id=overlay_id,
            payload=payload,
        ),
        loop,
    )


# ── Worker ────────────────────────────────────────────────────────────────


def run_pipeline_batch(
    task_id: str,
    *,
    dataset_name: str,
    image_paths: list,
    blocks: list,
    tile_size: int,
    tile_pad: int,
    replace_recipe: bool,
) -> None:
    """Synchronous worker — GPU lane thread. Renders one overlay per target,
    isolating per-item failures and checking cancellation per item."""
    ok = 0
    failed = 0
    cancelled = False

    try:
        for i, path in enumerate(image_paths):
            if task_manager.is_cancelled(task_id):
                cancelled = True
                break
            try:
                _render_one(dataset_name, path, blocks, tile_size, tile_pad, replace_recipe)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.warning(
                    "pipeline_item_failed",
                    task_id=task_id,
                    rel_path=path,
                    error=str(exc),
                )
            task_manager.update(task_id, current=i + 1, item=path, ok=ok, failed=failed)

    except Exception as exc:  # unrecoverable setup error
        task_manager.fail(task_id, str(exc))
        return

    if cancelled:
        task_manager.finish_cancelled(task_id)
    else:
        task_manager.complete(task_id)
