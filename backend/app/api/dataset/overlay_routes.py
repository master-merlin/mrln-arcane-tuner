"""Overlay routes — non-destructive pipeline rendering, overlay CRUD, model listing."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.api._deps import dataset_or_404
from app.api._path_guard import reject_audio_op, safe_remove, validate_path_within
from app.core.dataset_manager import dataset_manager
from app.core.events import emit_entity_change, event_manager
from app.core.logger import get_logger
from app.api.schemas.overlay_schemas import (
    ModelDownloadRequest,
    ModelDownloadResponse,
    ModelRegistryResponse,
    OverlayActionResponse,
    OverlayCommitRequest,
    OverlayRecipeResponse,
    RenderPipelineBatchRequest,
    RenderPipelineRequest,
    RenderPipelineResponse,
    RestoreModelListRequest,
    RestoreModelListResponse,
)
from app.api.schemas.common_schemas import TaskEnqueuedResponse
from app.core import model_registry
from app.core.image_processing.pipeline_batch import run_pipeline_batch
from app.core.tasks.task_manager import task_manager

router = APIRouter()
logger = get_logger(__name__)

# backend/ root — parents[3] from backend/app/api/dataset/overlay_routes.py
_BACKEND_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_RESTORE_FOLDER = _BACKEND_ROOT / "models" / "restore"


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------


def _overlays_dir(dataset_root: Path) -> Path:
    """Return the overlays directory, creating it if needed."""
    d = dataset_root / "overlays"
    d.mkdir(exist_ok=True)
    return d


def _overlays_json_path(dataset_root: Path) -> Path:
    return dataset_root / "overlays.json"


def _read_overlays_json(dataset_root: Path) -> dict:
    path = _overlays_json_path(dataset_root)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _write_overlays_json(dataset_root: Path, data: dict) -> None:
    path = _overlays_json_path(dataset_root)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_dataset(name: str) -> tuple:
    """Resolve dataset and return (dataset, dataset_root). Raises HTTPException if not found."""
    dataset = dataset_or_404(dataset_manager.get_dataset(name))
    return dataset, Path(dataset.path)


def _overlay_id(dataset_name: str, image_path: str) -> str:
    """Composite id used by the frontend OverlayStore.

    Mirrors the media_item key shape so the overlay can be associated
    with its underlying media file. Forward-slash normalized.
    """
    return f"{dataset_name}/{image_path.replace(chr(92), '/')}"


# ---------------------------------------------------------------------------
# Render Pipeline
# ---------------------------------------------------------------------------


@router.post("/datasets/{name}/render-pipeline", response_model=RenderPipelineResponse)
async def render_pipeline(name: str, request: RenderPipelineRequest):
    """Execute the full pipeline on an image and save the overlay."""
    dataset, dataset_root = await asyncio.to_thread(_resolve_dataset, name)
    img_path = validate_path_within(dataset_root / request.image_path, dataset_root)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    reject_audio_op(request.image_path, "Render pipeline")

    def _render():
        from PIL import Image

        from app.core.image_processing.pipeline import PipelineBlock, execute_pipeline

        # Chain operations: if an overlay already exists, use it as the
        # starting point so sequential operations build on each other
        # (e.g. denoise → upscale keeps the denoised result).
        stem = Path(request.image_path).stem
        existing_overlay = _overlays_dir(dataset_root) / f"{stem}.png"
        # replace_recipe=true (Save) is authoritative: render must source
        # from the original. replace_recipe=false (restore/upscale chaining)
        # keeps its legacy chain-from-overlay behavior.
        source_path = (
            img_path
            if request.replace_recipe
            else (existing_overlay if existing_overlay.exists() else img_path)
        )

        with Image.open(source_path) as img:
            img_rgb = img.convert("RGB")

        # Convert schema blocks to dataclass blocks, injecting tile config
        blocks = []
        for b in request.blocks:
            params = dict(b.params)
            # Inject tile config for GPU ops
            if b.type in ("denoise", "face_restore", "deartifact", "dehaze", "upscale"):
                params.setdefault("tile_size", request.tile_size)
                params.setdefault("tile_pad", request.tile_pad)
            blocks.append(PipelineBlock(type=b.type, enabled=b.enabled, params=params))

        result = execute_pipeline(img_rgb, blocks)

        # Save overlay
        overlays_dir = _overlays_dir(dataset_root)
        stem = Path(request.image_path).stem
        overlay_path = overlays_dir / f"{stem}.png"
        result.save(str(overlay_path), quality=95)

        return overlay_path, result.size

    try:
        overlay_path, dimensions = await asyncio.to_thread(_render)
    except (OSError, RuntimeError, MemoryError) as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Update overlays.json recipe — MERGE or REPLACE based on flag
    overlays_data = await asyncio.to_thread(_read_overlays_json, dataset_root)

    if request.replace_recipe:
        # Full render: replace entire recipe with exactly the blocks sent
        merged_ops = [b.model_dump() for b in request.blocks if b.enabled]
    else:
        # Individual op (restore/upscale): merge into existing recipe
        existing_entry = overlays_data.get(request.image_path, {})
        existing_ops = existing_entry.get("operations", [])
        ops_by_type = {op["type"]: op for op in existing_ops}
        for b in request.blocks:
            if b.enabled:
                ops_by_type[b.type] = b.model_dump()
        merged_ops = list(ops_by_type.values())

    overlays_data[request.image_path] = {
        "overlay_file": f"overlays/{Path(request.image_path).stem}.png",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "operations": merged_ops,
    }
    await asyncio.to_thread(_write_overlays_json, dataset_root, overlays_data)

    # Update metadata
    overlay_hash = await asyncio.to_thread(_compute_file_hash, overlay_path)
    lookup_key = request.image_path.replace("\\", "/")
    if lookup_key in dataset.media_metadata:
        await dataset_manager.update_media_flags_async(
            name, request.image_path,
            has_overlay=True,
            overlay_hash=overlay_hash,
            overlay_score_stale=True,
            overlay_dimensions=list(dimensions),
        )

    logger.info(f"Overlay saved for {request.image_path} in dataset '{name}'")

    # Broadcast for the frontend OverlayStore. Upsert semantics: a render
    # either creates a brand-new overlay or rewrites an existing one — we
    # emit `updated` for both rather than tracking the pre-state, which
    # matches how registry/settings stores handle PUT-style endpoints.
    overlay_id = _overlay_id(name, request.image_path)
    await emit_entity_change(
        event_manager.broadcast,
        entity="overlay",
        op="updated",
        id=overlay_id,
        payload={
            "id": overlay_id,
            "dataset_name": name,
            "media_file": request.image_path,
            "overlay_file": f"overlays/{Path(request.image_path).stem}.png",
            "dimensions": list(dimensions),
            "hash": overlay_hash,
            "operations": merged_ops,
        },
    )

    return {
        "status": "overlay_saved",
        "file": request.image_path,
        "overlay": f"overlays/{Path(request.image_path).stem}.png",
        "dimensions": list(dimensions),
        "hash": overlay_hash,
    }


@router.post("/datasets/{name}/render-pipeline/batch", response_model=TaskEnqueuedResponse)
async def render_pipeline_batch(name: str, request: RenderPipelineBatchRequest):
    """Start a backend-owned task that applies one pipeline recipe to many
    images (mass-edit). Returns the task id immediately; monitored via TaskStore."""
    _, dataset_root = await asyncio.to_thread(_resolve_dataset, name)
    # This route hands every path in image_paths straight to a background
    # task (run_pipeline_batch -> _render_one), which does an unguarded
    # READ + overlay WRITE per item. No response has started streaming yet,
    # so validate containment for the WHOLE batch up front and reject the
    # entire request (fail-closed) before anything is enqueued — a partial
    # accept would still let one escaping entry through.
    for image_path in request.image_paths:
        validate_path_within(dataset_root / image_path, dataset_root)

    blocks = [b.model_dump() for b in request.blocks]
    task = task_manager.create(
        type="adjust_batch", title=f"Adjustments · {name}",
        total=len(request.image_paths), dataset_name=name,
    )
    task_manager.enqueue(
        task.id,
        lambda tid: run_pipeline_batch(
            tid, dataset_name=name, image_paths=request.image_paths, blocks=blocks,
            tile_size=request.tile_size, tile_pad=request.tile_pad,
            replace_recipe=request.replace_recipe,
        ),
        lane="gpu",
    )
    return {"task_id": task.id}


@router.post("/datasets/{name}/render-pipeline/task", response_model=TaskEnqueuedResponse)
async def render_pipeline_task(name: str, request: RenderPipelineRequest):
    """Run a SINGLE-image pipeline render as a gpu-lane background task (used by
    the edit workspace when the pipeline contains a GPU op — denoise/upscale).
    Returns the task id immediately; the overlay updates via entity.changed."""
    _, dataset_root = await asyncio.to_thread(_resolve_dataset, name)
    # Same RenderPipelineRequest schema as the synchronous render_pipeline
    # route above — validate containment before enqueueing the background
    # task (run_pipeline_batch -> _render_one), which has no guard of its own.
    validate_path_within(dataset_root / request.image_path, dataset_root)

    blocks = [b.model_dump() for b in request.blocks]
    task = task_manager.create(
        type="render_task", title=f"Render · {Path(request.image_path).stem}",
        total=1, dataset_name=name,
    )
    task_manager.enqueue(
        task.id,
        lambda tid: run_pipeline_batch(
            tid, dataset_name=name, image_paths=[request.image_path], blocks=blocks,
            tile_size=request.tile_size, tile_pad=request.tile_pad,
            replace_recipe=request.replace_recipe,
        ),
        lane="gpu",
    )
    return {"task_id": task.id}


# ---------------------------------------------------------------------------
# Serve overlay
# ---------------------------------------------------------------------------


@router.get("/datasets/{name}/overlay/{image_path:path}")
async def get_overlay(name: str, image_path: str):
    """Serve the overlay image for a given media file."""
    _, dataset_root = await asyncio.to_thread(_resolve_dataset, name)
    stem = Path(image_path).stem
    overlay_path = dataset_root / "overlays" / f"{stem}.png"
    if not overlay_path.exists():
        raise HTTPException(status_code=404, detail="No overlay exists for this image")
    return FileResponse(str(overlay_path), media_type="image/png")


# ---------------------------------------------------------------------------
# Get overlay recipe
# ---------------------------------------------------------------------------


@router.get(
    "/datasets/{name}/overlay-recipe/{image_path:path}",
    response_model=OverlayRecipeResponse,
)
async def get_overlay_recipe(name: str, image_path: str):
    """Return the pipeline recipe that produced an overlay."""
    _, dataset_root = await asyncio.to_thread(_resolve_dataset, name)
    overlays_data = await asyncio.to_thread(_read_overlays_json, dataset_root)
    recipe = overlays_data.get(image_path)
    if not recipe:
        raise HTTPException(status_code=404, detail="No overlay recipe found")
    return {"image_path": image_path, "recipe": recipe}


# ---------------------------------------------------------------------------
# Revert (delete overlay)
# ---------------------------------------------------------------------------


@router.delete(
    "/datasets/{name}/overlay/{image_path:path}",
    response_model=OverlayActionResponse,
)
async def delete_overlay(name: str, image_path: str):
    """Delete the overlay for an image, reverting to the original."""
    dataset, dataset_root = await asyncio.to_thread(_resolve_dataset, name)
    stem = Path(image_path).stem
    overlay_path = dataset_root / "overlays" / f"{stem}.png"

    if overlay_path.exists():
        await asyncio.to_thread(safe_remove, overlay_path)

    # Remove from overlays.json
    overlays_data = await asyncio.to_thread(_read_overlays_json, dataset_root)
    overlays_data.pop(image_path, None)
    await asyncio.to_thread(_write_overlays_json, dataset_root, overlays_data)

    # Clear overlay metadata
    lookup_key = image_path.replace("\\", "/")
    if lookup_key in dataset.media_metadata:
        remove = dataset_manager.REMOVE_FIELD
        await dataset_manager.update_media_flags_async(
            name, image_path,
            has_overlay=remove,
            overlay_hash=remove,
            overlay_score_stale=remove,
            overlay_dimensions=remove,
        )

    logger.info(f"Overlay reverted for {image_path} in dataset '{name}'")

    await emit_entity_change(
        event_manager.broadcast,
        entity="overlay",
        op="deleted",
        id=_overlay_id(name, image_path),
        payload=None,
    )

    return {"status": "reverted", "file": image_path}


# ---------------------------------------------------------------------------
# Commit (flatten overlay → original)
# ---------------------------------------------------------------------------


@router.post("/datasets/{name}/overlay/commit", response_model=OverlayActionResponse)
async def commit_overlay(name: str, request: OverlayCommitRequest):
    """Flatten the overlay — into the original (destructive) or a control slot."""
    dataset, dataset_root = await asyncio.to_thread(_resolve_dataset, name)
    stem = Path(request.image_path).stem
    overlay_path = dataset_root / "overlays" / f"{stem}.png"
    # WRITE primitive below (shutil.copy2 into img_path) — validate containment
    # before anything else runs.
    img_path = validate_path_within(dataset_root / request.image_path, dataset_root)

    if not overlay_path.exists():
        raise HTTPException(status_code=404, detail="No overlay to commit")

    # ── Save into a control slot (non-destructive pair production) ──────────
    if request.target != "original":
        from app.core.dataset.control_helpers import (
            CONTROL_SLOTS,
            prepare_control_slot_path,
        )

        slot_index = CONTROL_SLOTS.index(request.target) + 1

        def _save_to_slot() -> str:
            dest = prepare_control_slot_path(
                str(dataset_root), slot_index, stem, ".png",
            )
            # Copy the rendered overlay into the slot; the original (and its
            # overlay/recipe) are left untouched — no mask invalidation.
            shutil.copy2(str(overlay_path), dest)
            return f"{request.target}/{stem}.png"

        rel = await asyncio.to_thread(_save_to_slot)
        # Refresh control_info on the paired target (no full rescan); this
        # emits the media_item entity.changed event so the grid/pair UX updates.
        await asyncio.to_thread(
            dataset_manager.refresh_control_metadata, name, stem,
        )
        logger.info(
            "overlay_saved_to_control",
            dataset_name=name, image_path=request.image_path, slot=request.target,
        )
        return {"status": "saved_to_control", "file": rel}

    def _commit():
        # Overwrite original with overlay
        shutil.copy2(str(overlay_path), str(img_path))
        # Remove overlay file
        safe_remove(overlay_path)

    await asyncio.to_thread(_commit)

    # Source pixels were overwritten by _commit — refresh thumbnail.
    from app.core.dataset import thumbnails

    await asyncio.to_thread(
        thumbnails.invalidate_thumbnail, dataset.path, request.image_path,
    )
    await asyncio.to_thread(
        thumbnails.ensure_thumbnail, dataset.path, request.image_path,
    )

    # Remove from overlays.json
    overlays_data = await asyncio.to_thread(_read_overlays_json, dataset_root)
    overlays_data.pop(request.image_path, None)
    await asyncio.to_thread(_write_overlays_json, dataset_root, overlays_data)

    # Update metadata: overlay is now the original
    lookup_key = request.image_path.replace("\\", "/")
    if lookup_key in dataset.media_metadata:
        remove = dataset_manager.REMOVE_FIELD

        # Recalculate size (paired exists+stat in one thread hop)
        def _size_if_exists(p: Path) -> int | None:
            return p.stat().st_size if p.exists() else None

        new_size = await asyncio.to_thread(_size_if_exists, img_path)

        changes: dict[str, Any] = {
            # Clear overlay fields — the overlay is now the original.
            "overlay_dimensions": remove,
            "has_overlay": remove,
            "overlay_hash": remove,
            "overlay_score_stale": remove,
            # Invalidate masks (dimensions may have changed)
            "has_mask": False,
            "has_masked": False,
            "has_masked_caption": False,
            "mask_info": remove,
        }
        if new_size is not None:
            changes["size_bytes"] = new_size

        def _derive_dims_from_overlay(meta: dict[str, Any]) -> dict[str, Any]:
            # Carry over overlay dimensions as the new original dimensions.
            # Evaluated by update_media_flags AFTER it takes the mutation
            # lock — reading the live dict here (instead of before this
            # call) closes the race where a concurrent request (e.g.
            # another overlay render) changes overlay_dimensions between a
            # pre-lock read and this call's own lock acquisition, which
            # would otherwise let this call silently revert the concurrent
            # update with a stale width/height.
            dims = meta.get("overlay_dimensions")
            if not dims:
                return {}
            return {"width": dims[0], "height": dims[1]}

        await dataset_manager.update_media_flags_async(
            name,
            request.image_path,
            derive=_derive_dims_from_overlay,
            **changes,
        )

    # Bump version
    await asyncio.to_thread(dataset_manager.bump_dataset_version, name, "patch")

    logger.info(f"Overlay committed for {request.image_path} in dataset '{name}'")

    # Commit flattens the overlay into the original and removes the
    # overlay file, so from the OverlayStore's perspective the overlay
    # is gone — emit `deleted`. The underlying media item's mutations
    # (new size/dims, mask invalidation) are broadcast separately by
    # update_media_flags_async above.
    await emit_entity_change(
        event_manager.broadcast,
        entity="overlay",
        op="deleted",
        id=_overlay_id(name, request.image_path),
        payload=None,
    )

    return {"status": "committed", "file": request.image_path}


# ---------------------------------------------------------------------------
# Model listing for restoration
# ---------------------------------------------------------------------------


@router.post("/restore/list-models", response_model=RestoreModelListResponse)
async def list_restore_models(request: RestoreModelListRequest):
    """Scan a folder for restoration model files."""
    folder_str = request.folder.strip() if request.folder else ""
    if not folder_str:
        folder = _DEFAULT_RESTORE_FOLDER
    else:
        folder = Path(folder_str)

    if not folder.is_dir():
        # Return empty list instead of 404 — folder might not exist yet
        return {"models": [], "folder": str(folder)}

    model_exts = {".pth", ".safetensors", ".safetensor", ".pt", ".onnx", ".bin"}

    def _scan_models() -> list[dict]:
        items: list[dict] = []
        for f in folder.iterdir():
            if f.is_file() and f.suffix.lower() in model_exts:
                size_mb = f.stat().st_size / (1024 * 1024)
                items.append({
                    "name": f.name,
                    "path": str(f),
                    "size_mb": round(size_mb, 1),
                })
        return items

    models = await asyncio.to_thread(_scan_models)
    models.sort(key=lambda m: m["name"])
    return {"models": models, "folder": str(folder)}


# ---------------------------------------------------------------------------
# Model registry & download
# ---------------------------------------------------------------------------


def _default_folder_for_category(category: str) -> Path:
    """Return the default folder for a model category."""
    return _BACKEND_ROOT / "models" / category


@router.get("/models/registry/{category}", response_model=ModelRegistryResponse)
async def get_model_registry(category: str):
    """Return all known models for a category with download status."""
    if category not in ("restore", "upscale"):
        raise HTTPException(status_code=400, detail=f"Unknown category: {category}")

    folder = _default_folder_for_category(category)
    models = model_registry.get_download_status(category, folder)
    return {
        "category": category,
        "folder": str(folder),
        "models": models,
    }


@router.post("/models/download", response_model=ModelDownloadResponse)
async def download_model_route(request: ModelDownloadRequest):
    """Download a model from the curated registry to the target folder."""
    if request.target_folder.strip():
        folder = Path(request.target_folder)
    else:
        folder = _default_folder_for_category(request.category)

    try:
        path = await model_registry.download_model(
            request.category, request.filename, folder
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Model download failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")

    return {
        "status": "downloaded",
        "filename": request.filename,
        "path": str(path),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
    }
