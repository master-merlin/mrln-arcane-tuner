"""Overlay routes — non-destructive pipeline rendering, overlay CRUD, model listing."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.api._path_guard import safe_remove
from app.core.dataset_manager import dataset_manager
from app.core.events import emit_entity_change, event_manager
from app.core.logger import get_logger
from app.api.schemas.overlay_schemas import (
    ModelDownloadRequest,
    OverlayCommitRequest,
    RenderPipelineRequest,
    RestoreModelListRequest,
)
from app.core import model_registry

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
    dataset = dataset_manager.get_dataset(name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
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


@router.post("/datasets/{name}/render-pipeline")
async def render_pipeline(name: str, request: RenderPipelineRequest):
    """Execute the full pipeline on an image and save the overlay."""
    dataset, dataset_root = await asyncio.to_thread(_resolve_dataset, name)
    img_path = dataset_root / request.image_path
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    def _render():
        from PIL import Image

        from app.core.image_processing.pipeline import PipelineBlock, execute_pipeline

        # Chain operations: if an overlay already exists, use it as the
        # starting point so sequential operations build on each other
        # (e.g. denoise → upscale keeps the denoised result).
        stem = Path(request.image_path).stem
        existing_overlay = _overlays_dir(dataset_root) / f"{stem}.png"
        source_path = existing_overlay if existing_overlay.exists() else img_path

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
        dataset.media_metadata[lookup_key]["has_overlay"] = True
        dataset.media_metadata[lookup_key]["overlay_hash"] = overlay_hash
        dataset.media_metadata[lookup_key]["overlay_score_stale"] = True
        dataset.media_metadata[lookup_key]["overlay_dimensions"] = list(dimensions)
        await dataset_manager._persist_media_item_async(dataset, request.image_path)

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


@router.get("/datasets/{name}/overlay-recipe/{image_path:path}")
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


@router.delete("/datasets/{name}/overlay/{image_path:path}")
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
        dataset.media_metadata[lookup_key].pop("has_overlay", None)
        dataset.media_metadata[lookup_key].pop("overlay_hash", None)
        dataset.media_metadata[lookup_key].pop("overlay_score_stale", None)
        dataset.media_metadata[lookup_key].pop("overlay_dimensions", None)
        await dataset_manager._persist_media_item_async(dataset, image_path)

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


@router.post("/datasets/{name}/overlay/commit")
async def commit_overlay(name: str, request: OverlayCommitRequest):
    """Flatten the overlay into the original file (destructive)."""
    dataset, dataset_root = await asyncio.to_thread(_resolve_dataset, name)
    stem = Path(request.image_path).stem
    overlay_path = dataset_root / "overlays" / f"{stem}.png"
    img_path = dataset_root / request.image_path

    if not overlay_path.exists():
        raise HTTPException(status_code=404, detail="No overlay to commit")

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
        # Carry over overlay dimensions as the new original dimensions
        overlay_dims = dataset.media_metadata[lookup_key].pop("overlay_dimensions", None)
        if overlay_dims:
            dataset.media_metadata[lookup_key]["width"] = overlay_dims[0]
            dataset.media_metadata[lookup_key]["height"] = overlay_dims[1]
        # Clear overlay fields
        dataset.media_metadata[lookup_key].pop("has_overlay", None)
        dataset.media_metadata[lookup_key].pop("overlay_hash", None)
        dataset.media_metadata[lookup_key].pop("overlay_score_stale", None)
        # Recalculate size (paired exists+stat in one thread hop)
        def _size_if_exists(p: Path) -> int | None:
            return p.stat().st_size if p.exists() else None

        new_size = await asyncio.to_thread(_size_if_exists, img_path)
        if new_size is not None:
            dataset.media_metadata[lookup_key]["size_bytes"] = new_size
        # Invalidate masks (dimensions may have changed)
        dataset.media_metadata[lookup_key]["has_mask"] = False
        dataset.media_metadata[lookup_key]["has_masked"] = False
        dataset.media_metadata[lookup_key]["has_masked_caption"] = False
        dataset.media_metadata[lookup_key].pop("mask_info", None)
        await dataset_manager._persist_media_item_async(dataset, request.image_path)

    # Bump version
    await asyncio.to_thread(dataset_manager.bump_dataset_version, name, "patch")

    logger.info(f"Overlay committed for {request.image_path} in dataset '{name}'")

    # Commit flattens the overlay into the original and removes the
    # overlay file, so from the OverlayStore's perspective the overlay
    # is gone — emit `deleted`. The underlying media item's mutations
    # (new size/dims, mask invalidation) are broadcast separately by
    # _persist_media_item_async above.
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


@router.post("/restore/list-models")
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


@router.get("/models/registry/{category}")
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


@router.post("/models/download")
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
