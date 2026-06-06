"""Masking API — generate, preview, apply, and delete image masks."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api._path_guard import safe_remove
from app.api.schemas.common_schemas import TaskEnqueuedResponse
from app.core.dataset_manager import dataset_manager
from app.core.logger import get_logger
from app.core.masking.masking_service import MaskingService
from app.core.masking.mask_generate_batch import run_mask_generate_batch
from app.core.masking.mask_apply_batch import run_mask_apply_batch
from app.core.tasks.task_manager import task_manager

router = APIRouter()
logger = get_logger(__name__)
masking_service = MaskingService.get_instance()


class MaskGenerationRequest(BaseModel):
    """Request body for mask generation."""

    dataset_name: str
    image_rel_path: str
    model_id: str
    params: dict[str, Any] = {}


class MaskGenerationResponse(BaseModel):
    """Response body for mask generation."""

    mask_path: str
    message: str


@router.post("/datasets/{name}/masking/generate", response_model=MaskGenerationResponse)
async def generate_mask(name: str, request: MaskGenerationRequest):
    """Generate a segmentation mask for a single image."""
    dataset = await asyncio.to_thread(dataset_manager.get_dataset, name)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")

    dataset_root = Path(dataset.path)
    image_full_path = dataset_root / request.image_rel_path
    if not image_full_path.exists():
        raise HTTPException(status_code=404, detail=f"Image file not found: {request.image_rel_path}")

    masks_dir = dataset_root / "masks"
    await asyncio.to_thread(masks_dir.mkdir, parents=True, exist_ok=True)

    logger.info("generating_mask", dataset=name, image=request.image_rel_path, model=request.model_id)

    try:
        mask_img = await asyncio.to_thread(
            masking_service.generate_mask, str(image_full_path), request.model_id, request.params,
        )
    except (OSError, RuntimeError, ValueError) as e:
        logger.error("mask_generation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Mask generation failed: {e}")

    if mask_img is None:
        raise HTTPException(status_code=500, detail="Mask generation returned no result")

    # Save mask
    original_stem = Path(request.image_rel_path).stem
    mask_filename = f"{original_stem}.png"
    mask_full_path = masks_dir / mask_filename

    await asyncio.to_thread(mask_img.save, str(mask_full_path))

    # Targeted metadata update (replaces full scan_dataset)
    lookup_key = request.image_rel_path.replace("\\", "/")
    if lookup_key in dataset.media_metadata:
        dataset.media_metadata[lookup_key]["has_mask"] = True
        await dataset_manager._persist_media_item_async(dataset, request.image_rel_path)

    return MaskGenerationResponse(
        mask_path=f"masks/{mask_filename}",
        message="Mask generated successfully",
    )


class ApplyMaskRequest(BaseModel):
    """Request body for applying a mask to an image."""

    image_rel_path: str
    opacity: float = 0.0


class ApplyMaskResponse(BaseModel):
    """Response body for applying a mask to an image."""

    status: str = "success"
    message: str
    output_path: str


@router.post("/datasets/{name}/masking/apply", response_model=ApplyMaskResponse)
async def apply_mask(name: str, request: ApplyMaskRequest):
    """Composite an image with its mask and save the result."""
    dataset = await asyncio.to_thread(dataset_manager.get_dataset, name)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")

    dataset_root = Path(dataset.path)
    image_full_path = dataset_root / request.image_rel_path
    original_stem = Path(request.image_rel_path).stem
    mask_full_path = dataset_root / "masks" / f"{original_stem}.png"

    if not mask_full_path.exists():
        raise HTTPException(status_code=404, detail="Mask for this image does not exist yet")

    masked_dir = dataset_root / "masked"
    await asyncio.to_thread(masked_dir.mkdir, parents=True, exist_ok=True)

    output_filename = f"{original_stem}.jpg"
    output_full_path = masked_dir / output_filename

    logger.info("applying_mask", dataset=name, image=request.image_rel_path, opacity=request.opacity)
    await asyncio.to_thread(
        masking_service.combine_mask, str(image_full_path), str(mask_full_path), str(output_full_path), request.opacity,
    )

    # Targeted metadata update (replaces full scan_dataset)
    lookup_key = request.image_rel_path.replace("\\", "/")
    if lookup_key in dataset.media_metadata:
        dataset.media_metadata[lookup_key]["has_masked"] = True
        await dataset_manager._persist_media_item_async(dataset, request.image_rel_path)

    return {
        "status": "success",
        "message": f"Masked image saved as {output_filename}",
        "output_path": output_filename,
    }


@router.get("/datasets/{name}/masking/preview")
async def preview_mask(name: str, image_rel_path: str, opacity: float = 0.5):
    """Generate an in-memory mask preview and stream it as PNG."""
    dataset = await asyncio.to_thread(dataset_manager.get_dataset, name)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")

    dataset_root = Path(dataset.path)
    image_full_path = dataset_root / image_rel_path
    original_stem = Path(image_rel_path).stem
    mask_full_path = dataset_root / "masks" / f"{original_stem}.png"

    if not mask_full_path.exists():
        raise HTTPException(status_code=404, detail="Mask not found")

    logger.debug("generating_mask_preview", image=image_rel_path)
    preview_img = await asyncio.to_thread(
        masking_service.generate_preview, str(image_full_path), str(mask_full_path), opacity,
    )

    buf = io.BytesIO()
    await asyncio.to_thread(preview_img.save, buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


class DeleteMaskResponse(BaseModel):
    """Response body for mask deletion."""

    status: str = "deleted"
    message: str = "Mask deleted successfully"


@router.delete("/datasets/{name}/masking/delete", response_model=DeleteMaskResponse)
async def delete_mask(name: str, image_rel_path: str):
    """Delete the mask file associated with an image."""
    dataset = await asyncio.to_thread(dataset_manager.get_dataset, name)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")

    original_stem = Path(image_rel_path).stem
    dataset_root = Path(dataset.path)
    mask_full_path = dataset_root / "masks" / f"{original_stem}.png"

    if not mask_full_path.exists():
        raise HTTPException(status_code=404, detail="Mask not found")

    logger.info("deleting_mask", dataset=name, mask=f"{original_stem}.png")
    await asyncio.to_thread(safe_remove, mask_full_path)

    # Also clean up masked image + masked caption (derived from this mask)
    for masked_ext in (".jpg", ".txt"):
        masked_path = dataset_root / "masked" / f"{original_stem}{masked_ext}"
        await asyncio.to_thread(safe_remove, masked_path)

    # Targeted metadata update (replaces full scan_dataset)
    lookup_key = image_rel_path.replace("\\", "/")
    if lookup_key in dataset.media_metadata:
        dataset.media_metadata[lookup_key]["has_mask"] = False
        dataset.media_metadata[lookup_key]["has_masked"] = False
        dataset.media_metadata[lookup_key]["has_masked_caption"] = False
        dataset.media_metadata[lookup_key].pop("mask_info", None)
        await dataset_manager._persist_media_item_async(dataset, image_rel_path)

    return {"status": "deleted", "message": "Mask deleted successfully"}


class MaskGenerateBatchRequest(BaseModel):
    """Request body for batch mask generation."""

    image_rel_paths: list[str]
    model_id: str
    params: dict[str, Any] = {}


@router.post("/datasets/{name}/masking/generate/batch", response_model=TaskEnqueuedResponse)
async def generate_masks_batch(name: str, request: MaskGenerateBatchRequest):
    """Start a backend-owned mask-generation task (queued on the GPU lane)."""
    dataset = await asyncio.to_thread(dataset_manager.get_dataset, name)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")

    task = task_manager.create(
        type="mask_generate_batch", title=f"Masks · {name}",
        total=len(request.image_rel_paths), dataset_name=name,
    )
    task_manager.enqueue(
        task.id,
        lambda tid: run_mask_generate_batch(
            tid, dataset_name=name, image_rel_paths=request.image_rel_paths,
            model_id=request.model_id, params=request.params,
        ),
        lane="gpu",
    )
    return {"task_id": task.id}


@router.post("/datasets/{name}/masking/apply/batch", response_model=TaskEnqueuedResponse)
async def apply_masks_batch(name: str, opacity: float = 0.0, overwrite: bool = False):
    """Start a backend-owned mask-apply task (queued on the GPU lane)."""
    dataset = await asyncio.to_thread(dataset_manager.get_dataset, name)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")

    def _count_masks() -> int:
        masks_dir = Path(dataset.path) / "masks"
        if not masks_dir.is_dir():
            return 0
        return sum(1 for f in masks_dir.iterdir() if f.suffix.lower() == ".png")

    total = await asyncio.to_thread(_count_masks)

    task = task_manager.create(
        type="mask_apply_batch", title=f"Apply masks · {name}",
        total=total, dataset_name=name,
    )
    task_manager.enqueue(
        task.id,
        lambda tid: run_mask_apply_batch(
            tid, dataset_name=name, opacity=opacity, overwrite=overwrite,
        ),
        lane="gpu",
    )
    return {"task_id": task.id}
