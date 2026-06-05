"""Caption generation API — single-image captioning and model lifecycle."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api._path_guard import validate_path_within
from app.core.captioning.caption_batch import run_caption_batch
from app.core.captioning.caption_service import CaptionService
from app.core.logger import get_logger
from app.core.tasks.task_manager import task_manager

router = APIRouter()
logger = get_logger(__name__)


class GenerateCaptionRequest(BaseModel):
    """Request body for single-image caption generation."""

    dataset_name: str
    image_rel_path: str
    model_id: str
    params: dict[str, Any]
    system_prompt: str | None = None
    target: str = "original"  # "original" or "masked"


@router.post("/generate")
async def generate_caption_api(request: GenerateCaptionRequest):
    """Generate a caption for a single image using the specified model."""
    logger.info(
        "caption_request_received",
        dataset=request.dataset_name,
        image=request.image_rel_path,
        model=request.model_id,
    )

    service = CaptionService.get_instance()

    from app.core.dataset_manager import dataset_manager as manager

    dataset = await asyncio.to_thread(manager.get_dataset, request.dataset_name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_root = Path(dataset.path)
    full_path = validate_path_within(
        dataset_root / request.image_rel_path, dataset_root,
    )

    # When target is "masked", remap to masked/{stem}.jpg
    if request.target == "masked":
        stem = Path(request.image_rel_path).stem
        masked_path = dataset_root / "masked" / f"{stem}.jpg"
        if not masked_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Masked image not found: masked/{stem}.jpg",
            )
        full_path = masked_path

    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found at {full_path}")

    try:
        params = request.params.copy()
        if request.system_prompt:
            params["system_prompt"] = request.system_prompt

        caption = await asyncio.to_thread(
            service.generate_caption,
            image_path=str(full_path),
            model_id=request.model_id,
            params=params,
        )

        # When target is "masked", auto-save caption alongside masked image
        if request.target == "masked":
            stem = Path(request.image_rel_path).stem
            masked_dir = dataset_root / "masked"
            await asyncio.to_thread(masked_dir.mkdir, parents=True, exist_ok=True)
            caption_path = masked_dir / f"{stem}.txt"

            def _write_caption():
                caption_path.write_text(caption, encoding="utf-8")

            await asyncio.to_thread(_write_caption)

            # Update has_masked_caption in DB
            lookup_key = request.image_rel_path.replace("\\", "/")
            if lookup_key in dataset.media_metadata:
                dataset.media_metadata[lookup_key]["has_masked_caption"] = True
                await manager._persist_media_item_async(dataset, request.image_rel_path)

        return {"caption": caption}
    except (OSError, RuntimeError, ValueError) as e:
        logger.error("caption_generation_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/unload")
async def unload_models_api():
    """Unload all caption models and free VRAM."""
    try:
        logger.info("unloading_caption_models")
        service = CaptionService.get_instance()
        await asyncio.to_thread(service.unload_models)
        return {"status": "success", "message": "All models unloaded and VRAM cleared."}
    except (OSError, RuntimeError) as e:
        logger.error("unload_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


class BatchCaptionRequest(BaseModel):
    """Request body for batch caption generation."""

    dataset_name: str
    image_rel_paths: list[str]
    model_id: str
    params: dict[str, Any]
    system_prompt: str | None = None
    target: str = "original"


@router.post("/batch")
async def batch_caption_api(request: BatchCaptionRequest):
    """Start a backend-owned captioning task. Always creates the task (queued if
    the GPU lane is busy) and returns its id immediately."""
    title = f"Captioning · {request.dataset_name}"
    task = task_manager.create(
        type="caption_batch", title=title,
        total=len(request.image_rel_paths), dataset_name=request.dataset_name,
        target=request.target,
    )
    task_manager.enqueue(
        task.id,
        lambda tid: run_caption_batch(
            tid,
            dataset_name=request.dataset_name,
            image_rel_paths=request.image_rel_paths,
            model_id=request.model_id,
            params=request.params,
            system_prompt=request.system_prompt,
            target=request.target,
        ),
        lane="gpu",
    )
    return {"task_id": task.id}
