"""Caption generation API — single-image captioning and model lifecycle."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.captioning.caption_service import CaptionService
from app.core.logger import get_logger

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

    dataset = await asyncio.to_thread(manager.datasets.get, request.dataset_name)
    if not dataset:
        if request.dataset_name not in manager.datasets:
            raise HTTPException(status_code=404, detail="Dataset not found")
        dataset = manager.datasets[request.dataset_name]

    full_path = os.path.join(dataset.path, request.image_rel_path.replace("/", os.sep))

    # When target is "masked", remap to masked/{stem}.jpg
    if request.target == "masked":
        stem = os.path.splitext(os.path.basename(request.image_rel_path))[0]
        masked_path = os.path.join(dataset.path, "masked", f"{stem}.jpg")
        if not os.path.exists(masked_path):
            raise HTTPException(
                status_code=404,
                detail=f"Masked image not found: masked/{stem}.jpg",
            )
        full_path = masked_path

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"Image not found at {full_path}")

    try:
        params = request.params.copy()
        if request.system_prompt:
            params["system_prompt"] = request.system_prompt

        caption = await asyncio.to_thread(
            service.generate_caption,
            image_path=full_path,
            model_id=request.model_id,
            params=params,
        )

        # When target is "masked", auto-save caption alongside masked image
        if request.target == "masked":
            stem = os.path.splitext(os.path.basename(request.image_rel_path))[0]
            masked_dir = os.path.join(dataset.path, "masked")
            await asyncio.to_thread(os.makedirs, masked_dir, exist_ok=True)
            caption_path = os.path.join(masked_dir, f"{stem}.txt")
            await asyncio.to_thread(
                lambda: open(caption_path, "w", encoding="utf-8").write(caption)
            )

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
