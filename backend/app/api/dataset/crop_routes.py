"""Cropping routes — crop media and calculate crop targets."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from app.core.dataset_manager import dataset_manager
from app.core.logger import get_logger
from app.api.schemas.crop_schemas import CropRequest, CalcCropTargetRequest

router = APIRouter()
logger = get_logger(__name__)


@router.post("/datasets/{name}/crop")
async def crop_media(name: str, request: CropRequest):
    """Crop an image to the specified dimensions."""
    try:
        logger.info("cropping_media", dataset_name=name, path=request.path)
        await asyncio.to_thread(
            dataset_manager.crop_media,
            name,
            request.path,
            request.target_width,
            request.target_height,
            request.origin,
            request.crop_x,
            request.crop_y,
        )
        return {"status": "cropped", "file": request.path}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/datasets/{name}/calc-crop-targets")
async def calc_crop_targets(name: str, request: CalcCropTargetRequest):
    """Calculate crop target dimensions for a given image size and aspect ratio."""
    try:
        w, h = request.width, request.height
        ratio = request.aspect_ratio
        orientation = "landscape" if w >= h else "portrait"
        long_side = max(w, h)

        t_w, t_h = dataset_manager.calculate_target_dims(long_side, ratio, orientation)
        while t_w > w or t_h > h:
            long_side -= 32
            if long_side <= 0:
                t_w, t_h = w, h
                break
            t_w, t_h = dataset_manager.calculate_target_dims(long_side, ratio, orientation)

        return {"target_width": t_w, "target_height": t_h}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
