"""Image quality scoring API — single/batch scoring and model lifecycle."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.dataset_manager import dataset_manager as manager
from app.core.db.repositories.media_item_repo import MediaItemRepository
from app.core.events import event_manager
from app.core.logger import get_logger
from app.core.scoring.scoring_service import ScoringService

router = APIRouter()
logger = get_logger(__name__)
media_repo = MediaItemRepository()


# ── Request / Response models ────────────────────────────────────────────

class ScoreImageRequest(BaseModel):
    """Request body for single-image scoring."""
    image_rel_path: str
    model_id: str = "hpsv2"
    hps_version: str = "v2.1"
    prompt: str | None = None


class ScoreBatchRequest(BaseModel):
    """Request body for batch dataset scoring."""
    model_id: str = "hpsv2"
    hps_version: str = "v2.1"
    use_captions: bool = True
    fallback_prompt: str = ""


# ── Routes ───────────────────────────────────────────────────────────────

@router.post("/datasets/{name}/score")
async def score_image_api(name: str, request: ScoreImageRequest):
    """Score a single image for quality."""
    dataset = await asyncio.to_thread(manager.datasets.get, name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    full_path = os.path.join(dataset.path, request.image_rel_path.replace("/", os.sep))
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Image not found")

    service = ScoringService.get_instance()
    params: dict[str, Any] = {"hps_version": request.hps_version}

    # Use provided prompt, or try to read paired caption
    if request.prompt is not None:
        params["prompt"] = request.prompt
    else:
        caption_path = os.path.splitext(full_path)[0] + ".txt"
        if os.path.exists(caption_path):
            with open(caption_path, "r", encoding="utf-8") as f:
                params["prompt"] = f.read().strip()
        else:
            params["prompt"] = ""

    try:
        score = await asyncio.to_thread(
            service.score_image, full_path, request.model_id, params
        )

        # Persist score to DB
        dataset_id = dataset.name
        await asyncio.to_thread(
            media_repo.update,
            dataset_id,
            request.image_rel_path,
            {"quality_score": score},
        )

        return {"score": score, "file": request.image_rel_path}
    except (OSError, RuntimeError, ValueError) as e:
        logger.error("scoring_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/{name}/score-batch")
async def score_batch_api(name: str, request: ScoreBatchRequest):
    """Score all images in a dataset for quality.

    Progress is streamed via WebSocket. Returns summary statistics.
    """
    dataset = await asyncio.to_thread(manager.datasets.get, name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Collect all image paths
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".tiff"}
    all_files = await asyncio.to_thread(os.listdir, dataset.path)
    image_files = [
        f for f in all_files
        if os.path.splitext(f)[1].lower() in image_exts
    ]

    if not image_files:
        return {"scored": 0, "message": "No images found in dataset"}

    image_paths = [os.path.join(dataset.path, f) for f in image_files]

    # Load captions if requested
    params: dict[str, Any] = {"hps_version": request.hps_version}
    if request.use_captions:
        captions: dict[str, str] = {}
        for f in image_files:
            caption_path = os.path.join(
                dataset.path, os.path.splitext(f)[0] + ".txt"
            )
            if os.path.exists(caption_path):
                try:
                    with open(caption_path, "r", encoding="utf-8") as fh:
                        captions[f] = fh.read().strip()
                except OSError:
                    pass
        params["captions"] = captions
    params["prompt"] = request.fallback_prompt

    service = ScoringService.get_instance()

    try:
        results = await asyncio.to_thread(
            service.score_batch,
            image_paths,
            request.model_id,
            params,
            None,  # no sync callback — we broadcast after
        )

        # Persist all scores to DB and broadcast progress
        dataset_id = dataset.name
        total = len(results)
        for i, (filename, score) in enumerate(results.items(), 1):
            await asyncio.to_thread(
                media_repo.update,
                dataset_id,
                filename,
                {"quality_score": score},
            )
            await event_manager.broadcast("scoring_progress", {
                "dataset": name,
                "current": i,
                "total": total,
                "file": filename,
                "score": round(score, 4),
            })

        # Compute summary statistics
        scores = [s for s in results.values() if s > 0]
        summary = {
            "scored": len(results),
            "min": round(min(scores), 4) if scores else 0,
            "max": round(max(scores), 4) if scores else 0,
            "avg": round(sum(scores) / len(scores), 4) if scores else 0,
            "median": round(
                sorted(scores)[len(scores) // 2], 4
            ) if scores else 0,
        }

        await event_manager.broadcast("scoring_complete", {
            "dataset": name,
            "summary": summary,
        })

        return summary
    except (OSError, RuntimeError, ValueError) as e:
        logger.error("batch_scoring_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{name}/scores")
async def get_scores_api(name: str):
    """Get all stored quality scores for a dataset."""
    dataset = await asyncio.to_thread(manager.datasets.get, name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    items = await asyncio.to_thread(media_repo.get_by_dataset, dataset.name)
    scores = {
        item["rel_path"]: item.get("quality_score")
        for item in items
        if item.get("quality_score") is not None
    }
    return {"dataset": name, "scores": scores, "count": len(scores)}


@router.delete("/datasets/{name}/scores")
async def clear_scores_api(name: str):
    """Clear all quality scores for a dataset."""
    dataset = await asyncio.to_thread(manager.datasets.get, name)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    items = await asyncio.to_thread(media_repo.get_by_dataset, dataset.name)
    cleared = 0
    for item in items:
        if item.get("quality_score") is not None:
            await asyncio.to_thread(
                media_repo.update,
                dataset.name,
                item["rel_path"],
                {"quality_score": None},
            )
            cleared += 1

    return {"cleared": cleared}


@router.delete("/scoring/unload")
async def unload_scoring_models_api():
    """Unload all scoring models and free VRAM."""
    try:
        logger.info("unloading_scoring_models")
        service = ScoringService.get_instance()
        await asyncio.to_thread(service.unload_models)
        return {"status": "success", "message": "Scoring models unloaded."}
    except (OSError, RuntimeError) as e:
        logger.error("scoring_unload_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
