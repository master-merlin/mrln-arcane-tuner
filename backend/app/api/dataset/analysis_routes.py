"""Analysis & harmonization routes — dataset analysis, version bump, harmonize files."""

from __future__ import annotations

import asyncio
from fractions import Fraction

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.dataset.harmonize_batch import run_harmonize_batch
from app.core.dataset_manager import dataset_manager
from app.core.logger import get_logger
from app.core.tasks.task_manager import task_manager

router = APIRouter()
logger = get_logger(__name__)


@router.get("/datasets/{name}/analysis")
async def analyze_dataset(
    name: str,
    similarity_threshold: float = 0.9,
    resolutions: str | None = None,
    bucketing_mode: str = "kohya",
):
    """Analyze dataset for aspect ratio harmonization opportunities."""
    try:
        logger.info(
            "analyzing_dataset",
            dataset_name=name,
            threshold=similarity_threshold,
            resolutions=resolutions,
            bucketing_mode=bucketing_mode,
        )
        result = await asyncio.to_thread(
            dataset_manager.analyze_harmonization, name, similarity_threshold,
        )

        if resolutions:
            from app.core.bucket_preview import preview_buckets

            res_list = [int(r.strip()) for r in resolutions.split(",") if r.strip()]
            if res_list:
                for orientation_data in result.values():
                    if "images" in orientation_data:
                        preview_buckets(orientation_data["images"], res_list, bucketing_mode)

        # Ensure majority_ar_display is populated (fallback for hot-reload)
        for ori_key, orientation_data in result.items():
            ar = orientation_data.get("majority_ar")
            if ar and not orientation_data.get("majority_ar_display"):
                if abs(ar - 1.0) < 0.01:
                    orientation_data["majority_ar_display"] = "1:1"
                else:
                    frac = Fraction(ar).limit_denominator(32)
                    w_part, h_part = frac.numerator, frac.denominator
                    if ori_key == "portrait":
                        orientation_data["majority_ar_display"] = f"{h_part}:{w_part}"
                    else:
                        orientation_data["majority_ar_display"] = f"{w_part}:{h_part}"

        return result
    except ValueError as e:
        msg = str(e)
        status = 404 if "not found" in msg.lower() else 400
        raise HTTPException(status_code=status, detail=msg)


@router.post("/datasets/{name}/bump")
async def bump_version(name: str, type: str = "patch"):
    """Bump the dataset's semantic version."""
    logger.info("bumping_version", dataset_name=name, type=type)
    new_version = await asyncio.to_thread(dataset_manager.bump_dataset_version, name, type)
    if not new_version:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"version": new_version}


class SetVersionRequest(BaseModel):
    """Body for ``POST /datasets/{name}/version``."""
    version: str


@router.post("/datasets/{name}/version")
async def set_version(name: str, body: SetVersionRequest):
    """Manually overwrite the dataset's semantic version.

    Companion to ``POST /datasets/{name}/bump`` — used by the
    version-edit modal to fix an accidentally-bumped version.
    Returns 400 on invalid semver, 404 on unknown dataset.
    """
    logger.info("setting_version", dataset_name=name, version=body.version)
    try:
        new_version = await asyncio.to_thread(
            dataset_manager.set_dataset_version, name, body.version,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not new_version:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"version": new_version}


@router.post("/datasets/{name}/harmonize/task")
async def harmonize_files_task(name: str):
    """Start a backend-owned harmonize task (convert→rename→rescan all files).
    Returns the task id immediately; progress is monitored via TaskStore."""
    pairs = await asyncio.to_thread(dataset_manager.get_dataset_pairs, name)
    task = task_manager.create(
        type="harmonize", title=f"Harmonize · {name}",
        total=len(pairs), dataset_name=name,
    )
    task_manager.enqueue(
        task.id,
        lambda tid: run_harmonize_batch(tid, dataset_name=name),
        lane="gpu",
    )
    return {"task_id": task.id}
