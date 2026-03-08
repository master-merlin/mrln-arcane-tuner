"""Analysis & harmonization routes — dataset analysis, version bump, harmonize files."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from app.core.dataset_manager import dataset_manager
from app.core.logger import get_logger

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
        from fractions import Fraction
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


@router.post("/datasets/{name}/harmonize")
async def harmonize_files(name: str):
    """Convert all media to JPG and rename pairs to a consistent naming scheme."""
    try:
        logger.info("harmonizing_dataset", dataset_name=name)
        return await asyncio.to_thread(dataset_manager.harmonize_files, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
