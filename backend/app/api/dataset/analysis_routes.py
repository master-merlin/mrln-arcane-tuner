"""Analysis & harmonization routes — dataset analysis, version bump, harmonize files."""

from __future__ import annotations

import asyncio
from fractions import Fraction

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.schemas.common_schemas import TaskEnqueuedResponse
from app.core.dataset.harmonize_batch import run_harmonize_batch
from app.core.dataset.tag_analytics import compute_tag_analytics
from app.core.dataset_manager import dataset_manager
from app.core.logger import get_logger
from app.core.tasks.task_manager import task_manager

router = APIRouter()
logger = get_logger(__name__)


class VersionResponse(BaseModel):
    """The dataset's semantic version after a bump/set operation."""
    version: str


class TagCount(BaseModel):
    tag: str
    count: int


class Cooccurrence(BaseModel):
    labels: list[str]
    matrix: list[list[int]]


class Contradiction(BaseModel):
    a: str
    b: str
    count: int
    images: list[str]


class TagAnalyticsResponse(BaseModel):
    total_images: int
    total_tags: int
    style: str = "tags"  # "tags" (comma-split) or "prose" (word/phrase) analysis
    top_tags: list[TagCount]
    orphan_tags: list[str]
    cooccurrence: Cooccurrence
    contradictions: list[Contradiction]


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


@router.get("/datasets/{name}/tag-analytics", response_model=TagAnalyticsResponse)
async def tag_analytics(
    name: str,
    top_n: int = 30,
    definition_id: str | None = None,
    masked: bool = False,
):
    """Tag frequency, orphans, co-occurrence matrix, and contradictions.

    When ``definition_id`` is given (model-aware), captions are resolved through
    the per-definition variant resolver and the analysis style is derived from
    that model (CLIP/SDXL → tag-split, T5/large-context → prose word/phrase).
    Without it, general captions are analysed with an auto-detected style.
    """
    ds = dataset_manager.get_dataset(name)
    if ds is None:
        raise HTTPException(404, f"Dataset '{name}' not found.")

    def _compute() -> dict:
        pairs = dataset_manager.get_dataset_pairs(name)
        style: str | None = None
        if definition_id:
            from pathlib import Path

            from app.core.captioning import caption_variants
            from app.core.dataset.tag_analytics import detect_style
            from app.core.llm import caption_refine
            from app.engine.core.caption_target import resolve_caption_target

            items = [
                (
                    p.get("media_file", ""),
                    caption_variants.resolve_caption(
                        ds.path, Path(p.get("media_file", "")).stem, definition_id, masked
                    ),
                )
                for p in pairs
            ]
            try:
                target = resolve_caption_target(definition_id)
                style = "tags" if caption_refine.caption_style_for(target) == "tags" else "prose"
            except Exception:
                style = None
            # Honor the model's style, but never tag-split captions that are
            # actually prose (that's the "one giant tag" uselessness) — fall back
            # to prose so the analysis stays meaningful.
            if style == "tags" and detect_style(items) == "prose":
                style = "prose"
        else:
            items = [(p.get("media_file", ""), p.get("caption_content", "") or "") for p in pairs]
        return compute_tag_analytics(items, top_n=top_n, style=style)

    return await asyncio.to_thread(_compute)


@router.post("/datasets/{name}/bump", response_model=VersionResponse)
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


@router.post("/datasets/{name}/version", response_model=VersionResponse)
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


@router.post("/datasets/{name}/harmonize/task", response_model=TaskEnqueuedResponse)
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
