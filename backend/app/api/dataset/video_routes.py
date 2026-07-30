"""Video curation routes — cutlist import, scene-detect, split, trim, health.

The headline curation surface for video-LoRA datasets:

  * ``POST /datasets/{name}/video/cutlist/parse``   — parse an uploaded
        LosslessCut ``.llc`` / CSV / TSV cutlist into segments (synchronous).
  * ``POST /datasets/{name}/video/split``           — split a long source video
        into clips per the reviewed segments (cpu-lane background task).
  * ``POST /datasets/{name}/video/scene-detect``    — auto-detect scene cuts and
        write reviewable proposals (cpu-lane background task).
  * ``GET  /datasets/{name}/video/scene-proposals`` — fetch written proposals.
  * ``PATCH /datasets/{name}/video/trim``           — non-destructive per-clip
        trim; recomputes clip-health warnings; emits ``dataset.invalidated``.
  * ``GET  /datasets/{name}/video/health``          — per-family clip-health summary.

The split + scene-detect tasks use ``lane="cpu"`` so the heavy ffmpeg/opencv
work never blocks the shared GPU lane (captioning, masking, training preflight).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api._deps import dataset_or_404
from app.api._path_guard import sanitize_filename, validate_path_within
from app.api.schemas.common_schemas import TaskEnqueuedResponse
from app.api.schemas.video_schemas import (
    ClipHealthSummaryResponse,
    CutlistParseResponse,
    SceneDetectRequest,
    SceneProposalsResponse,
    Segment,
    TrimRequest,
    TrimResponse,
    VideoSplitRequest,
)
from app.core.dataset_manager import dataset_manager
from app.core.events import event_manager
from app.core.logger import get_logger
from app.core.tasks.task_manager import task_manager
from app.core.video import clip_health
from app.core.video.cutlist import parse_cutlist

router = APIRouter()
logger = get_logger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _resolve_dataset(name: str):
    """Resolve a dataset or 404. Returns the Dataset object."""
    return dataset_or_404(dataset_manager.get_dataset(name))


def _guard_source(dataset, source_rel_path: str) -> Path:
    """Resolve + containment-check a source rel path inside the dataset root.

    Raises 403 if the path escapes the dataset directory.
    """
    root = Path(dataset.path)
    return validate_path_within(root / source_rel_path, root)


def _source_duration_s(path: Path) -> float:
    """Best-effort source duration (seconds) for cutlist clamping. 0.0 on error."""
    try:
        from app.core.video.probe import probe_video

        return float(probe_video(path).duration_s)
    except Exception:  # noqa: BLE001
        return 0.0


# ── Cutlist parse (synchronous) ──────────────────────────────────────────────


@router.post(
    "/datasets/{name}/video/cutlist/parse", response_model=CutlistParseResponse
)
async def cutlist_parse(
    name: str, file: UploadFile = File(...), source_rel_path: str | None = None
):
    """Parse an uploaded cutlist (.llc / .csv / .tsv) into segments.

    If ``source_rel_path`` is supplied (query param) the referenced clip is
    probed for its duration so ranges can be clamped and open-ended segments
    closed at the clip end. Synchronous — returns the parsed segments directly.
    """
    dataset = _resolve_dataset(name)

    duration = 0.0
    if source_rel_path:
        src = _guard_source(dataset, source_rel_path)
        if src.exists():
            duration = await asyncio.to_thread(_source_duration_s, src)

    data = await file.read()
    filename = file.filename or "cutlist"
    result = await asyncio.to_thread(parse_cutlist, data, filename, duration)

    return CutlistParseResponse(
        segments=result.segments,
        format=result.format,
        warnings=result.warnings,
    )


# ── Split (cpu-lane task) ────────────────────────────────────────────────────


@router.post("/datasets/{name}/video/split", response_model=TaskEnqueuedResponse)
async def video_split(name: str, request: VideoSplitRequest):
    """Enqueue a clip-split task over the reviewed segments (cpu lane)."""
    dataset = _resolve_dataset(name)
    _guard_source(dataset, request.source_rel_path)
    if not request.segments:
        raise HTTPException(status_code=400, detail="No segments to split")

    from app.core.video.split_batch import run_video_split_batch

    segments = [s.model_dump() for s in request.segments]
    # ``output_prefix`` becomes the leading component of every output filename
    # (``{prefix}_{i:03d}.mp4``) inside the dataset dir, and ffmpeg runs with
    # ``-y``. Unsanitized it was an arbitrary-location overwrite. Strip any
    # directory component; an all-separator prefix collapses to "" and falls
    # back to the source stem in the worker.
    output_prefix = (
        sanitize_filename(request.output_prefix) if request.output_prefix else None
    )
    task = task_manager.create(
        type="video_split",
        title=f"Split · {Path(request.source_rel_path).stem}",
        total=len(segments),
        dataset_name=name,
    )
    task_manager.enqueue(
        task.id,
        lambda tid: run_video_split_batch(
            tid,
            dataset_name=name,
            source_rel_path=request.source_rel_path,
            segments=segments,
            mode=request.mode,
            output_prefix=output_prefix,
            archive_source=request.archive_source,
        ),
        lane="cpu",
    )
    return {"task_id": task.id}


# ── Scene detection (cpu-lane task) ──────────────────────────────────────────


@router.post("/datasets/{name}/video/scene-detect", response_model=TaskEnqueuedResponse)
async def video_scene_detect(name: str, request: SceneDetectRequest):
    """Enqueue an auto scene-detection task; writes reviewable proposals (cpu lane)."""
    dataset = _resolve_dataset(name)
    src = _guard_source(dataset, request.source_rel_path)
    if not src.exists():
        raise HTTPException(status_code=404, detail="Source video not found")

    from app.core.video.scene_detect_batch import run_scene_detect_batch

    # Seed total with the source frame count so the progress bar is meaningful.
    total = 0
    try:
        from app.core.video.probe import probe_video

        total = int(probe_video(src).frame_count)
    except Exception:  # noqa: BLE001
        total = 0

    task = task_manager.create(
        type="scene_detect",
        title=f"Scene detect · {Path(request.source_rel_path).stem}",
        total=total,
        dataset_name=name,
    )
    task_manager.enqueue(
        task.id,
        lambda tid: run_scene_detect_batch(
            tid,
            dataset_name=name,
            source_rel_path=request.source_rel_path,
            threshold=request.threshold,
            min_scene_len_s=request.min_scene_len_s,
        ),
        lane="cpu",
    )
    return {"task_id": task.id}


@router.get(
    "/datasets/{name}/video/scene-proposals", response_model=SceneProposalsResponse
)
async def video_scene_proposals(name: str, source_rel_path: str):
    """Return scene-detection proposals for a source clip (``ready=False`` until written)."""
    dataset = _resolve_dataset(name)
    root = Path(dataset.path)
    stem = Path(source_rel_path).stem
    proposals_path = root / ".video" / "proposals" / f"{stem}.json"
    # Containment guard for the resolved proposals file.
    validate_path_within(proposals_path, root)

    if not proposals_path.exists():
        return SceneProposalsResponse(segments=[], ready=False)

    def _read() -> list[Segment]:
        raw = json.loads(proposals_path.read_text(encoding="utf-8"))
        out: list[Segment] = []
        for seg in raw.get("segments", []):
            out.append(
                Segment(
                    start_s=float(seg["start_s"]),
                    end_s=float(seg["end_s"]),
                    label=seg.get("label"),
                )
            )
        return out

    try:
        segments = await asyncio.to_thread(_read)
    except Exception as exc:  # noqa: BLE001
        logger.warning("scene_proposals_read_failed", name=name, error=str(exc))
        raise HTTPException(status_code=500, detail="Could not read proposals")

    return SceneProposalsResponse(segments=segments, ready=True)


# ── Trim (non-destructive) ───────────────────────────────────────────────────


@router.patch("/datasets/{name}/video/trim", response_model=TrimResponse)
async def video_trim(name: str, request: TrimRequest):
    """Set non-destructive trim bounds on a clip and recompute clip-health.

    Updates the media_item row + in-memory metadata, recomputes per-family
    ``clip_warnings`` via :mod:`app.core.video.clip_health`, persists, and emits
    ``dataset.invalidated`` so every connected client reconciles.
    """
    dataset = _resolve_dataset(name)
    lookup_key = request.media_file.replace("\\", "/")

    meta = dataset.media_metadata.get(lookup_key)
    if meta is None:
        raise HTTPException(status_code=404, detail="Media file not found")

    # Recompute clip-health from a prospective post-trim view (a local copy —
    # the None clears the bound back to the clip edge). The live dict isn't
    # touched until update_media_flags applies all three fields atomically.
    prospective = {
        **meta,
        "trim_start_s": request.trim_start_s,
        "trim_end_s": request.trim_end_s,
    }
    warnings = clip_health.compute_clip_warnings(prospective)

    # Persist the single row atomically (also emits media_item entity.changed).
    await dataset_manager.update_media_flags_async(
        name, request.media_file,
        trim_start_s=request.trim_start_s,
        trim_end_s=request.trim_end_s,
        clip_warnings=warnings,
    )

    # Structural reconcile signal — same coarse broadcast DatasetManager uses.
    await event_manager.broadcast("dataset.invalidated", {"name": name})

    logger.info(
        "video_trim_updated",
        dataset_name=name,
        media_file=request.media_file,
        trim_start=request.trim_start_s,
        trim_end=request.trim_end_s,
    )

    return TrimResponse(status="trimmed", clip_warnings=warnings)


# ── Health summary ───────────────────────────────────────────────────────────


@router.get("/datasets/{name}/video/health", response_model=ClipHealthSummaryResponse)
async def video_health(name: str):
    """Return a per-family clip-health summary for all video clips in the dataset."""
    dataset = _resolve_dataset(name)

    items: list[dict] = []
    for rel_path, meta in dataset.media_metadata.items():
        if not meta.get("is_video"):
            continue
        item = dict(meta)
        item["media_file"] = rel_path
        items.append(item)

    summary = clip_health.summarize(items)
    return ClipHealthSummaryResponse.model_validate(summary)
