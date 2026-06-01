"""Training job lifecycle, sampling control, and sample image routes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.api._path_guard import validate_path_within
from app.core.job import Job
from app.core.job_manager import job_manager
from app.core.logger import get_logger
from app.api.schemas.job_schemas import CreateJobRequest, SetSamplingCadenceRequest

router = APIRouter()
logger = get_logger(__name__)


# ── Job CRUD & Lifecycle ────────────────────────────────────────────────


@router.post("/jobs", response_model=Job)
async def create_job(request: CreateJobRequest):
    """Create a new training job."""
    try:
        logger.info("creating_job", plugin_id=request.plugin_id)
        return await asyncio.to_thread(job_manager.create_job, request.plugin_id, request.config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/jobs", response_model=list[Job])
async def list_jobs():
    """List all training jobs."""
    return await asyncio.to_thread(job_manager.list_jobs)


@router.post("/jobs/{job_id}/start")
async def start_job(job_id: str):
    """Start a pending training job."""
    try:
        logger.info("starting_job", job_id=job_id)
        await asyncio.to_thread(job_manager.start_job, job_id)
        return {"status": "started", "job_id": job_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (OSError, RuntimeError) as e:
        logger.error("job_start_failed", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/stop")
async def stop_job(job_id: str):
    """Force-stop a running training job."""
    try:
        logger.info("stopping_job", job_id=job_id)
        await asyncio.to_thread(job_manager.stop_job, job_id)
        return {"status": "stopped", "job_id": job_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    """Send a pause signal to a running job."""
    try:
        logger.info("pausing_job", job_id=job_id)
        await asyncio.to_thread(job_manager.pause_job, job_id)
        return {"status": "paused", "job_id": job_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    """Resume a paused training job."""
    try:
        logger.info("resuming_job", job_id=job_id)
        await asyncio.to_thread(job_manager.resume_job, job_id)
        return {"status": "running", "job_id": job_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/jobs/{job_id}/soft-stop")
async def soft_stop_job(job_id: str):
    """Signal the trainer to save a checkpoint and exit gracefully."""
    try:
        logger.info("soft_stopping_job", job_id=job_id)
        await asyncio.to_thread(job_manager.soft_stop_job, job_id)
        return {"status": "soft_stopping", "job_id": job_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/jobs/{job_id}/restart")
async def restart_job(job_id: str, fresh: bool = False):
    """Reset a finished/failed job and re-launch it.

    ``fresh=true`` deletes the run's output folder first (clean restart).
    """
    try:
        logger.info("restarting_job", job_id=job_id, fresh=fresh)
        await asyncio.to_thread(job_manager.restart_job, job_id, fresh)
        return {"status": "restarted", "job_id": job_id, "fresh": fresh}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (OSError, RuntimeError) as e:
        logger.error("job_restart_failed", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/reorder")
async def reorder_job(job_id: str, direction: str = "up"):
    """Move a pending job up/down in the run queue (priority reorder)."""
    try:
        await asyncio.to_thread(job_manager.reorder_pending, job_id, direction)
        return {"status": "reordered", "job_id": job_id, "direction": direction}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/jobs/{job_id}/logs", response_model=list[str])
async def get_job_logs(job_id: str):
    """Return buffered log lines for a job."""
    job = await asyncio.to_thread(job_manager.get_job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.logs


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Remove a job from the registry."""
    logger.info("deleting_job", job_id=job_id)
    await asyncio.to_thread(job_manager.delete_job, job_id)
    return {"status": "deleted", "job_id": job_id}


# ── Sampling Control ────────────────────────────────────────────────────


@router.post("/jobs/{job_id}/pause-sampling")
async def pause_sampling(job_id: str):
    """Pause sampling for a running job (training continues)."""
    try:
        await asyncio.to_thread(job_manager.pause_sampling, job_id)
        return {"status": "sampling_paused", "job_id": job_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/jobs/{job_id}/resume-sampling")
async def resume_sampling(job_id: str):
    """Resume sampling for a job."""
    try:
        await asyncio.to_thread(job_manager.resume_sampling, job_id)
        return {"status": "sampling_resumed", "job_id": job_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/jobs/{job_id}/sampling-status")
async def get_sampling_status(job_id: str):
    """Check if sampling is paused for a job."""
    paused = await asyncio.to_thread(job_manager.is_sampling_paused, job_id)
    return {"job_id": job_id, "sampling_paused": paused}


@router.post("/jobs/{job_id}/sampling-cadence")
async def set_sampling_cadence(job_id: str, request: SetSamplingCadenceRequest):
    """Change how often samples are generated during training."""
    if request.interval <= 0:
        raise HTTPException(status_code=400, detail="Interval must be a positive integer")
    try:
        await asyncio.to_thread(job_manager.set_sampling_cadence, job_id, request.interval)
        return {"status": "cadence_set", "job_id": job_id, "interval": request.interval}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/jobs/{job_id}/sampling-cadence")
async def get_sampling_cadence(job_id: str):
    """Get the current sampling cadence (override + config default)."""
    job = await asyncio.to_thread(job_manager.get_job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    override = await asyncio.to_thread(job_manager.get_sampling_cadence, job_id)
    default_interval = int(job.config.get("sample_every_n_steps", 0))
    return {
        "job_id": job_id,
        "interval": override if override is not None else default_interval,
        "default_interval": default_interval,
    }


# ── Sample Images ───────────────────────────────────────────────────────


def _resolve_sample_dir(job: Job) -> Path:
    """Resolve the sample images directory for a job."""
    cfg = job.config
    output_dir = Path(cfg.get("output_dir", "outputs"))
    lora_name = cfg.get("lora_name", "lora")
    definition_id = cfg.get("definition_id", "")
    model_part = definition_id.split("/")[-1].replace(":", "_")
    return output_dir / f"{lora_name}_{model_part}" / "samples"


@router.get("/jobs/{job_id}/samples")
async def list_job_samples(job_id: str):
    """List all sample images for a job, sorted by step.

    Returns:
        List of dicts with keys: filename, step, index, path, created_at.
    """
    job = await asyncio.to_thread(job_manager.get_job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    sample_dir = _resolve_sample_dir(job)
    if not sample_dir.is_dir():
        return []

    def _scan_samples() -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for fpath in sample_dir.iterdir():
            if not fpath.suffix.lower() == ".png":
                continue

            parts = fpath.stem.split("_")
            step = 0
            index = 0
            is_final = "final" in parts
            for i, part in enumerate(parts):
                if part == "step" or (part.startswith("step") and len(part) > 4):
                    try:
                        step = int(part.replace("step", ""))
                    except ValueError:
                        pass
                elif i == 1:
                    try:
                        index = int(part)
                    except ValueError:
                        pass
            if is_final:
                step = 999999

            items.append({
                "filename": fpath.name,
                "step": step,
                "index": index,
                "path": str(fpath),
                "created_at": fpath.stat().st_mtime,
            })
        return items

    samples = await asyncio.to_thread(_scan_samples)
    samples.sort(key=lambda s: (s["step"], s["index"]), reverse=True)
    return samples


@router.get("/jobs/{job_id}/samples/{filename}")
async def get_sample_image(job_id: str, filename: str):
    """Serve a specific sample image file for modal preview."""
    job = await asyncio.to_thread(job_manager.get_job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    sample_dir = _resolve_sample_dir(job)
    # Validate the filename stays within the sample directory
    fpath = validate_path_within(sample_dir / filename, sample_dir)

    if not fpath.is_file():
        raise HTTPException(status_code=404, detail="Sample not found")

    return FileResponse(str(fpath), media_type="image/png")
