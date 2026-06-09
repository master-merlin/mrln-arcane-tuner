"""Training job lifecycle, sampling control, and sample image routes."""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from app.api._path_guard import validate_path_within
from app.core.job import Job
from app.core.job_manager import job_manager
from app.core.naming import model_part_from_definition_id
from app.core.logger import get_logger
from app.api.schemas.job_schemas import (
    CreateJobRequest,
    UpdateJobConfigRequest,
    SetSamplingCadenceRequest,
    SetAutoQueueRequest,
    JobActionResponse,
    JobRestartResponse,
    JobReorderResponse,
    JobCadenceSetResponse,
    AutoQueueResponse,
    SamplingStatusResponse,
    SamplingCadenceResponse,
    JobSampleResponse,
    JobCheckpointResponse,
)
from app.core.settings_manager import get_settings_manager

router = APIRouter()
logger = get_logger(__name__)


# ── Job CRUD & Lifecycle ────────────────────────────────────────────────


@router.post("/jobs", response_model=Job)
async def create_job(request: CreateJobRequest):
    """Create a new training job."""
    try:
        logger.info("creating_job", plugin_id=request.plugin_id)
        job = await asyncio.to_thread(job_manager.create_job, request.plugin_id, request.config)
        # Kick the queue so a freshly-created job auto-starts when auto-queue is
        # on and the GPU is idle — otherwise the only job in an empty queue sits
        # PENDING forever (advance_queue otherwise only fires on a prior job's
        # terminal transition). Gated + single-GPU-safe inside advance_queue, so
        # a no-op when auto-queue is off or a job is already running. Off-thread,
        # non-blocking — mirrors the toggle-on drain in set_auto_queue.
        job_manager.schedule_advance_queue()
        return job
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/jobs/{job_id}/config")
async def update_job_config(job_id: str, request: UpdateJobConfigRequest):
    """Edit a pending or terminal job's stored config (running/paused locked)."""
    try:
        logger.info("updating_job_config", job_id=job_id)
        return await asyncio.to_thread(job_manager.update_job_config, job_id, request.config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/jobs", response_model=list[Job])
async def list_jobs():
    """List all training jobs."""
    return await asyncio.to_thread(job_manager.list_jobs)


@router.get("/jobs/settings/auto-queue", response_model=AutoQueueResponse)
async def get_auto_queue():
    """Return the persisted backend auto-queue preference."""
    sm = get_settings_manager()
    mod = await asyncio.to_thread(sm.get_module_settings, "jobs")
    return {"auto_queue": bool(mod.get("auto_queue", False))}


@router.put("/jobs/settings/auto-queue", response_model=AutoQueueResponse)
async def set_auto_queue(request: SetAutoQueueRequest):
    """Persist the auto-queue preference server-side.

    Stored server-side (not browser localStorage) so the backend advances the
    queue unattended. Turning it on immediately drains a backlog if the GPU is
    idle.
    """
    sm = get_settings_manager()
    await asyncio.to_thread(
        sm.update_module_settings, "jobs", {"auto_queue": request.enabled}
    )
    logger.info("auto_queue_setting_changed", enabled=request.enabled)
    if request.enabled:
        # Idle GPU + pending jobs → start draining now (off-thread, non-blocking).
        job_manager.schedule_advance_queue()
    return {"auto_queue": request.enabled}


@router.post("/jobs/{job_id}/start", response_model=JobActionResponse)
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


@router.post("/jobs/{job_id}/stop", response_model=JobActionResponse)
async def stop_job(job_id: str):
    """Force-stop a running training job."""
    try:
        logger.info("stopping_job", job_id=job_id)
        await asyncio.to_thread(job_manager.stop_job, job_id)
        return {"status": "stopped", "job_id": job_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/jobs/{job_id}/pause", response_model=JobActionResponse)
async def pause_job(job_id: str):
    """Send a pause signal to a running job."""
    try:
        logger.info("pausing_job", job_id=job_id)
        await asyncio.to_thread(job_manager.pause_job, job_id)
        return {"status": "paused", "job_id": job_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/jobs/{job_id}/resume", response_model=JobActionResponse)
async def resume_job(job_id: str):
    """Resume a paused training job."""
    try:
        logger.info("resuming_job", job_id=job_id)
        await asyncio.to_thread(job_manager.resume_job, job_id)
        return {"status": "running", "job_id": job_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/jobs/{job_id}/soft-stop", response_model=JobActionResponse)
async def soft_stop_job(job_id: str):
    """Signal the trainer to save a checkpoint and exit gracefully."""
    try:
        logger.info("soft_stopping_job", job_id=job_id)
        await asyncio.to_thread(job_manager.soft_stop_job, job_id)
        return {"status": "soft_stopping", "job_id": job_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/jobs/{job_id}/restart", response_model=JobRestartResponse)
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


@router.post("/jobs/{job_id}/reorder", response_model=JobReorderResponse)
async def reorder_job(job_id: str, direction: str = "up"):
    """Move a pending job up/down in the run queue (priority reorder)."""
    try:
        await asyncio.to_thread(job_manager.reorder_pending, job_id, direction)
        return {"status": "reordered", "job_id": job_id, "direction": direction}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/jobs/{job_id}/logs", response_model=list[str])
async def get_job_logs(job_id: str):
    """Return buffered log lines for a job.

    Falls back to the persisted ``job_log.jsonl`` on disk when the in-memory
    buffer is empty (a stopped/failed job, or after a backend restart) so the
    tail always survives.
    """
    def _resolve() -> list[str] | None:
        job = job_manager.get_job(job_id)
        if not job:
            return None
        return job.logs if job.logs else job_manager.read_persisted_logs(job)

    result = await asyncio.to_thread(_resolve)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return result


@router.delete("/jobs/{job_id}", response_model=JobActionResponse)
async def delete_job(job_id: str):
    """Remove a job from the registry."""
    logger.info("deleting_job", job_id=job_id)
    await asyncio.to_thread(job_manager.delete_job, job_id)
    return {"status": "deleted", "job_id": job_id}


# ── Sampling Control ────────────────────────────────────────────────────


@router.post("/jobs/{job_id}/pause-sampling", response_model=JobActionResponse)
async def pause_sampling(job_id: str):
    """Pause sampling for a running job (training continues)."""
    try:
        await asyncio.to_thread(job_manager.pause_sampling, job_id)
        return {"status": "sampling_paused", "job_id": job_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/jobs/{job_id}/resume-sampling", response_model=JobActionResponse)
async def resume_sampling(job_id: str):
    """Resume sampling for a job."""
    try:
        await asyncio.to_thread(job_manager.resume_sampling, job_id)
        return {"status": "sampling_resumed", "job_id": job_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/jobs/{job_id}/sampling-status", response_model=SamplingStatusResponse)
async def get_sampling_status(job_id: str):
    """Check if sampling is paused for a job."""
    paused = await asyncio.to_thread(job_manager.is_sampling_paused, job_id)
    return {"job_id": job_id, "sampling_paused": paused}


@router.post("/jobs/{job_id}/sampling-cadence", response_model=JobCadenceSetResponse)
async def set_sampling_cadence(job_id: str, request: SetSamplingCadenceRequest):
    """Change how often samples are generated during training."""
    if request.interval <= 0:
        raise HTTPException(status_code=400, detail="Interval must be a positive integer")
    try:
        await asyncio.to_thread(job_manager.set_sampling_cadence, job_id, request.interval)
        return {"status": "cadence_set", "job_id": job_id, "interval": request.interval}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/jobs/{job_id}/sampling-cadence", response_model=SamplingCadenceResponse)
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
    model_part = model_part_from_definition_id(definition_id)
    return output_dir / f"{lora_name}_{model_part}" / "samples"


@router.get("/jobs/{job_id}/samples", response_model=list[JobSampleResponse])
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


def _resolve_run_dir(job: Job) -> Path:
    """Resolve a job's run output directory (parent of ``samples/``).

    Holds the LoRA ``.safetensors`` files written at each checkpoint
    (``{name}_{step:06d}.safetensors`` / ``{name}_final.safetensors``) plus
    the ``checkpoint-NNNNNN/`` training-state folders.
    """
    return _resolve_sample_dir(job).parent


def _checkpoint_dir_name(step: int, is_final: bool) -> str:
    """Training-state folder name paired with a distribution LoRA at ``step``.

    Mirrors the checkpoint saver: ``final`` for the final save, else the
    zero-padded ``checkpoint-NNNNNN``.
    """
    return "final" if is_final else f"checkpoint-{step:06d}"


# Folder names accepted by the zipped-checkpoint endpoint.
_CHECKPOINT_DIR_RE = re.compile(r"^(final|checkpoint-\d{3,})$")


@router.get("/jobs/{job_id}/checkpoints", response_model=list[JobCheckpointResponse])
async def list_job_checkpoints(job_id: str):
    """List the LoRA ``.safetensors`` artifacts a job has saved.

    Each entry corresponds to one saved checkpoint and is directly
    downloadable via ``GET /jobs/{job_id}/checkpoints/{filename}``.

    Returns:
        List of dicts: filename, step, is_final, size_bytes, created_at.
    """
    job = await asyncio.to_thread(job_manager.get_job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    run_dir = _resolve_run_dir(job)
    if not run_dir.is_dir():
        return []

    def _scan() -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for fpath in run_dir.iterdir():
            if not fpath.is_file() or fpath.suffix.lower() != ".safetensors":
                continue
            stem = fpath.stem
            is_final = stem.endswith("_final") or stem == "final"
            step = 0
            m = re.search(r"_(\d{3,})$", stem)
            if m:
                step = int(m.group(1))
            stat = fpath.stat()
            # A resumable training-state folder exists when its
            # training_state.json is present (it may have been pruned by
            # keep_last_checkpoints, leaving only the distribution LoRA).
            ckpt_name = _checkpoint_dir_name(step, is_final)
            resumable = (run_dir / ckpt_name / "training_state.json").is_file()
            items.append({
                "filename": fpath.name,
                "step": 999999 if is_final else step,
                "is_final": is_final,
                "size_bytes": stat.st_size,
                "created_at": stat.st_mtime,
                "resumable": resumable,
                "checkpoint_dir": ckpt_name if resumable else None,
            })
        return items

    checkpoints = await asyncio.to_thread(_scan)
    checkpoints.sort(key=lambda c: c["step"], reverse=True)
    return checkpoints


@router.get("/jobs/{job_id}/checkpoints/{filename}")
async def download_job_checkpoint(job_id: str, filename: str):
    """Download a job's LoRA ``.safetensors`` checkpoint file."""
    job = await asyncio.to_thread(job_manager.get_job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    run_dir = _resolve_run_dir(job)
    fpath = validate_path_within(run_dir / filename, run_dir)

    if not fpath.is_file() or fpath.suffix.lower() != ".safetensors":
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    # `filename=` sets Content-Disposition: attachment so the browser downloads
    # the file (works cross-origin, independent of the <a download> attribute).
    return FileResponse(
        str(fpath),
        media_type="application/octet-stream",
        filename=fpath.name,
    )


@router.get("/jobs/{job_id}/checkpoints/{folder}/zip")
async def download_checkpoint_zip(job_id: str, folder: str):
    """Download a resumable training-state checkpoint as a ``.zip``.

    Bundles the whole ``checkpoint-NNNNNN/`` (or ``final/``) folder — adapters,
    optimizer/scheduler/scaler/EMA state and ``training_state.json`` — so it can
    be moved to another pod and used to resume via ``resume_from_checkpoint``
    (manual import for now). Stored uncompressed (the contents are dense tensor
    blobs that barely deflate) and streamed from a temp file so a multi-GB
    checkpoint never sits in memory.
    """
    job = await asyncio.to_thread(job_manager.get_job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not _CHECKPOINT_DIR_RE.match(folder):
        raise HTTPException(status_code=400, detail="Invalid checkpoint folder")

    run_dir = _resolve_run_dir(job)
    ckpt_dir = validate_path_within(run_dir / folder, run_dir)
    if not ckpt_dir.is_dir():
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    def _build_zip() -> str:
        fd, tmp_path = tempfile.mkstemp(suffix=".zip", prefix="ckpt_")
        os.close(fd)
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
            for p in sorted(ckpt_dir.rglob("*")):
                if p.is_file():
                    # POSIX arcnames so the archive extracts cleanly on a Linux pod.
                    zf.write(p, p.relative_to(ckpt_dir).as_posix())
        return tmp_path

    tmp_path = await asyncio.to_thread(_build_zip)
    zip_name = f"{run_dir.name}_{folder}.zip"
    return FileResponse(
        tmp_path,
        media_type="application/zip",
        filename=zip_name,
        background=BackgroundTask(os.remove, tmp_path),
    )
