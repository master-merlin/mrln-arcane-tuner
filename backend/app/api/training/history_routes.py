"""Job history, metrics, checkpoints, samples, and rerun config routes."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api._deps import dataset_or_404

router = APIRouter()


def get_dataset_or_404(name: str):
    """Path-operation dependency: resolve a dataset by name or 404.

    Lazy-imports ``dataset_manager`` (matching this module's existing
    local-import style) so ``@patch("app.core.dataset_manager.dataset_manager")``
    — this file's established test-mocking convention — is observed.
    """
    from app.core.dataset_manager import dataset_manager

    return dataset_or_404(dataset_manager.get_dataset(name))


# ── Response schemas ─────────────────────────────────────────────────────


class ModelFamilyCount(BaseModel):
    id: str | None = None
    count: int


class OptimizerCount(BaseModel):
    name: str | None = None
    count: int


class LastJobSummary(BaseModel):
    lora_name: str | None = None
    definition_id: str | None = None
    status: str | None = None
    created_at: float | None = None


class JobStatsResponse(BaseModel):
    total_jobs: int
    completed: int
    failed: int
    stopped: int
    running: int
    paused: int
    success_rate: float
    total_steps: int
    total_runtime_sec: float
    total_training_sec: float
    avg_steps: int
    avg_loss: float
    avg_min_loss: float
    avg_step_time_sec: float
    avg_runtime_sec: float
    model_families: list[ModelFamilyCount]
    optimizers: list[OptimizerCount]
    unique_datasets: int
    last_job: LastJobSummary | None = None


class BackfillResponse(BaseModel):
    completed_runs: int
    rows_updated: int
    fields_recovered: int
    definitions: dict[str, int]


class DefinitionStatsResponse(BaseModel):
    definition_id: str
    stats_available: bool
    run_count: int
    stats: dict[str, Any]
    updated_at: float | None = None


class Checkpoint(BaseModel):
    id: int
    job_id: str
    step: int
    path: str
    lora_file: str | None = None
    lora_size_bytes: int | None = None
    created_at: float
    loss_at_step: float | None = None
    lr_at_step: float | None = None
    is_final: bool = False
    is_deleted: bool = False


class SampleImage(BaseModel):
    id: int
    job_id: str
    step: int
    prompt: str = ""
    seed: int | None = None
    path: str
    width: int = 0
    height: int = 0
    created_at: float


class LossCurvePoint(BaseModel):
    step: int
    loss: float | None = None
    lr: float | None = None
    grad_norm: float | None = None
    timestep_mean: float | None = None
    epoch: float | None = None


class JobMetricsResponse(BaseModel):
    curve: list[LossCurvePoint]
    summary: dict[str, Any]


class JobReplayResponse(BaseModel):
    available: bool
    source: str
    output_dir: str | None = None
    loss: list[Any]


@router.get("/jobs/history/stats", response_model=JobStatsResponse)
async def get_job_stats():
    """Aggregate training statistics for the dashboard card (read-only)."""
    from app.core.db.repositories.job_repo import JobHistoryRepository

    return await asyncio.to_thread(JobHistoryRepository().get_stats)


# ── Estimation-wall statistics ──────────────────────────────────────────


@router.post("/jobs/stats/recompute", response_model=BackfillResponse)
async def recompute_definition_stats():
    """Backfill run costs from disk + rebuild per-definition coefficients.

    Powers the wall's "Update stats from history" action: reconciles missing
    cost fields from each run's output directory, then recomputes the
    calibration coefficients used by ``POST /jobs/estimate``.
    """
    from app.core.stats.backfill import run_backfill

    return await asyncio.to_thread(run_backfill)


@router.get("/jobs/stats/{definition_id}", response_model=DefinitionStatsResponse)
async def get_definition_stats(definition_id: str):
    """Raw calibration stats + freshness for a single definition."""
    from app.core.stats import definition_stats_service

    return await asyncio.to_thread(definition_stats_service.get, definition_id)


@router.get("/jobs/history")
async def list_job_history(
    limit: int = 50,
    offset: int = 0,
    definition_id: str | None = None,
    status: str | None = None,
    lora_name: str | None = None,
    project_id: str | None = None,
):
    """Paginated job history with optional filters."""
    from app.core.db.repositories.job_repo import JobHistoryRepository
    repo = JobHistoryRepository()
    return await asyncio.to_thread(
        repo.list_recent, limit, offset, definition_id, status, lora_name, project_id
    )


@router.get("/jobs/history/{job_id}")
async def get_job_history_detail(job_id: str):
    """Full job detail with checkpoints and samples."""
    from app.core.db.repositories.job_repo import JobHistoryRepository
    from app.core.db.repositories.checkpoint_repo import CheckpointRepository
    from app.core.db.repositories.sample_repo import SampleImageRepository

    job_repo = JobHistoryRepository()
    job = await asyncio.to_thread(job_repo.get_by_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    cp_repo = CheckpointRepository()
    sample_repo = SampleImageRepository()
    job["checkpoints"] = await asyncio.to_thread(cp_repo.list_by_job, job_id)
    job["samples"] = await asyncio.to_thread(sample_repo.list_by_job, job_id)
    job["datasets_linkage"] = await asyncio.to_thread(
        job_repo.get_datasets_for_job, job_id
    )
    return job


@router.get("/jobs/history/{job_id}/checkpoints", response_model=list[Checkpoint])
async def get_job_checkpoints(job_id: str):
    """Checkpoint list for a job."""
    from app.core.db.repositories.checkpoint_repo import CheckpointRepository
    repo = CheckpointRepository()
    return await asyncio.to_thread(repo.list_by_job, job_id)


@router.get("/jobs/history/{job_id}/samples", response_model=list[SampleImage])
async def get_job_samples(job_id: str):
    """Sample images for a job."""
    from app.core.db.repositories.sample_repo import SampleImageRepository
    repo = SampleImageRepository()
    return await asyncio.to_thread(repo.list_by_job, job_id)


@router.get("/jobs/history/{job_id}/metrics", response_model=JobMetricsResponse)
async def get_job_metrics(job_id: str):
    """Loss curve data for charting."""
    from app.core.db.repositories.metrics_repo import MetricsRepository
    repo = MetricsRepository()
    return {
        "curve": await asyncio.to_thread(repo.get_loss_curve, job_id),
        "summary": await asyncio.to_thread(repo.get_summary, job_id),
    }


@router.get("/jobs/history/{job_id}/replay", response_model=JobReplayResponse)
async def get_job_replay(job_id: str):
    """Replay an archived run's loss history for the Jobs detail pane.

    Prefers the on-disk ``{output_dir}/loss_history.json`` written by the
    trainer (the user-facing source of truth); falls back to the persisted
    ``step_metrics`` DB curve so replay still works if the output folder was
    removed. Also reports whether the output folder still exists on disk so the
    UI can offer "restart fresh" / flag missing artifacts.
    """
    from app.core.db.repositories.job_repo import JobHistoryRepository

    job = await asyncio.to_thread(JobHistoryRepository().get_by_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    def _load():
        import json
        from pathlib import Path

        from app.core.db.repositories.metrics_repo import MetricsRepository

        output_dir = job.get("output_dir")
        disk_available = bool(output_dir) and Path(output_dir).is_dir()

        loss: list = []
        source = "none"
        if output_dir:
            history_file = Path(output_dir) / "loss_history.json"
            if history_file.is_file():
                try:
                    data = json.loads(history_file.read_text(encoding="utf-8"))
                    if isinstance(data, list) and data:
                        loss, source = data, "disk"
                except (OSError, ValueError):
                    pass
        if not loss:
            curve = MetricsRepository().get_loss_curve(job_id)
            if curve:
                loss, source = curve, "db"

        return {
            "available": disk_available,
            "source": source,
            "output_dir": output_dir,
            "loss": loss,
        }

    return await asyncio.to_thread(_load)


@router.get("/jobs/history/{job_id}/rerun-config")
async def get_rerun_config(job_id: str):
    """Extract config from a past job for re-submission."""
    from app.core.db.repositories.job_repo import JobHistoryRepository
    repo = JobHistoryRepository()
    config = await asyncio.to_thread(repo.get_config_for_rerun, job_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return config


@router.get("/datasets/{name}/jobs")
async def get_dataset_jobs(ds=Depends(get_dataset_or_404)):
    """All jobs that used a specific dataset."""
    from app.core.db.repositories.job_repo import JobHistoryRepository

    repo = JobHistoryRepository()
    return await asyncio.to_thread(repo.get_by_dataset, ds.id)
