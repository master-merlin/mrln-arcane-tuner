"""Job history, metrics, checkpoints, samples, and rerun config routes."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


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
    """Aggregate training statistics for the dashboard card."""
    from app.core.db.engine import get_db

    def _compute():
        conn = get_db().connection()

        # ── Core counts ──────────────────────────────────────────
        totals = conn.execute("""
            SELECT
                COUNT(*)                                              AS total_jobs,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed,
                SUM(CASE WHEN status = 'failed'    THEN 1 ELSE 0 END) AS failed,
                SUM(CASE WHEN status = 'stopped'   THEN 1 ELSE 0 END) AS stopped,
                SUM(CASE WHEN status = 'running'   THEN 1 ELSE 0 END) AS running,
                SUM(CASE WHEN status = 'paused'    THEN 1 ELSE 0 END) AS paused,
                COALESCE(SUM(completed_steps), 0)                      AS total_steps,
                COALESCE(SUM(duration_seconds), 0)                     AS total_runtime_sec,
                COALESCE(SUM(training_seconds), 0)                     AS total_training_sec
            FROM job_history
        """).fetchone()

        total_jobs = totals["total_jobs"] or 0
        completed  = totals["completed"] or 0

        # ── Averages (completed only) ────────────────────────────
        avgs = conn.execute("""
            SELECT
                AVG(completed_steps) AS avg_steps,
                AVG(avg_loss)        AS avg_loss,
                AVG(min_loss)        AS avg_min_loss,
                AVG(avg_step_time)   AS avg_step_time_sec,
                AVG(duration_seconds) AS avg_runtime_sec
            FROM job_history WHERE status = 'completed'
        """).fetchone()

        # ── Model family breakdown ───────────────────────────────
        # Auto-repair legacy records where definition_id was stored as
        # the plugin_id placeholder "standard" instead of the real model ID.
        db = get_db()
        with db.write() as wconn:
            wconn.execute("""
                UPDATE job_history
                SET definition_id = json_extract(config, '$.definition_id')
                WHERE definition_id = 'standard'
                  AND json_extract(config, '$.definition_id') IS NOT NULL
                  AND json_extract(config, '$.definition_id') != ''
            """)

        families = conn.execute("""
            SELECT definition_id, COUNT(*) AS count
            FROM job_history
            GROUP BY definition_id
            ORDER BY count DESC
        """).fetchall()

        # ── Optimizer breakdown ──────────────────────────────────
        optimizers = conn.execute("""
            SELECT optimizer_type, COUNT(*) AS count
            FROM job_history
            WHERE optimizer_type IS NOT NULL
            GROUP BY optimizer_type
            ORDER BY count DESC
        """).fetchall()

        # ── Dataset usage ────────────────────────────────────────
        dataset_stats = conn.execute("""
            SELECT COUNT(DISTINCT dataset_name) AS unique_datasets
            FROM job_datasets
        """).fetchone()

        # ── Most recent job ──────────────────────────────────────
        last_job = conn.execute("""
            SELECT lora_name, definition_id, status, created_at
            FROM job_history ORDER BY created_at DESC LIMIT 1
        """).fetchone()

        return {
            "total_jobs": total_jobs,
            "completed": completed,
            "failed": totals["failed"] or 0,
            "stopped": totals["stopped"] or 0,
            "running": totals["running"] or 0,
            "paused": totals["paused"] or 0,
            "success_rate": round(completed / total_jobs * 100, 1) if total_jobs > 0 else 0,
            "total_steps": totals["total_steps"],
            "total_runtime_sec": round(totals["total_runtime_sec"], 1),
            "total_training_sec": round(totals["total_training_sec"], 1),
            "avg_steps": round(avgs["avg_steps"] or 0),
            "avg_loss": round(avgs["avg_loss"] or 0, 6),
            "avg_min_loss": round(avgs["avg_min_loss"] or 0, 6),
            "avg_step_time_sec": round(avgs["avg_step_time_sec"] or 0, 3),
            "avg_runtime_sec": round(avgs["avg_runtime_sec"] or 0, 1),
            "model_families": [
                {"id": r["definition_id"], "count": r["count"]}
                for r in families
            ],
            "optimizers": [
                {"name": r["optimizer_type"], "count": r["count"]}
                for r in optimizers
            ],
            "unique_datasets": dataset_stats["unique_datasets"] or 0,
            "last_job": dict(last_job) if last_job else None,
        }

    return await asyncio.to_thread(_compute)


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
async def get_dataset_jobs(name: str):
    """All jobs that used a specific dataset."""
    from app.core.db.repositories.job_repo import JobHistoryRepository
    from app.core.dataset_manager import dataset_manager

    ds = await asyncio.to_thread(dataset_manager.get_dataset, name)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    repo = JobHistoryRepository()
    return await asyncio.to_thread(repo.get_by_dataset, ds.id)
