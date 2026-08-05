"""Job history, metrics, checkpoints, samples, and rerun config routes."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

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


class OptimizerCount(BaseModel):
    name: str | None = None
    count: int


class LastJobSummary(BaseModel):
    lora_name: str | None = None
    definition_id: str | None = None
    status: str | None = None
    created_at: float | None = None


class ActivityWeek(BaseModel):
    week_start: str
    completed: int
    failed: int
    stopped: int
    other: int


class FamilyStats(BaseModel):
    id: str | None = None
    count: int
    completed: int
    success_rate: float
    avg_step_time: float | None = None
    best_loss: float | None = None


class HyperparamCount(BaseModel):
    value: str
    count: int


class LossHistogram(BaseModel):
    edges: list[float]
    counts: list[int]


class DatasetUseCount(BaseModel):
    name: str
    count: int


class JobRecord(BaseModel):
    job_id: str
    lora_name: str
    definition_id: str | None = None
    value: float


class JobRecords(BaseModel):
    longest_run: JobRecord | None = None
    most_steps: JobRecord | None = None
    best_loss: JobRecord | None = None


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
    optimizers: list[OptimizerCount]
    unique_datasets: int
    last_job: LastJobSummary | None = None
    activity: list[ActivityWeek]
    gpu_hours: float
    overhead_pct: float
    lora_count: int
    lora_bytes: int
    lora_on_disk: int
    lora_size_known: int
    checkpoint_count: int
    families: list[FamilyStats]
    loss_histogram: LossHistogram
    hyperparams: dict[str, list[HyperparamCount]]
    resume_rate: float
    top_datasets: list[DatasetUseCount]
    records: JobRecords


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


class JobAdaptiveHistoryResponse(BaseModel):
    """A run's ``adaptive_targeting.json`` (spec §6).

    Event rows and the heat map are written by the trainer-side controller and
    grow with it, so both stay open — the UI renders whatever fields the run
    that produced them emitted. ``heat`` values are nullable: a non-finite
    measurement is stored as null rather than 0.0, which would read as a layer
    that never learned.
    """

    events: list[dict[str, Any]] = Field(default_factory=list)
    modules: list[str] = Field(default_factory=list)
    heat: dict[str, float | None] = Field(default_factory=dict)


class JobHistoryRow(BaseModel):
    """One raw ``job_history`` table row (``SELECT *``, with
    config/datasets_used/loss_history/targeted_layers/tags JSON-decoded by
    the repo's ``_from_row``). Open model: this table has grown via 7+ ALTER
    TABLE migrations and the frontend consumes it as an open ``Job`` record
    (``job.ts`` — ``config`` is explicitly typed ``Record<string, unknown>``
    there) — only the NOT-NULL core is declared here; every other (and any
    future) column passes through untouched via ``extra=\"allow\"``."""

    model_config = ConfigDict(extra="allow")

    id: str
    lora_name: str
    definition_id: str
    status: str
    created_at: float


class JobHistoryDetail(JobHistoryRow):
    """Full job detail: the history row plus its linked checkpoints, sample
    images, and dataset-linkage rows."""

    checkpoints: list[Checkpoint] = Field(default_factory=list)
    samples: list[SampleImage] = Field(default_factory=list)
    # `job_datasets` join-table rows — small/stable shape, but kept as an
    # open dict (not a named model) since it isn't otherwise exposed as a
    # standalone contract elsewhere.
    datasets_linkage: list[dict[str, Any]] = Field(default_factory=list)


@router.get("/jobs/history/stats", response_model=JobStatsResponse)
async def get_job_stats(project_id: str | None = None):
    """Aggregate training statistics for the stats modal (read-only)."""
    from app.core.db.repositories.job_repo import JobHistoryRepository

    pid = project_id if project_id and project_id != "all" else None
    return await asyncio.to_thread(JobHistoryRepository().get_stats, pid)


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


@router.get("/jobs/history", response_model=list[JobHistoryRow])
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


@router.get("/jobs/history/{job_id}", response_model=JobHistoryDetail)
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


@router.get(
    "/jobs/history/{job_id}/adaptive", response_model=JobAdaptiveHistoryResponse,
)
async def get_job_adaptive_history(job_id: str):
    """Adaptive layer-targeting event history for a run (spec §6).

    Reads the run dir's ``adaptive_targeting.json`` — the same
    ``job_history.output_dir`` resolution the sibling ``/replay`` route uses.
    A run with the feature off (or one predating it) has no such file: that is
    the empty shape with HTTP 200, not an error, so the UI hides the section
    without special-casing failures. A corrupt/unreadable file degrades the
    same way rather than failing the whole job-detail view.
    """
    from app.core.db.repositories.job_repo import JobHistoryRepository

    job = await asyncio.to_thread(JobHistoryRepository().get_by_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    empty: dict[str, Any] = {"events": [], "modules": [], "heat": {}}

    def _read() -> dict[str, Any]:
        import json
        from pathlib import Path

        from app.engine.components.adaptive_targeting import HISTORY_FILENAME

        output_dir = job.get("output_dir")
        if not output_dir:
            return empty
        path = Path(output_dir) / HISTORY_FILENAME
        if not path.is_file():
            return empty
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return empty
        if not isinstance(data, dict):
            return empty
        # Shape-check each key independently: a half-written or older-schema
        # file must degrade to what IS readable, never 500 the route through
        # response-model validation.
        events = data.get("events")
        modules = data.get("modules")
        heat = data.get("heat")
        return {
            "events": [e for e in events if isinstance(e, dict)]
            if isinstance(events, list) else [],
            "modules": [m for m in modules if isinstance(m, str)]
            if isinstance(modules, list) else [],
            "heat": {
                k: v for k, v in heat.items()
                if isinstance(k, str) and (v is None or isinstance(v, (int, float)))
            } if isinstance(heat, dict) else {},
        }

    return await asyncio.to_thread(_read)


@router.get("/jobs/history/{job_id}/rerun-config", response_model=dict[str, Any])
async def get_rerun_config(job_id: str):
    """Extract config from a past job for re-submission.

    The training config is a plugin-schema-driven blob whose fields vary per
    model family (mirrors ``TrainingConfig`` in job.ts) — ``dict[str, Any]``
    is an intentional open passthrough, not a stand-in for an unwritten model.
    """
    from app.core.db.repositories.job_repo import JobHistoryRepository
    repo = JobHistoryRepository()
    config = await asyncio.to_thread(repo.get_config_for_rerun, job_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return config


@router.get("/datasets/{name}/jobs", response_model=list[JobHistoryRow])
async def get_dataset_jobs(ds=Depends(get_dataset_or_404)):
    """All jobs that used a specific dataset."""
    from app.core.db.repositories.job_repo import JobHistoryRepository

    repo = JobHistoryRepository()
    return await asyncio.to_thread(repo.get_by_dataset, ds.id)
