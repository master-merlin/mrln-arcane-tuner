"""Training job data models.

Defines the ``Job`` Pydantic model and ``JobStatus`` enum used by
the ``JobManager`` to track training lifecycle states.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Lifecycle states for a training job."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class Job(BaseModel):
    """Single training job with config, status, and process metadata."""

    id: str
    plugin_id: str
    config: dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    pid: int | None = None
    error: str | None = None
    logs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    status_label: str | None = None
    paused_at: float | None = None
    # In-memory queue priority for pending jobs (lower = runs sooner). Not
    # persisted: resets to FIFO-by-created_at on a server restart.
    priority: int = 0

    @classmethod
    def create(cls, plugin_id: str, config: dict[str, Any]) -> Job:
        """Factory: create a new pending job with a UUID."""
        return cls(
            id=str(uuid.uuid4()),
            plugin_id=plugin_id,
            config=config,
            created_at=time.time(),
        )


class JobHistory(BaseModel):
    """Persistent training run record stored in SQLite.

    Unlike ``Job`` (transient, in-memory queue item), ``JobHistory``
    captures the full lifecycle of a completed/failed training run
    for post-hoc analysis and re-running.
    """

    id: str
    lora_name: str = ""
    definition_id: str = ""
    status: str = "pending"
    config: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    duration_seconds: float | None = None
    training_seconds: float | None = None
    error: str | None = None
    output_dir: str | None = None
    final_checkpoint: str | None = None
    final_lora_file: str | None = None
    final_lora_size_bytes: int | None = None
    total_steps: int = 0
    completed_steps: int = 0
    config_version: str | None = None
    config_schema_version: int | None = None
    resumed_from: str | None = None
    datasets_used: list[str] = Field(default_factory=list)
    network_rank: int | None = None
    network_alpha: int | None = None
    optimizer_type: str | None = None
    learning_rate: float | None = None
    lr_scheduler: str | None = None
    timestep_sampling: str | None = None
    batch_size: int | None = None
    grad_accum: int | None = None
    avg_loss: float | None = None
    min_loss: float | None = None
    avg_step_time: float | None = None
    avg_save_time: float | None = None
    targeted_layers: list[str] | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    parent_job_id: str | None = None
    quantization: str | None = None
    mixed_precision: str | None = None
    ema_enabled: bool = False
