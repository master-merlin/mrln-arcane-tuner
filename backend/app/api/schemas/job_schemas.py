"""Training job schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class CreateJobRequest(BaseModel):
    """Request body for creating a training job."""
    plugin_id: str
    config: dict[str, Any]


class UpdateJobConfigRequest(BaseModel):
    """Request body for editing a job's stored training config."""
    config: dict[str, Any]


class ResumeFromCheckpointRequest(BaseModel):
    """Request body for continuing a job from one of its checkpoints."""
    checkpoint_dir: str


class SetSamplingCadenceRequest(BaseModel):
    """Request body for changing the sampling cadence at runtime."""
    interval: int


class SetAutoQueueRequest(BaseModel):
    """Request body for toggling backend-owned auto-queue advancement."""
    enabled: bool


class SetAutoResumeRequest(BaseModel):
    """Request body for toggling auto-resume after a transient GPU fault."""
    enabled: bool


class JobActionResponse(BaseModel):
    """Status + job id returned by a job lifecycle action."""
    status: str
    job_id: str


class JobRestartResponse(JobActionResponse):
    """Restart action result, including whether the run folder was wiped."""
    fresh: bool


class JobReorderResponse(JobActionResponse):
    """Reorder action result, including the direction moved."""
    direction: str


class JobCadenceSetResponse(JobActionResponse):
    """Sampling-cadence change result, including the new interval."""
    interval: int


class AutoQueueResponse(BaseModel):
    """Current backend auto-queue preference."""
    auto_queue: bool


class AutoResumeResponse(BaseModel):
    """Current auto-resume-on-GPU-fault preference."""
    auto_resume: bool


class SamplingStatusResponse(BaseModel):
    """Whether sampling is currently paused for a job."""
    job_id: str
    sampling_paused: bool


class SamplingCadenceResponse(BaseModel):
    """Effective sampling cadence (override + config default) for a job."""
    job_id: str
    interval: int
    default_interval: int


class JobSampleResponse(BaseModel):
    """A single sample image produced during training."""
    filename: str
    step: int
    index: int
    path: str
    created_at: float


class JobCheckpointResponse(BaseModel):
    """A single saved LoRA checkpoint artifact for a job."""
    filename: str
    step: int
    is_final: bool
    size_bytes: int
    created_at: float
    # Whether a resumable training-state folder (checkpoint-NNNNNN/ or final/)
    # exists for this step — drives the "download .zip checkpoint" affordance.
    resumable: bool = False
    # Name of that folder, or None when only the distribution LoRA remains
    # (e.g. the training-state was pruned by keep_last_checkpoints).
    checkpoint_dir: str | None = None
