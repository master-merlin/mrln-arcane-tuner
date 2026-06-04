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


class SetSamplingCadenceRequest(BaseModel):
    """Request body for changing the sampling cadence at runtime."""
    interval: int


class SetAutoQueueRequest(BaseModel):
    """Request body for toggling backend-owned auto-queue advancement."""
    enabled: bool
