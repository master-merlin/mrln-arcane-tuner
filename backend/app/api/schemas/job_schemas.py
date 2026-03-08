"""Training job schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class CreateJobRequest(BaseModel):
    """Request body for creating a training job."""
    plugin_id: str
    config: dict[str, Any]


class SetSamplingCadenceRequest(BaseModel):
    """Request body for changing the sampling cadence at runtime."""
    interval: int
