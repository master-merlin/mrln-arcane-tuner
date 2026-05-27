"""Unified download-progress event channel.

Producers (curated `model_registry.download_model` and the Hugging Face
retrofit callsites) emit `model.download_progress` events via the shared
`event_manager` WS bus. Frontend consumers key downloads by
`(source, model_id)`.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel


class DownloadProgress(BaseModel):
    """Single download-progress event payload.

    `total_bytes` / `percent` are nullable because HF often can't determine
    the total upfront (multipart, snapshot of unknown size). The frontend
    renders an indeterminate spinner in that case.
    """
    source: Literal["curated", "hf"]
    model_id: str
    category: Literal["restore", "upscale", "caption", "mask", "training"]
    status: Literal["starting", "downloading", "complete", "error"]
    current_bytes: int = 0
    total_bytes: Optional[int] = None
    percent: Optional[int] = None
    error: Optional[str] = None
