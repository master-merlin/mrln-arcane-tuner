"""Unified download-progress event channel.

Producers (curated `model_registry.download_model` and the Hugging Face
retrofit callsites) emit `model.download_progress` events via the shared
`event_manager` WS bus. Frontend consumers key downloads by
`(source, model_id)`.
"""
from __future__ import annotations

import time as _time
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


class RateLimiter:
    """Per-download throttle for `model.download_progress` emits.

    Intended for single-threaded, per-download-instance use — each
    in-flight download (curated or HF) instantiates its own RateLimiter,
    so no shared state crosses downloads.

    `starting`, `complete`, `error` are pass-through.
    `downloading` is throttled — passes only on either:
      - ≥ `min_interval_s` seconds since the last permitted emit, or
      - ≥ `min_delta_pct` percent-point jump since the last permitted emit.
    """

    def __init__(self, min_interval_s: float = 0.2, min_delta_pct: float = 5.0):
        self.min_interval_s = min_interval_s
        self.min_delta_pct = min_delta_pct
        self._last_emit_t: float = 0.0
        self._last_emit_pct: Optional[float] = None

    def allow(self, status: str, percent: Optional[float]) -> bool:
        if status in ("starting", "complete", "error"):
            self._last_emit_t = _time.monotonic()
            self._last_emit_pct = percent
            return True
        # status == "downloading"
        now = _time.monotonic()
        time_ok = (now - self._last_emit_t) >= self.min_interval_s
        if percent is None or self._last_emit_pct is None:
            # Without a percent we can only use time
            if time_ok:
                self._last_emit_t = now
                self._last_emit_pct = percent
                return True
            return False
        pct_ok = abs(percent - self._last_emit_pct) >= self.min_delta_pct
        if time_ok or pct_ok:
            self._last_emit_t = now
            self._last_emit_pct = percent
            return True
        return False
