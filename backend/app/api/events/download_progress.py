"""Unified download-progress event channel.

Producers (curated `model_registry.download_model` and the Hugging Face
retrofit callsites) emit `model.download_progress` events via the shared
`event_manager` WS bus. Frontend consumers key downloads by
`(source, model_id)`.
"""
from __future__ import annotations

import asyncio
import time as _time
from typing import Literal, Optional

from pydantic import BaseModel

from app.core.events import event_manager


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


# ── Loop capture + emit helpers ───────────────────────────────────────────

# Captured by main.lifespan on startup. Worker-thread code (WSProgressTqdm)
# uses this with `asyncio.run_coroutine_threadsafe` to schedule emits onto
# the main loop. None before startup → emits become no-ops (safe for tests
# that don't start the app).
_APP_LOOP: Optional[asyncio.AbstractEventLoop] = None


def set_app_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Called by main.lifespan on startup."""
    global _APP_LOOP
    _APP_LOOP = loop


def get_app_loop() -> Optional[asyncio.AbstractEventLoop]:
    return _APP_LOOP


async def emit_download_progress(payload: DownloadProgress) -> None:
    """Broadcast a single download-progress event. Awaitable variant —
    use directly from async code (e.g., model_registry.download_model)."""
    await event_manager.broadcast("model.download_progress", payload.model_dump())


def schedule_emit_from_thread(payload: DownloadProgress) -> None:
    """Fire-and-forget emit from a worker thread.

    No-op if the loop hasn't been captured yet (e.g., during tests
    before app startup). Errors are swallowed — progress UI is
    best-effort and must never affect download correctness.
    """
    loop = _APP_LOOP
    if loop is None:
        return
    try:
        asyncio.run_coroutine_threadsafe(emit_download_progress(payload), loop)
    except RuntimeError:
        # Loop closed or stopped — ignore
        pass
