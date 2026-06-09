"""Unified download-progress event channel.

Producers (curated `model_registry.download_model` and the Hugging Face
retrofit callsites) emit `model.download_progress` events via the shared
`event_manager` WS bus. Frontend consumers key downloads by
`(source, model_id)`.
"""
from __future__ import annotations

import asyncio
import time as _time
from contextlib import contextmanager
from typing import Generator, Literal, Optional

from huggingface_hub.utils import tqdm as hf_tqdm  # type: ignore[attr-defined]
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


# ── HF tqdm subclass + context manager ───────────────────────────────────────

def _make_payload(
    *,
    source: str,
    model_id: str,
    category: str,
    status: str,
    current: int,
    total: Optional[int],
    error: Optional[str] = None,
) -> DownloadProgress:
    percent = (int(current * 100 / total) if (total and total > 0) else None)
    return DownloadProgress(
        source=source,  # type: ignore[arg-type]
        model_id=model_id,
        category=category,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        current_bytes=current,
        total_bytes=total,
        percent=percent,
        error=error,
    )


class WSProgressTqdm(hf_tqdm):
    """Drop-in tqdm subclass for `huggingface_hub`'s `tqdm_class=` parameter.

    Emits a 'starting' event on init, throttled 'downloading' events on
    `update()`, and a 'complete' event on `close()`. Errors during the
    download are reported by the surrounding `with_progress` context manager
    — not from inside the tqdm itself, since exceptions surface in the
    caller, not in the bar.
    """

    def __init__(
        self,
        *args,
        source: str = "hf",
        model_id: str = "",
        category: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._meta_source = source
        self._meta_model_id = model_id or (kwargs.get("desc") or "unknown")
        self._meta_category = category
        self._rate = RateLimiter()
        self._closed_emit_sent = False
        self._update_called = False
        schedule_emit_from_thread(
            _make_payload(
                source=source,
                model_id=self._meta_model_id,
                category=category,
                status="starting",
                current=0,
                total=self.total,
            )
        )

    def update(self, n: int = 1) -> "bool | None":
        self._update_called = True
        ret = super().update(n)
        total = self.total
        percent = (int(self.n * 100 / total) if (total and total > 0) else None)
        if self._rate.allow("downloading", percent):
            schedule_emit_from_thread(
                _make_payload(
                    source=self._meta_source,
                    model_id=self._meta_model_id,
                    category=self._meta_category,
                    status="downloading",
                    current=int(self.n),
                    total=total,
                )
            )
        return ret

    def close(self) -> None:
        if not self._closed_emit_sent and self._update_called:
            self._closed_emit_sent = True
            schedule_emit_from_thread(
                _make_payload(
                    source=self._meta_source,
                    model_id=self._meta_model_id,
                    category=self._meta_category,
                    status="complete",
                    current=int(self.n),
                    total=self.total,
                )
            )
        super().close()


def make_progress_tqdm(
    *, source: str = "hf", model_id: str = "", category: str = "",
) -> type["WSProgressTqdm"]:
    """Return a ``WSProgressTqdm`` SUBCLASS with the WS metadata pre-bound.

    Pass the result as ``snapshot_download(tqdm_class=...)``.

    We bind the metadata with a real subclass rather than
    ``functools.partial`` because ``huggingface_hub`` downloads snapshot files
    concurrently via ``tqdm.contrib.concurrent.thread_map``, which calls the
    **classmethod** ``tqdm_class.get_lock()``. A ``functools.partial`` does not
    expose inherited classmethods — that path raises
    ``'functools.partial' object has no attribute 'get_lock'`` and aborts the
    download. A subclass inherits ``get_lock`` / ``set_lock`` and works.
    """

    class _BoundProgressTqdm(WSProgressTqdm):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("source", source)
            kwargs.setdefault("model_id", model_id)
            kwargs.setdefault("category", category)
            super().__init__(*args, **kwargs)

    return _BoundProgressTqdm


def _is_repo_cached(repo_id: str) -> bool:
    """Best-effort: True when the repo's ``config.json`` is already in the HF
    cache, i.e. the model was downloaded before and loading it won't hit the
    network. Used to suppress a spurious download-bar flash on a pure cache hit
    (loading cached shards from disk into VRAM is not a download)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(repo_id, "config.json"), str)
    except Exception:
        return False


@contextmanager
def with_progress(
    *, model_id: str, category: str, repo_id: str | None = None,
) -> Generator[None, None, None]:
    """Wrap an HF download callsite to ensure starting/complete/error events fire.

    When the HF library does its own download, `WSProgressTqdm` (if passed
    via `tqdm_class`) emits its own starting/downloading/complete events.
    This context manager guarantees a starting/complete pair even when no
    tqdm was created (cache hit) and emits an `error` event if the body
    raises — the tqdm subclass cannot observe caller exceptions.

    Note: when both the context manager and `WSProgressTqdm` fire, the
    frontend store deduplicates by `(source, model_id)` and treats the
    later state as authoritative.

    Pass ``repo_id`` to suppress the start/complete pair when that repo is
    already fully cached — no download happens, so flashing the download
    indicator would be misleading. Callers that can't cheaply know the repo
    (e.g. single-file fetches) omit it and keep the old always-emit behavior.
    """
    if repo_id and _is_repo_cached(repo_id):
        yield
        return

    payload_kw: dict = dict(source="hf", model_id=model_id, category=category)
    schedule_emit_from_thread(
        _make_payload(**payload_kw, status="starting", current=0, total=None)
    )
    try:
        yield
    except Exception as exc:
        schedule_emit_from_thread(
            _make_payload(
                **payload_kw, status="error", current=0, total=None, error=str(exc)
            )
        )
        raise
    else:
        schedule_emit_from_thread(
            _make_payload(**payload_kw, status="complete", current=0, total=None)
        )
