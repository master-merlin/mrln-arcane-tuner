"""Unified download-progress event channel.

Producers (curated `model_registry.download_model` and the Hugging Face
retrofit callsites) emit `model.download_progress` events via the shared
`event_manager` WS bus. Frontend consumers key downloads by
`(source, model_id)`.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time as _time
from contextlib import contextmanager
from typing import Generator, Literal, Optional

import structlog
from huggingface_hub.utils import tqdm as hf_tqdm  # type: ignore[attr-defined]
from pydantic import BaseModel, Field, model_validator

from app.core.events import event_manager

logger = structlog.get_logger(__name__)

# How often the snapshot byte-progress poller samples on-disk cache growth.
SNAPSHOT_POLL_INTERVAL_S = 0.5


class FileProgress(BaseModel):
    """One in-flight file inside a snapshot download."""
    name: str
    current_bytes: int = 0
    total_bytes: Optional[int] = None
    percent: Optional[int] = None

    @model_validator(mode="after")
    def _compute_percent(self) -> "FileProgress":
        if self.percent is None and self.total_bytes and self.total_bytes > 0:
            self.percent = int(self.current_bytes * 100 / self.total_bytes)
        return self


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
    files: list[FileProgress] = Field(default_factory=list)


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
    files: Optional[list["FileProgress"]] = None,
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
        files=files or [],
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


# ── Snapshot byte-progress poller ────────────────────────────────────────────
#
# huggingface_hub routes a caller's ``tqdm_class`` ONLY to the coarse
# "Fetching N files" bar, never the per-file byte transfers (its own docstring
# says so). Wrapping that bar yields file-granularity updates that look frozen
# while a single multi-GB shard downloads (and report file *counts* rendered as
# "0.0 MB"). So for a full-repo snapshot we ignore tqdm entirely and instead
# poll the repo's on-disk cache growth against its total size — which also
# reports a *partial resume* correctly (on-disk already includes the cached
# bytes). All helpers are best-effort: any failure degrades to an indeterminate
# spinner rather than breaking the download.


def _repo_cache_dir(repo_id: str) -> Optional[str]:
    """Local HF cache folder for *repo_id* (…/models--org--name), or None."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        folder = "models--" + repo_id.replace("/", "--")
        return os.path.join(HF_HUB_CACHE, folder)
    except Exception:
        return None


def _on_disk_bytes(repo_cache_dir: Optional[str]) -> int:
    """Bytes currently on disk for a repo — the ``blobs/`` dir (completed blobs
    plus in-flight ``*.incomplete`` parts). 0 if the dir doesn't exist yet."""
    if not repo_cache_dir:
        return 0
    blobs = os.path.join(repo_cache_dir, "blobs")
    total = 0
    try:
        with os.scandir(blobs) as it:
            for entry in it:
                try:
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat().st_size
                except OSError:
                    continue
    except (FileNotFoundError, NotADirectoryError):
        return 0
    except OSError:
        return 0
    return total


def _repo_total_bytes(repo_id: str) -> Optional[int]:
    """Sum of every file's size in *repo_id* (one metadata call), or None when
    unknown (offline, missing sizes) — the caller then shows an indeterminate
    spinner instead of a wrong percentage."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().repo_info(repo_id, files_metadata=True)
        sizes = [
            s.size for s in (info.siblings or []) if getattr(s, "size", None)
        ]
        return sum(sizes) if sizes else None
    except Exception as e:
        logger.debug("repo_total_bytes_failed", repo=repo_id, error=str(e))
        return None


class SnapshotProgressRegistry:
    """Thread-safe collector of per-file byte progress for one snapshot
    download. Populated by ``_PerFileTqdm`` (worker threads) and read by the
    poller. Holds state only — it never emits."""

    def __init__(self, total: Optional[int]):
        self.total = total
        self._lock = threading.Lock()
        self._active: dict[str, tuple[int, Optional[int]]] = {}

    def update(self, name: str, current: int, total: Optional[int]) -> None:
        with self._lock:
            self._active[name] = (current, total)

    def done(self, name: str) -> Optional[int]:
        """Mark *name* finished, return its size, and log one concise line."""
        with self._lock:
            entry = self._active.pop(name, None)
        size = entry[1] if entry else None
        if entry is not None:
            logger.info("downloaded_file", file=name, size_bytes=size)
        return size

    def snapshot(self) -> list["FileProgress"]:
        with self._lock:
            items = sorted(self._active.items())  # stable order by name
        out: list[FileProgress] = []
        for name, (cur, tot) in items:
            pct = int(cur * 100 / tot) if (tot and tot > 0) else None
            out.append(
                FileProgress(
                    name=name, current_bytes=cur, total_bytes=tot, percent=pct,
                )
            )
        return out


@contextmanager
def _capture_per_file(
    registry: SnapshotProgressRegistry,
) -> Generator[None, None, None]:
    """Temporarily route HF's per-file byte bars into *registry*.

    huggingface_hub builds each per-file download bar via
    ``_get_progress_bar_context`` → the module-global ``tqdm`` in the
    ``huggingface_hub.utils.tqdm`` submodule. We swap that name for an
    emitting subclass for the duration of a download and restore it in
    ``finally``. (Note: ``import huggingface_hub.utils.tqdm`` binds the tqdm
    *class*, not the module — the package re-exports the class under that
    name — so we reach the real module through ``sys.modules``.) The outer
    "Fetching N files" bar uses unit "files" (not "B"), so it is ignored.
    Best-effort: if the module/attr is absent on some future HF version, the
    hook no-ops and only aggregate progress is shown.
    """
    import sys

    import huggingface_hub.utils.tqdm  # noqa: F401  (registers the submodule)

    mod = sys.modules.get("huggingface_hub.utils.tqdm")
    original = getattr(mod, "tqdm", None) if mod is not None else None
    if mod is None or original is None:
        yield
        return

    class _PerFileTqdm(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._reg_name: Optional[str] = None
            # Only per-file byte transfers (unit="B" + a filename desc) count;
            # the file-count bar and any non-download bar are ignored.
            if getattr(self, "unit", "") == "B" and getattr(self, "desc", ""):
                self._reg_name = self.desc
                registry.update(self._reg_name, int(self.n or 0), self.total)

        def update(self, n: int = 1) -> "bool | None":
            ret = super().update(n)
            if self._reg_name:
                registry.update(self._reg_name, int(self.n or 0), self.total)
            return ret

        def close(self) -> None:
            if self._reg_name:
                registry.done(self._reg_name)
            super().close()

    mod.tqdm = _PerFileTqdm
    try:
        yield
    finally:
        mod.tqdm = original


def _progress_signature(p: DownloadProgress) -> tuple:
    """A coarse fingerprint of a payload — used to drop unchanged ticks so the
    poller emits only on meaningful movement (integer-percent granularity)."""
    return (p.percent, tuple((f.name, f.percent) for f in p.files))


@contextmanager
def snapshot_byte_progress(
    *, repo_id: str, model_id: str, category: str,
) -> Generator[None, None, None]:
    """Emit aggregate + per-file BYTE progress for an in-flight snapshot.

    The aggregate comes from polling on-disk cache growth (resume-aware) against
    the repo's total size; the per-file list comes from ``_capture_per_file``.
    The poller is the SINGLE emitter — it sends one consolidated event per tick,
    and only when the fingerprint changed (``starting``/``complete``/``error``
    always fire). Per-file tqdm churn produces no WS frames of its own.

    Best-effort: metadata failure → indeterminate aggregate; the hook no-ops on
    HF-internals drift; the original tqdm is always restored.
    """
    total = _repo_total_bytes(repo_id)
    cache_dir = _repo_cache_dir(repo_id)
    registry = SnapshotProgressRegistry(total)
    last_sig: dict = {"v": object()}  # sentinel: forces first downloading emit

    def _payload(status: str, *, error: Optional[str] = None) -> DownloadProgress:
        current = _on_disk_bytes(cache_dir)
        if total is not None:
            current = min(current, total)
        return _make_payload(
            source="hf", model_id=model_id, category=category, status=status,
            current=current, total=total, error=error, files=registry.snapshot(),
        )

    def _emit(status: str, *, error: Optional[str] = None, force: bool = False) -> None:
        p = _payload(status, error=error)
        sig = _progress_signature(p)
        if force or sig != last_sig["v"]:
            last_sig["v"] = sig
            schedule_emit_from_thread(p)

    _emit("starting", force=True)
    stop = threading.Event()

    def _poll() -> None:
        while not stop.wait(SNAPSHOT_POLL_INTERVAL_S):
            _emit("downloading")

    poller = threading.Thread(target=_poll, daemon=True, name="hf_dl_progress")
    with _capture_per_file(registry):
        poller.start()
        try:
            yield
        except Exception as exc:
            stop.set()
            _emit("error", error=str(exc), force=True)
            raise
        else:
            stop.set()
            _emit("complete", force=True)
            logger.info("snapshot_complete", repo=repo_id, total_bytes=total)
        finally:
            stop.set()
            poller.join(timeout=1.0)
