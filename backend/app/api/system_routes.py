"""System administration API — server restart and log management."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.core.logger import SERVER_LOG_PATH, get_logger
from app.core.self_update import self_update_service

# ``restart_launcher`` sits at backend/ rather than inside ``app`` because it
# runs BEFORE and AROUND the server and must never import the app (same reason,
# and the same anchored-append pattern, as ``port_resolver`` in
# ``app/core/container_config.py`` — see its comment for the measurement).
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BACKEND_DIR not in sys.path:
    sys.path.append(_BACKEND_DIR)

import restart_launcher  # noqa: E402 - requires the sys.path line above

router = APIRouter(prefix="/system", tags=["System"])
logger = get_logger(__name__)


# ── Response Models ──────────────────────────────────────────────────────


class MessageResponse(BaseModel):
    """Simple ``{"message": ...}`` acknowledgement."""

    message: str


class HealthResponse(BaseModel):
    """Lightweight health snapshot for the Server screen KPI rail."""

    status: str
    uptime_seconds: float
    model_count: int
    active_jobs: int


class VersionResponse(BaseModel):
    """Identity of this backend instance — version, and how it is deployed.

    ``container`` exists because the SPA cannot otherwise know: the Server
    screen's port field is authoritative locally and **ignored** in a
    container, where the port comes from argv or ``PORT`` and the host side of
    ``-p`` lives in the daemon, unreadable from inside. Without this the screen
    would offer an operator a control that silently does nothing.

    It rides here rather than on a new ``/system/deployment`` route
    deliberately: both are additive and D2-safe today, but a field we regret
    can be deprecated and left vestigial, while a public *route* we regret is
    frozen forever. Reserved in ECOSYSTEM §6.
    """

    version: str
    container: bool


class UpdateStatusResponse(BaseModel):
    """Current git/version info + update-state, mirrors
    ``SelfUpdateService.status_payload()`` exactly (also broadcast verbatim
    over the ``update.status`` WebSocket event)."""

    state: str
    available: bool
    branch: str | None = None
    commit: str | None = None
    dirty: bool
    is_repo: bool
    behind: int | None = None
    active: int
    error: str | None = None


class UpdateCheckResponse(BaseModel):
    """How many commits behind ``origin/<branch>`` we are + their subjects."""

    behind: int
    commits: list[str]


_LOG_FILE = SERVER_LOG_PATH

# Captured at import (≈ process start). A graceful restart spawns a fresh
# process that re-imports this module, so uptime resets — which is correct.
_BOOT_TIME = time.time()


def report_pending_restart_failure() -> dict | None:
    """Surface a previous restart's failure in the app's own log, once.

    A restart that fails leaves nobody running to tell anyone: the console this
    process was watching is gone with the process, and ``server.log`` is
    unlinked by ``setup_logging`` on the next startup. So the launcher's record
    outlives it in ``restart.log``, and the first server to start afterwards —
    which is the user's recovery, ``start_backend.bat`` — says it out loud in
    the log the Server screen already shows.

    Returns the record it reported, or None. Never raises: this runs on the
    startup path, and a diagnostic may never be a reason not to start.
    """
    try:
        pending = restart_launcher.pending_failure()
        if pending is None:
            return None
        logger.error(
            "previous_restart_failed",
            restart_log=str(restart_launcher.RESTART_LOG_PATH),
            # ``timestamp`` is renamed, never forwarded: the log pipeline stamps
            # its own, so passing the record's through silently loses WHEN the
            # restart failed — measured on the live run that verified this path.
            failed_at=pending.get("timestamp"),
            **{k: v for k, v in pending.items()
               if k not in ("level", "service", "event", "timestamp")},
        )
        restart_launcher.mark_reported()
        return pending
    except Exception as e:  # noqa: BLE001 - see docstring
        logger.warning("restart_report_failed", error=str(e))
        return None


async def _restart_server_logic() -> None:
    """Hand this server's port to a replacement, through the launcher.

    We do NOT spawn the replacement directly any more (LANE-51). Doing so meant
    (a) its output went to DEVNULL, so a start that failed was invisible in the
    console, in the logs and in the app, and (b) this process slept 1.0 s and
    exited, overlapping the two servers on one port by construction.

    ``restart_launcher`` waits for this process to release the port, resolves
    the port afresh (settings may have moved it since this process launched),
    starts the replacement with the restart log as its stdout/stderr, and
    watches it until it serves or dies. See that module's docstring.
    """
    await asyncio.sleep(2.0)  # Let the HTTP response flush

    orig_args = getattr(sys, "orig_argv", [sys.executable] + sys.argv)
    replacement = [sys.executable] + list(orig_args[1:])
    cmd = [
        sys.executable,
        restart_launcher.__file__,
        "--old-pid", str(os.getpid()),
        "--",
        *replacement,
    ]

    try:
        # A FILE, never our own stdio: our handles may be a dead pipe (the IDE
        # terminal that launched the original server is gone), and a full pipe
        # blocks console logging while holding the logging lock — freezing the
        # whole event loop (e2e3cfc8, live incident 2026-07-16). A file has no
        # reader that can stall it, so the property is kept while the output
        # survives. Even the launcher's own traceback lands here.
        log_handle = restart_launcher.open_log()
    except OSError as e:
        # A restart we could not observe is a restart we do not perform: this
        # server keeps serving and says why, which is recoverable. Spawning
        # blind is what LANE-51 was.
        logger.error("restart_log_unavailable", error=str(e),
                     path=str(restart_launcher.RESTART_LOG_PATH))
        return

    try:
        restart_env = os.environ.copy()
        restart_env["MRLN_RESTART"] = "1"
        subprocess.Popen(
            cmd,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            close_fds=True,
            cwd=os.getcwd(),
            env=restart_env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=log_handle,
        )
    except OSError as e:
        logger.error("restart_spawn_failed", error=str(e))
        return
    finally:
        # The child holds its own duplicate of the handle.
        log_handle.close()

    # The last thing THIS process says. The console does not go quiet after it:
    # the launcher narrates the restart to the same terminal until the
    # replacement serves (LANE-56 — telling the user the console was about to go
    # quiet was accurate and was still the wrong answer, because a silent wait
    # of minutes is one he ends).
    logger.info(
        "restart_handoff",
        restart_log=str(restart_launcher.RESTART_LOG_PATH),
        message=("handing over to the restart launcher — it reports progress on this "
                 "terminal until the replacement is serving, and the full output of "
                 "its start is written to restart.log"),
    )

    # Not a port handoff (the launcher waits for the port itself) — only a
    # window for that last line to reach the queued WebSocket log mirror.
    # OFF the event loop, and that is the point (LANE-56): taken as
    # `await asyncio.sleep(0.25)` this window was MEASURED at 26 s, because the
    # loop had other work and did not come back — 26 s of a port this process
    # no longer wants, with the launcher and the user both waiting on it.
    restart_launcher.schedule_exit(0.25)


@router.post("/restart", response_model=MessageResponse)
async def restart_server(background_tasks: BackgroundTasks):
    """Trigger a graceful server restart."""
    background_tasks.add_task(_restart_server_logic)
    return {"message": "Server restart initiated. Connection will be lost temporarily."}


@router.post("/logs/clear", response_model=MessageResponse)
async def clear_logs():
    """Truncate the server log file."""

    def _truncate():
        with open(_LOG_FILE, "w") as f:
            f.truncate(0)

    try:
        await asyncio.to_thread(_truncate)
        logger.info("logs_cleared")
        return {"message": "Server logs cleared."}
    except OSError as e:
        logger.error("log_clear_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to clear logs: {e}")


@router.get("/logs")
async def get_logs(lines: int = 100):
    """Return the last *lines* lines of the server log."""
    if not _LOG_FILE.exists():
        return []

    def _read_tail() -> list[str]:
        try:
            with open(_LOG_FILE, "r", encoding="utf-8") as f:
                content = f.readlines()
                return content[-lines:]
        except OSError as e:
            logger.error("read_logs_failed", error=str(e))
            return []

    return await asyncio.to_thread(_read_tail)


# ── System & GPU Status ─────────────────────────────────────────────────


@router.get("/version", response_model=VersionResponse)
async def get_version():
    """Backend application version for the sidebar footer.

    Lives under ``/api`` so it is reachable through the dev-server proxy and
    the production SPA mount alike — unlike the backend root ``/``, which both
    of those serve as ``index.html``.

    Also reports whether this instance runs in a container, which the Server
    screen needs to explain that its port field is not the authority there.
    The flag is IMPORTED from ``container_config`` rather than re-derived from
    the environment: a second ``os.environ`` read would be a second producer of
    a fact that already has one (RULE-21), free to drift from the resolver that
    actually decides the port.
    """
    from app import __version__
    from app.core import container_config

    return {"version": __version__, "container": container_config.is_container()}


@router.get("/health", response_model=HealthResponse)
async def get_health():
    """Lightweight health snapshot for the Server screen KPI rail.

    A 200 response inherently means the backend is up, so ``status`` is
    always ``"healthy"`` here; the client downgrades it to "Offline" off the
    live WebSocket connection. Also returns process uptime, the number of
    loaded model definitions, and the count of in-flight (running or paused)
    training jobs.
    """
    from app.core.job import JobStatus
    from app.core.job_manager import job_manager
    from app.engine.models.registry import registry

    active_jobs = sum(
        1
        for j in job_manager.list_jobs()
        if j.status in (JobStatus.RUNNING, JobStatus.PAUSED)
    )
    return {
        "status": "healthy",
        "uptime_seconds": max(0.0, time.time() - _BOOT_TIME),
        "model_count": registry.count(),
        "active_jobs": active_jobs,
    }


# ── Self-Update ──────────────────────────────────────────────────────────


@router.get("/update/status", response_model=UpdateStatusResponse)
async def get_update_status():
    """Current git/version info + update-state for the Server screen + top-bar."""
    return self_update_service.status_payload()


@router.post("/update/check", response_model=UpdateCheckResponse)
async def check_update():
    """Fetch and report how many commits behind origin/<branch> we are."""
    if not self_update_service.available:
        raise HTTPException(status_code=403, detail="Self-update is not available.")
    return await self_update_service.check()


@router.post("/update/apply", response_model=MessageResponse)
async def apply_update():
    """Pull → build → drain → restart-when-idle. Returns immediately."""
    if not self_update_service.available:
        raise HTTPException(status_code=403, detail="Self-update is not available.")
    self_update_service.apply()
    return {"message": "Update started. Watch the update.status events for progress."}


# ── GPU plugin models: what is resident, and free it on demand ────────────
#
# The three GPU-plugin services are a closed set (they are exactly the classes
# that route through `core/gpu_unload.unload_gpu_plugins`). Each row is
# (wire id, human label, module, class, active-key attribute).
#
# The wire id is NOT the class name: it is a frozen public id (ECOSYSTEM §6),
# so renaming `CaptionService` must not rename `caption` on the wire.
#
# The trainer is deliberately absent: it runs in its own process and frees its
# VRAM on exit, so there is nothing this process could unload for it.
_GPU_SERVICES: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "caption",
        "Captioning",
        "app.core.captioning.caption_service",
        "CaptionService",
        "_active_model_key",
    ),
    (
        "masking",
        "Masking",
        "app.core.masking.masking_service",
        "MaskingService",
        "_active_model_id",
    ),
    (
        "scoring",
        "Scoring",
        "app.core.scoring.scoring_service",
        "ScoringService",
        "_active_model_id",
    ),
)


class GpuServiceState(BaseModel):
    """One GPU-plugin service's residency. ``model`` is display-only — the
    three services own their key formats independently, so no client may
    branch on its shape."""

    service: str
    label: str
    loaded: bool
    model: str | None = None


class GpuSkipped(BaseModel):
    """A service the unload left alone, and the reason to show the user."""

    service: str
    reason: str


class GpuLoadedResponse(BaseModel):
    """Which GPU-plugin services hold a model right now.

    ``any_loaded`` is precomputed rather than left to the client as
    ``services.some(...)`` (ARCHITECTURE D10, "compute at write time"): it is
    the single boolean the topbar's positive-only control gates on, so a client
    that does not yet know about a newly added service still hides the button
    correctly.
    """

    any_loaded: bool
    services: list[GpuServiceState]


class GpuUnloadResponse(GpuLoadedResponse):
    """Result of a global unload: what was actually freed, and what was not.

    ``any_loaded`` / ``services`` are re-read AFTER the unload, so the client
    never has to guess; a service in ``skipped`` will still show ``loaded``.
    """

    unloaded: list[str]
    skipped: list[GpuSkipped]


def _gpu_service_class(module_path: str, class_name: str):
    """Import a GPU-plugin service class LAZILY.

    Never at module import: these modules pull in torch and the plugin stacks,
    and nothing imported at startup may raise (ARCHITECTURE D1). Importing does
    NOT construct the singleton, so this cannot load a model.
    """
    import importlib

    return getattr(importlib.import_module(module_path), class_name)


def _gpu_snapshot() -> GpuLoadedResponse:
    """Read residency from the CLASS-level active-key attributes.

    Cheap and side-effect free by construction: it reads a class attribute and
    never calls ``get_instance()``, so a service whose singleton was never
    constructed reports "nothing loaded" without constructing one — and no code
    path here can load a model in order to answer whether a model is loaded.
    """
    states: list[GpuServiceState] = []
    for service_id, label, module_path, class_name, active_attr in _GPU_SERVICES:
        cls = _gpu_service_class(module_path, class_name)
        key = getattr(cls, active_attr, None)
        states.append(
            GpuServiceState(
                service=service_id,
                label=label,
                loaded=bool(key),
                model=str(key) if key else None,
            )
        )
    return GpuLoadedResponse(
        any_loaded=any(s.loaded for s in states),
        services=states,
    )


def _gpu_unload_all() -> GpuUnloadResponse:
    """Unload every GPU-plugin service that is not busy, then re-read state.

    Every service is asked in its ``skip_if_batch_active=True`` mode, whose
    check-then-act is closed under that service's ``_unload_lock`` — so a batch
    that starts concurrently either loses the race and is not started until the
    unload completes, or wins it and the unload is skipped. Without that mode
    this route would free a model out from under a running batch.
    """
    unloaded: list[str] = []
    skipped: list[GpuSkipped] = []

    for service_id, label, module_path, class_name, active_attr in _GPU_SERVICES:
        cls = _gpu_service_class(module_path, class_name)
        was_loaded = bool(getattr(cls, active_attr, None))
        ran = cls.unload_models(skip_if_batch_active=True)
        if not ran:
            skipped.append(
                GpuSkipped(
                    service=service_id,
                    reason=f"{label.lower()} is busy — a batch task is using the model",
                )
            )
        elif was_loaded:
            unloaded.append(service_id)

    after = _gpu_snapshot()
    return GpuUnloadResponse(
        any_loaded=after.any_loaded,
        services=after.services,
        unloaded=unloaded,
        skipped=skipped,
    )


@router.get("/gpu/loaded", response_model=GpuLoadedResponse)
async def get_gpu_loaded():
    """Which GPU-plugin services currently hold a model. Loads nothing."""
    return _gpu_snapshot()


@router.post("/gpu/unload", response_model=GpuUnloadResponse)
async def unload_gpu_models():
    """Free VRAM held by the caption/masking/scoring models, on user request.

    Runs in a worker thread: the unload does ``gc.collect()`` +
    ``torch.cuda.synchronize()`` + ``empty_cache()``, which blocks for as long
    as the driver needs, and must not stall the event loop.
    """
    return await asyncio.to_thread(_gpu_unload_all)
