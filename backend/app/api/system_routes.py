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
    """Backend application version (shown in the sidebar footer)."""

    version: str


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


async def _restart_server_logic() -> None:
    """Restart the server on Windows by spawning a new process."""
    await asyncio.sleep(2.0)  # Let the HTTP response flush

    orig_args = getattr(sys, "orig_argv", [sys.executable] + sys.argv)
    cmd = [sys.executable] + list(orig_args[1:])

    try:
        restart_env = os.environ.copy()
        restart_env["MRLN_RESTART"] = "1"
        subprocess.Popen(
            cmd,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            close_fds=True,
            cwd=os.getcwd(),
            env=restart_env,
            # Never inherit our stdio: it may be a dead pipe (e.g. the IDE
            # terminal that launched the original server is gone), and a full
            # pipe blocks console logging while holding the logging lock —
            # freezing the whole event loop. File/WS logging is unaffected.
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as e:
        logger.error("restart_spawn_failed", error=str(e))
        return

    await asyncio.sleep(1.0)
    os._exit(0)


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
    """
    from app import __version__

    return {"version": __version__}


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
