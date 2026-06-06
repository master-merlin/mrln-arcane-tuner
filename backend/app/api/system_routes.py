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
        "model_count": len(registry._definitions),
        "active_jobs": active_jobs,
    }


@router.get("/status")
async def get_system_status():
    """Full system + GPU health snapshot."""
    from app.core.system_monitor import system_monitor
    return system_monitor.snapshot().to_dict()


@router.get("/gpu")
async def get_gpu_status():
    """GPU-only snapshot (VRAM, temp, power, clocks, utilization)."""
    from app.core.system_monitor import system_monitor
    snap = system_monitor.snapshot()
    return {"gpus": [g.to_dict() for g in snap.gpus]}
