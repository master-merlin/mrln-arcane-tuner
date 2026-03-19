"""System administration API — server restart and log management."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.core.logger import get_logger

router = APIRouter(prefix="/system", tags=["System"])
logger = get_logger(__name__)

_LOG_FILE = Path("server.log")


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


@router.post("/restart")
async def restart_server(background_tasks: BackgroundTasks):
    """Trigger a graceful server restart."""
    background_tasks.add_task(_restart_server_logic)
    return {"message": "Server restart initiated. Connection will be lost temporarily."}


@router.post("/logs/clear")
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
