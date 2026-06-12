"""Self-update service: pull /app to origin/<branch>, rebuild, restart when idle.

Only meaningful in a deployed container where /app is a real git checkout.
A startup probe (`git ls-remote`, unauthenticated) decides whether the feature
is exposed (`available`) — true once the repo is public. Training jobs are
restart-safe subprocesses and are intentionally NOT awaited before restart;
only in-process task_manager work blocks the restart.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from enum import Enum

from app.core.drain import set_draining
from app.core.events import event_manager
from app.core.logger import get_logger

logger = get_logger(__name__)

_IDLE_POLL_S = 3.0
_GIT_TIMEOUT_S = 120.0
_BUILD_TIMEOUT_S = 1800.0


class UpdateState(str, Enum):
    IDLE = "idle"
    PULLING = "pulling"
    BUILDING = "building"
    PENDING_RESTART = "pending_restart"
    RESTARTING = "restarting"
    ERROR = "error"


# Indirection so tests can monkeypatch the manager query without importing the
# heavy singleton at module import time.
def _list_tasks():
    from app.core.tasks.task_manager import task_manager
    return task_manager.list()


class SelfUpdateService:
    def __init__(self, *, app_dir: str, branch: str, remote: str) -> None:
        self.app_dir = app_dir
        self.branch = branch
        self.remote = remote
        self.state = UpdateState.IDLE
        self.available = False
        self.error: str | None = None
        self.behind: int | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        self._loop = loop

    # ── git plumbing ────────────────────────────────────────────────────
    def _run_git(self, args: list[str], cwd: str | None = None, timeout: float | None = None):
        """Run `git <args>`; return (returncode, stdout, stderr). Never raises
        on non-zero exit. Credentials disabled so a private remote fails fast."""
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=cwd or self.app_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout or _GIT_TIMEOUT_S,
            )
            return (proc.returncode, proc.stdout, proc.stderr)
        except (OSError, subprocess.TimeoutExpired) as e:
            logger.error("git_run_failed", args=args, error=str(e))
            return (1, "", str(e))

    def git_status(self) -> dict:
        rc_b, out_b, _ = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        if rc_b != 0:
            return {"is_repo": False, "branch": None, "commit": None, "dirty": False}
        rc_c, out_c, _ = self._run_git(["rev-parse", "--short", "HEAD"])
        rc_s, out_s, _ = self._run_git(["status", "--porcelain"])
        return {
            "is_repo": True,
            "branch": out_b.strip(),
            "commit": out_c.strip() if rc_c == 0 else None,
            "dirty": bool(out_s.strip()) if rc_s == 0 else False,
        }

    def probe_availability(self) -> None:
        """Set `available` from an unauthenticated ls-remote. Run once at startup."""
        rc, _out, err = self._run_git(["ls-remote", "--heads", self.remote], timeout=20.0)
        self.available = rc == 0
        if not self.available:
            logger.info("self_update_unavailable", reason=err.strip()[:200])

    def active_task_count(self) -> int:
        """In-process task_manager tasks that a restart would kill. Training
        jobs are NOT counted — they survive the restart."""
        from app.core.tasks.task import TaskStatus
        return sum(1 for t in _list_tasks() if t.status == TaskStatus.RUNNING)

    def status_payload(self) -> dict:
        st = self.git_status()
        return {
            "state": self.state.value,
            "available": self.available,
            "branch": st["branch"],
            "commit": st["commit"],
            "dirty": st["dirty"],
            "is_repo": st["is_repo"],
            "behind": self.behind,
            "active": self.active_task_count(),
            "error": self.error,
        }

    def _broadcast(self) -> None:
        if self._loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                event_manager.broadcast("update.status", self.status_payload()),
                self._loop,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("update_status_broadcast_failed", error=str(e))

    def _set_state(self, state: UpdateState, *, error: str | None = None) -> None:
        self.state = state
        self.error = error
        self._broadcast()
