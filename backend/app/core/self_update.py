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
import re
import subprocess
import tempfile
from enum import Enum

from app.core.drain import set_draining
from app.core.events import event_manager
from app.core.logger import get_logger

logger = get_logger(__name__)

_IDLE_POLL_S = 3.0
_GIT_TIMEOUT_S = 120.0
_BUILD_TIMEOUT_S = 1800.0
_CHECK_INTERVAL_S = 1800.0  # periodic availability check (30 min)

# Same package set install-deps.sh excludes from its requirements install
# (split-stack: torch/torchvision/torchaudio/triton/triton-windows are baked
# into the image or installed manually in the local venv — never from
# requirements.txt). Used only by the install-deps.sh-missing fallback below.
_TORCH_STACK_RE = re.compile(
    r"^\s*(torch|torchvision|torchaudio|triton|triton-windows)(?=[\s=<>!~#]|$)",
    re.IGNORECASE,
)


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
    def _run_git(
        self, args: list[str], cwd: str | None = None, timeout: float | None = None
    ):
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
        rc, _out, err = self._run_git(
            ["ls-remote", "--heads", self.remote], timeout=20.0
        )
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

    # ── check ───────────────────────────────────────────────────────────
    async def check(self) -> dict:
        def _work():
            self._run_git(
                ["fetch", "--quiet", "origin", self.branch], timeout=_GIT_TIMEOUT_S
            )
            rc, out, _ = self._run_git(
                ["rev-list", "--count", f"HEAD..origin/{self.branch}"]
            )
            behind = int(out.strip()) if rc == 0 and out.strip().isdigit() else 0
            rc_l, out_l, _ = self._run_git(
                ["log", "--no-decorate", "--format=%s", f"HEAD..origin/{self.branch}"]
            )
            commits = (
                [ln for ln in out_l.splitlines() if ln.strip()] if rc_l == 0 else []
            )
            return behind, commits

        behind, commits = await asyncio.to_thread(_work)
        self.behind = behind
        self._broadcast()
        return {"behind": behind, "commits": commits}

    # ── apply ───────────────────────────────────────────────────────────
    def apply(self) -> None:
        """Kick off the update on a background task. Returns immediately."""
        if self.state not in (UpdateState.IDLE, UpdateState.ERROR):
            return
        if self._loop is None:
            return
        # Flip state synchronously here (not just inside the coroutine) so a
        # second near-simultaneous apply() sees a non-startable state and is
        # rejected — closes the double-spawn window. _apply_impl re-broadcasts.
        self.state = UpdateState.PULLING
        asyncio.run_coroutine_threadsafe(self._apply_impl(), self._loop)

    async def _apply_impl(self) -> None:
        try:
            self._set_state(UpdateState.PULLING)
            req_before = self._req_blob()
            if not await asyncio.to_thread(self._pull):
                self._set_state(
                    UpdateState.ERROR, error="git pull failed — see server log."
                )
                return

            self._set_state(UpdateState.BUILDING)
            # Reinstall Python deps when the pull changed requirements.txt —
            # otherwise a release adding a new dependency (e.g. json5 /
            # scenedetect / imageio-ffmpeg for video training) would pull the
            # code that imports it but never install it, crashing on restart.
            if self._req_blob() != req_before:
                await self._install_backend_deps()
            await self._build_frontend()

            self._set_state(UpdateState.PENDING_RESTART)
            set_draining(True)
            await self._wait_for_idle()

            self._set_state(UpdateState.RESTARTING)
            set_draining(False)
            await self._do_restart()
        except Exception as e:  # noqa: BLE001 — surface any failure as ERROR
            logger.error("self_update_failed", error=str(e))
            set_draining(False)
            self._set_state(UpdateState.ERROR, error=str(e))

    def _pull(self) -> bool:
        rc_f, _o, ef = self._run_git(["fetch", "--quiet", "origin", self.branch])
        if rc_f != 0:
            logger.error("self_update_fetch_failed", error=ef.strip()[:300])
            return False
        rc_r, _o2, er = self._run_git(["reset", "--hard", f"origin/{self.branch}"])
        if rc_r != 0:
            logger.error("self_update_reset_failed", error=er.strip()[:300])
            return False
        return True

    def _req_blob(self) -> str:
        """Git blob hash of backend/requirements.txt at HEAD (empty if unknown).
        Compared before/after a pull to decide whether deps must be reinstalled."""
        rc, out, _ = self._run_git(["rev-parse", "HEAD:backend/requirements.txt"])
        return out.strip() if rc == 0 else ""

    async def _install_backend_deps(self) -> None:
        """Reinstall backend Python deps after a pull that touched
        requirements.txt. Mirrors the Docker build via install-deps.sh (which
        handles the scenedetect --no-deps trap); falls back to a plain install
        if the script is absent. Container-only, so --break-system-packages
        matches the image build. Raises on failure → caller flags ERROR and
        skips the restart, leaving the still-running old process intact."""
        backend = os.path.join(self.app_dir, "backend")
        script = os.path.join(backend, "install-deps.sh")

        def _run():
            tmp_path: str | None = None
            if os.path.exists(script):
                cmd = ["bash", script]
            else:
                # install-deps.sh is missing (unexpected repo state) — fall
                # back to pip directly, but still filter out the torch-stack
                # lines: requirements.txt documents the LOCAL 2.12.1 trio, and
                # a plain `-r requirements.txt` here would clobber the 2.11.0
                # trio baked into this image's cached Dockerfile layer.
                req_path = os.path.join(backend, "requirements.txt")
                with open(req_path, encoding="utf-8") as f:
                    kept = [ln for ln in f if not _TORCH_STACK_RE.match(ln)]
                fd, tmp_path = tempfile.mkstemp(suffix=".txt", dir=backend)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.writelines(kept)
                cmd = [
                    "python", "-m", "pip", "install",
                    "--break-system-packages", "-r", tmp_path,
                ]
            try:
                proc = subprocess.run(
                    cmd, cwd=backend, capture_output=True, text=True,
                    timeout=_BUILD_TIMEOUT_S,
                )
            finally:
                if tmp_path is not None:
                    os.remove(tmp_path)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"backend dep install failed: {proc.stderr.strip()[:300]}"
                )

        logger.info("self_update_installing_backend_deps")
        await asyncio.to_thread(_run)

    async def _build_frontend(self) -> None:
        """Rebuild the SPA and sync it into the served dist dir.

        ng build emits to <app>/frontend/dist/frontend/browser, but the server
        serves <app>/frontend/browser (MRLN_FRONTEND_DIST). Rebuild, then replace
        the served dir with the fresh output."""
        fe = os.path.join(self.app_dir, "frontend")

        def _run(cmd: list[str], timeout: float):
            proc = subprocess.run(
                cmd, cwd=fe, capture_output=True, text=True, timeout=timeout
            )
            if proc.returncode != 0:
                raise RuntimeError(f"{cmd[0]} failed: {proc.stderr.strip()[:300]}")

        def _build():
            import shutil

            _run(["npm", "ci"], _BUILD_TIMEOUT_S)
            _run(
                ["npm", "run", "build", "--", "--configuration", "production"],
                _BUILD_TIMEOUT_S,
            )
            built = os.path.join(fe, "dist", "frontend", "browser")
            served = os.path.join(fe, "browser")
            if os.path.isdir(built):
                shutil.rmtree(served, ignore_errors=True)
                shutil.copytree(built, served)

        await asyncio.to_thread(_build)

    async def _wait_for_idle(self) -> None:
        """Wait until no in-process task is RUNNING. Training jobs are NOT
        awaited — they are restart-safe subprocesses."""
        while self.active_task_count() > 0:
            self._broadcast()
            await asyncio.sleep(_IDLE_POLL_S)

    async def _do_restart(self) -> None:
        from app.api.system_routes import _restart_server_logic

        await _restart_server_logic()

    # ── periodic check ──────────────────────────────────────────────────
    async def check_once_safe(self) -> None:
        """One periodic check; never raises (the loop depends on it)."""
        try:
            await self.check()
        except Exception as e:  # noqa: BLE001
            logger.warning("periodic_update_check_failed", error=str(e))

    async def run_periodic_check(self) -> None:
        """Background loop: re-check behind-count on an interval while available."""
        while True:
            if self.available:
                await self.check_once_safe()
            await asyncio.sleep(_CHECK_INTERVAL_S)


# Module singleton — configured at startup (lifespan) with the real env values.
self_update_service = SelfUpdateService(
    app_dir=os.environ.get("MRLN_APP_DIR", "/app"),
    branch=os.environ.get("MRLN_GIT_BRANCH", "main"),
    remote=os.environ.get("MRLN_GIT_REMOTE", ""),
)
