"""A cross-process lock for the tests that must have the machine to themselves.

``@pytest.mark.xdist_group("gpu")`` serialises a group against itself inside
ONE pytest run (``--dist loadgroup`` sends the whole group to one worker). It
says nothing about a second pytest — another worktree's gate, another agent's
run — on the same box, and the live GPU, the loopback ports, are per-machine.
This lock closes that harness-vs-harness window (LANE-63 step 4; adversarial
finding 3).

Design, item by item:

* **path** — ``_harness/.<group>.lock``. ``_harness/`` is junction-shared into
  every worktree, so every tree contends on the SAME file; a per-tree path
  would be a lock in name only. A clone without ``_harness`` (CI, a public
  checkout) falls back to the machine's temp dir, which is still one file per
  box. ``MRLN_MACHINE_LOCK_DIR`` overrides the directory (the lock's own tests
  use it so they never contend with a real run). Anchored on ``__file__``,
  never the CWD.
* **acquisition** — ``os.open(O_CREAT | O_EXCL)``: atomic on NTFS and POSIX.
  The winner writes its owner metadata (pid, xdist worker id, worktree, ISO
  timestamp) into the file before returning, i.e. before any GPU read.
* **stale recovery** — on ``FileExistsError`` the metadata is read; a dead
  owner pid (``psutil.pid_exists``, never ``os.kill(pid, 0)``, which on Windows
  TERMINATES the process) means the file is removed and the open retried. An
  unreadable file (the winner between ``open`` and ``write``) counts as live
  for ``UNREADABLE_GRACE_S`` and stale after. A live owner is never broken.
* **timeout** — a bounded wait, then ``MachineLockTimeout`` naming the holder.
  The pytest side turns that into a FAILURE, never a skip (a silent skip is a
  gate hole) and never a proceed (proceeding is the collision the lock exists
  to stop).
* **release** — in ``finally``, and only if the file still carries OUR pid:
  a stale-recovery by another process may have replaced the file with its own.

Honest limit: cooperative only. The user's live backend and non-harness
processes do not honour it; for those the operational boundary in the plan
(never beside a training run, never while a UAT pack is open) is the control.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from tests.support.worker_paths import current_worker

LOCK_DIR_ENV = "MRLN_MACHINE_LOCK_DIR"
LOCK_TIMEOUT_ENV = "MRLN_MACHINE_LOCK_TIMEOUT"
DEFAULT_TIMEOUT_S = 120.0
POLL_S = 0.1
# A file that exists but cannot be parsed is a winner mid-write for this long;
# after it, it is a crashed writer and is recovered like a dead pid.
UNREADABLE_GRACE_S = 10.0

_REPO_ROOT = Path(__file__).resolve().parents[3]  # backend/tests/support/x.py → repo


class MachineLockTimeout(RuntimeError):
    """The bounded wait ran out while a LIVE owner held the lock."""


@dataclass(frozen=True)
class Holder:
    pid: int
    worker: str | None
    tree: str
    since: str  # ISO 8601, UTC

    def describe(self) -> str:
        who = f"pid {self.pid}"
        if self.worker:
            who += f" ({self.worker})"
        return f"{who} in {self.tree} since {self.since}"


def lock_dir() -> Path:
    env = os.environ.get(LOCK_DIR_ENV)
    if env:
        return Path(env)
    harness = _REPO_ROOT / "_harness"
    if harness.is_dir():
        return harness
    return Path(tempfile.gettempdir())


def lock_path(name: str) -> Path:
    return lock_dir() / f".{name}.lock"


def read_holder(path: Path) -> Holder | None:
    """The owner recorded in *path*, or ``None`` when absent or not yet written."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Holder(pid=int(data["pid"]), worker=data.get("worker"),
                      tree=str(data["tree"]), since=str(data["since"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _pid_alive(pid: int) -> bool:
    import psutil  # class A (requirements.txt); pid_exists is safe on Windows

    return psutil.pid_exists(pid)


def _is_stale(path: Path) -> bool:
    holder = read_holder(path)
    if holder is not None:
        return not _pid_alive(holder.pid)
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False  # vanished under us — the retry will find out
    return age > UNREADABLE_GRACE_S


def _write_owner(fd: int) -> None:
    holder = Holder(pid=os.getpid(), worker=current_worker(), tree=str(_REPO_ROOT),
                    since=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(asdict(holder), f)


def acquire(path: Path, timeout_s: float, poll_s: float = POLL_S) -> None:
    """Take *path*; raise ``MachineLockTimeout`` naming the holder after *timeout_s*."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _is_stale(path):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass  # another waiter recovered it first; race for it below
                continue
            if time.monotonic() >= deadline:
                holder = read_holder(path)
                who = holder.describe() if holder else f"an unreadable owner ({path})"
                raise MachineLockTimeout(
                    f"{path.name} held by {who}; gave up after {timeout_s:.1f}s")
            time.sleep(poll_s)
            continue
        _write_owner(fd)
        return


def release(path: Path) -> None:
    """Remove *path* if it is ours. Never removes another process's lock."""
    holder = read_holder(path)
    if holder is not None and holder.pid != os.getpid():
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def resolve_timeout(timeout_s: float | None) -> float:
    if timeout_s is not None:
        return timeout_s
    return float(os.environ.get(LOCK_TIMEOUT_ENV, DEFAULT_TIMEOUT_S))


@contextmanager
def machine_lock(name: str, *, timeout_s: float | None = None,
                 path: Path | None = None) -> Iterator[Path]:
    """Hold ``_harness/.<name>.lock`` for the block; released in ``finally``."""
    path = path or lock_path(name)
    acquire(path, resolve_timeout(timeout_s))
    try:
        yield path
    finally:
        release(path)
