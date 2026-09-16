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
* **acquisition** — the owner metadata (pid, xdist worker id, worktree, ISO
  timestamp) is written into a temp file beside the lock and then ``os.link``ed
  onto the lock's name: atomic on NTFS and POSIX, and the name never exists
  without its content. ``O_CREAT | O_EXCL`` then a write was equally atomic
  about the NAME but left the file empty for an instant, and an empty lock is
  indistinguishable from a crashed writer — so a merely descheduled live writer
  was recovered under it and two holders resulted (VERIFY 3.01, ``_publish_owner``).
* **stale recovery** — on ``FileExistsError`` the metadata is read; a dead
  owner pid (``psutil.pid_exists``, never ``os.kill(pid, 0)``, which on Windows
  TERMINATES the process) means the file is removed and the open retried. An
  unreadable file (the winner between ``open`` and ``write``) counts as live
  for ``UNREADABLE_GRACE_S`` and stale after. A live owner is never broken.
  The removal is **bound to the identity that was judged**: a naked
  judge-then-unlink has a window in which a second waiter recovers the same
  stale file and takes the lock, and unlinking THEN deletes a live owner's
  lock and admits two holders at once (VERIFY 1.01). So recovery takes a
  turn — the one place an unlink of a stale lock may happen — and inside that
  turn re-reads the file and unlinks only if the bytes are still the ones it
  judged stale. A stale owner is dead, so it cannot release its own file; with
  every recovery serialised, identical bytes inside the turn mean the same file.
  The turn itself is the OS's exclusive lock on ``.<name>.lock.recover``, a
  file created once and NEVER removed. An earlier version took the turn with a
  second ``O_EXCL`` file and broke an orphaned one by age — but "old enough"
  judges the NAME, and between the ``stat`` and the ``unlink`` that name can
  already belong to a fresh, live turn: a merely slow recoverer was robbed and
  two recoverers ran at once, which is precisely what makes the identity check
  above conclusive (VERIFY 2.02). A kernel lock has no such window: it is bound
  to the open handle, so it cannot be stolen, and the kernel drops it when the
  holder dies, so a crashed recoverer cannot wedge the lock either.
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
# The recovery turn's file, beside the lock it recovers. Created once and never
# deleted: the turn is the OS lock on it, not its existence (see _recovery_turn).
RECOVER_SUFFIX = ".recover"
# The byte of that file the OS lock is taken on — past the pid written at 0,
# because a Windows lock makes its range unreadable to other processes.
TURN_LOCK_OFFSET = 1024
# A test-only barrier at the ONE moment an acquirer can be caught mid-flight:
# its metadata is built but not yet published under the lock's name. Set the
# env var to a path and the acquirer creates that file and waits (bounded)
# until the file is removed, so a test can park a REAL second process exactly
# there instead of hand-writing a file that merely looks like one
# (test_machine_lock.py::test_a_paused_live_writer_is_never_robbed_of_its_lock).
# Unset in every real run: no production path reads it.
PUBLISH_BARRIER_ENV = "MRLN_MACHINE_LOCK_PUBLISH_BARRIER"
PUBLISH_BARRIER_BOUND_S = 120.0

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


def _parse_holder(raw: str) -> Holder | None:
    try:
        data = json.loads(raw)
        return Holder(pid=int(data["pid"]), worker=data.get("worker"),
                      tree=str(data["tree"]), since=str(data["since"]))
    except (ValueError, KeyError, TypeError):
        return None


def read_holder(path: Path) -> Holder | None:
    """The owner recorded in *path*, or ``None`` when absent or not yet written."""
    try:
        return _parse_holder(path.read_text(encoding="utf-8"))
    except OSError:
        return None


def _pid_alive(pid: int) -> bool:
    import psutil  # class A (requirements.txt); pid_exists is safe on Windows

    return psutil.pid_exists(pid)


def _inspect(path: Path) -> tuple[bool, str | None]:
    """``(is_stale, the exact bytes judged)``; ``None`` when unreadable/absent.

    The bytes come back with the verdict so the caller can bind an unlink to
    the identity it judged, instead of to the path (VERIFY 1.01).
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return False, None  # vanished under us — the retry will find out
    holder = _parse_holder(raw)
    if holder is not None:
        return (not _pid_alive(holder.pid)), raw
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False, None
    return age > UNREADABLE_GRACE_S, raw


def recovery_sentinel(path: Path) -> Path:
    """The turn-taking file for recovering *path*; public for its own test."""
    return path.with_name(path.name + RECOVER_SUFFIX)


def _try_lock(fd: int) -> bool:
    """Take the OS's exclusive lock on *fd*'s turn byte, or report failure.

    The KERNEL owns this lock: it goes away when the handle closes and when the
    process dies, crash included. That is why the turn needs no age heuristic
    and no unlink — and an unlink is exactly what could never be bound to the
    file it judged (VERIFY 2.02): ``stat`` the name, ``unlink`` the name, and
    in between the name may belong to a fresh, live turn.

    The byte locked is ``TURN_LOCK_OFFSET``, past the pid the file carries for
    a human reading it: on Windows a locked range is MANDATORY, so locking
    byte 0 would make the file unreadable to everyone else, including this
    module's own test.
    """
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, TURN_LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False  # another process holds the turn
        return True
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, TURN_LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def _recovery_turn(path: Path) -> Iterator[bool]:
    """Yield ``True`` while THIS process is the only one recovering *path*.

    The turn file is created once and never removed — not on release, not by
    age. Its EXISTENCE grants nothing; the OS lock on it does, and that is
    bound to the open handle, so the turn cannot be stolen from a slow holder
    nor leaked by a crashed one.
    """
    sentinel = recovery_sentinel(path)
    try:
        fd = os.open(sentinel, os.O_CREAT | os.O_RDWR)
    except OSError:
        yield False  # e.g. the directory vanished — the caller re-polls
        return
    try:
        if not _try_lock(fd):
            yield False
            return
        try:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, str(os.getpid()).encode("ascii"))  # who holds it, for a human
            yield True
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


def _recover(path: Path, judged: str | None) -> None:
    """Remove *path* only if it is still byte-for-byte the lock we judged stale."""
    with _recovery_turn(path) as mine:
        if not mine:
            return
        stale, raw = _inspect(path)
        if not stale or raw != judged:
            return  # a different — possibly LIVE — lock lives here now
        try:
            path.unlink()
        except FileNotFoundError:
            pass


_barrier_passed = False


def _publish_barrier() -> None:
    """Park here while a test holds us, or return at once (the normal case).

    ONE-SHOT: acquisition retries, and a barrier that re-armed on every retry
    would park the writer again the instant the test released it.
    """
    global _barrier_passed

    marker = os.environ.get(PUBLISH_BARRIER_ENV)
    if not marker or _barrier_passed:
        return
    _barrier_passed = True
    flag = Path(marker)
    flag.write_text(str(os.getpid()), encoding="utf-8")
    deadline = time.monotonic() + PUBLISH_BARRIER_BOUND_S  # bounded: a dead
    while flag.exists() and time.monotonic() < deadline:   # parent cannot wedge us
        time.sleep(0.02)


def _publish_owner(path: Path) -> None:
    """Create *path* ALREADY carrying our metadata, or raise ``FileExistsError``.

    The lock must never exist without its content. ``O_CREAT | O_EXCL`` followed
    by a write leaves a window in which the file exists and parses as no holder,
    so ``_inspect``'s unreadable branch judges a LIVE writer by mtime and, past
    the grace, calls it stale; the identity re-check inside the recovery turn
    cannot save it, because the bytes really are unchanged. The recoverer then
    unlinks a live writer's file and a second process takes the replacement
    (VERIFY 3.01). A LARGER grace only makes the window rarer, never absent —
    the writer is descheduled for as long as the box decides.

    So the metadata is written into a temp file beside the lock and hard-linked
    into place: ``os.link`` fails when the name exists (NTFS and POSIX alike),
    which is the same all-or-nothing create ``O_EXCL`` gave, with the bytes
    already inside. It needs a filesystem with hard links; if the lock dir has
    none the error surfaces here rather than degrading into the window above.
    """
    holder = Holder(pid=os.getpid(), worker=current_worker(), tree=str(_REPO_ROOT),
                    since=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".new")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(asdict(holder), f)
        _publish_barrier()
        os.link(tmp, path)  # FileExistsError: someone else owns the name
    finally:
        os.unlink(tmp)  # the lock keeps the content under its own name


def acquire(path: Path, timeout_s: float, poll_s: float = POLL_S) -> None:
    """Take *path*; raise ``MachineLockTimeout`` naming the holder after *timeout_s*."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            _publish_owner(path)
        except FileExistsError:
            stale, judged = _inspect(path)
            if stale:
                _recover(path, judged)
                if not path.exists():
                    continue  # recovered: race for the free path at once
                # still there: another waiter's turn, or a lock that is no
                # longer the one we judged. Wait it out like any live owner —
                # recovery never spins unbounded.
            if time.monotonic() >= deadline:
                holder = read_holder(path)
                who = holder.describe() if holder else f"an unreadable owner ({path})"
                raise MachineLockTimeout(
                    f"{path.name} held by {who}; gave up after {timeout_s:.1f}s")
            time.sleep(poll_s)
            continue
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
