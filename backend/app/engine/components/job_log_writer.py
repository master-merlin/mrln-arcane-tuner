"""JobLogWriter — file-based IPC channel for trainer → backend communication.

Writes structured JSON-Lines to ``{output_dir}/job_log.jsonl`` so the
backend's ``LogTailer`` can read messages regardless of process lifecycle.

This replaces the fragile ``stdout`` pipe that was coupled to the parent
process and would crash the trainer on backend restart.

Message types:
    log          – Free-form log string
    status       – UI status label (e.g. "Loading Model", "Training")
    cache_ready  – List of dataset names whose caches are ready
    warning      – Warning message for the UI
    step         – Per-step training metrics dict
    gpu_health   – One reading of the card, from the recorder's thread
    exit         – Trainer process exit (code + optional error)
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import threading
import time
import weakref
from collections.abc import Callable
from typing import Any

LOG_FILENAME = "job_log.jsonl"
# How long a line waits for another thread's write, and on the way out for its own. One
# that takes longer is stuck on the disk: the line is given up, and the log is stuck until
# a write gets through again.
LOCK_WAIT_S = 5.0
# How long a diagnostic line waits on stderr or stdout, which a job sends to
# trainer_stdout.log in its output folder: the same disk as its job log.
NOTE_WAIT_S = 1.0


class BestEffortStream:
    """A stream whose writes hold the caller for ``NOTE_WAIT_S`` at most.

    Each write runs on a side thread. One that doesn't return in time is left behind,
    and the lines after it are dropped at once while it hangs; once it has landed, lines
    get through again. ``flush`` does nothing: each write flushes on its own thread.
    """

    _of: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._stuck_on: threading.Thread | None = None

    @classmethod
    def of(cls, stream: Any) -> BestEffortStream:
        """The one best-effort stream over ``stream``, so that all its callers share
        what they know of a stall."""
        if isinstance(stream, cls):
            return stream
        try:
            found = cls._of.get(stream)
            if found is None:
                found = cls._of[stream] = cls(stream)
            return found
        except TypeError:  # a stream that takes no weak reference
            return cls(stream)

    def write(self, text: str) -> int:
        """Write ``text``, and give the number of characters written: 0 if it was given up."""
        stuck = self._stuck_on
        if stuck is not None:
            if stuck.is_alive():
                return 0
            self._stuck_on = None
        thread = threading.Thread(
            target=self._write, args=(text,), name="best-effort-write", daemon=True,
        )
        try:
            thread.start()
        except RuntimeError:
            return 0
        thread.join(NOTE_WAIT_S)
        if thread.is_alive():
            self._stuck_on = thread
            return 0
        return len(text)

    def flush(self) -> None:
        pass

    def _write(self, text: str) -> None:
        try:
            self._stream.write(text)
            self._stream.flush()
        except Exception:
            pass  # a diagnostic that can't be written is lost, and says nothing more


def note(text: str) -> bool:
    """Write one diagnostic line to stderr, best-effort: it never holds the caller for
    longer than ``NOTE_WAIT_S``. Returns whether it was written.

    For the lines a person reads in trainer_stdout.log when the job log can't take them,
    and for everything the job's way out says there.
    """
    line = f"{text}\n"
    return BestEffortStream.of(sys.stderr).write(line) == len(line)


class JobLogWriter:
    """Append-only JSON-Lines writer for the trainer subprocess.

    Each call to :meth:`emit` writes a single JSON object on its own
    line, terminated by ``\\n``.  The file is opened in append mode with
    line buffering so every ``emit`` is immediately flushed to disk.

    Two threads write: the training loop and the GPU health recorder. The lock
    covers the write and nothing else. It is reentrant because logging is bridged
    to this writer, so a finaliser or a warning that logs during a write comes back
    here on the same thread, where a plain lock would hang the job.

    No line waits for another thread's write without a bound: a write stuck on the disk
    must not stop the training, nor the job's way out. A line that has waited
    ``LOCK_WAIT_S`` is given up, and while the log is stuck the next ones give up at
    once, until a write gets through. Only the exit line waits its full bound each time.

    On the way out (``bound_every_write``) no line waits on the disk itself without a
    bound either: each write runs on a side thread, waited on for ``LOCK_WAIT_S``, so a
    stalled disk can't keep the job from ending. The training's own lines don't pay for
    a thread each: before the way out, a write stalled on the disk with the lock free
    holds the thread that writes it, as it always has, and as that disk would hold the
    training's own saves.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.log_path = os.path.join(output_dir, LOG_FILENAME)
        os.makedirs(output_dir, exist_ok=True)
        # Append mode + line-buffered (buffering=1 in text mode)
        self._file = open(  # noqa: SIM115
            self.log_path, "a", encoding="utf-8", buffering=1,
        )
        self._exit_emitted = False
        self._lock = threading.RLock()
        self._stuck = False
        self._bounded = False  # on the way out: every write on a side thread, within a bound
        self._local = threading.local()  # how deep this thread is inside a write
        self.last_error: str | None = None  # what cost the last line lost, for a warning
        # Ensure we close cleanly on process exit
        atexit.register(self.close)

    # ── Public API ────────────────────────────────────────────────────

    @property
    def stuck(self) -> bool:
        """Whether the last line was given up, waiting on a write stuck on the disk."""
        return self._stuck

    def bound_every_write(self) -> None:
        """From now on no write waits on the disk without a bound, on any thread.

        For the job's way out, where a line costs a thread: the summary, a traceback, the
        finaliser's warnings, the exit line and the close.
        """
        self._bounded = True

    def emit(self, msg_type: str, data: Any) -> bool:
        """Write a single structured JSON line, and return whether nothing failed.

        Never raises: a log write must not crash the trainer. False means the line is
        lost, to a disk error or to a write stuck past the bound. A closed writer takes
        no more lines, as it should, which is no failure.

        Args:
            msg_type: One of ``log``, ``status``, ``cache_ready``,
                      ``warning``, ``step``, ``gpu_health``, ``exit``.
            data: Arbitrary JSON-serialisable payload.
        """
        if self._file.closed:
            return True
        entry = {
            "t": time.time(),
            "type": msg_type,
            "data": data,
        }
        try:
            line = json.dumps(entry, separators=(",", ":"), default=str) + "\n"
        except (TypeError, ValueError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False
        return bool(self._within_bound(lambda: self._write_line(line)))

    def log(self, message: str) -> bool:
        """Shorthand for ``emit("log", message)``."""
        return self.emit("log", message)

    def status(self, label: str) -> bool:
        """Shorthand for ``emit("status", label)``."""
        return self.emit("status", label)

    def warning(self, message: str) -> bool:
        """Shorthand for ``emit("warning", message)``."""
        return self.emit("warning", message)

    def step(self, metrics: dict[str, Any]) -> bool:
        """Shorthand for ``emit("step", metrics)``."""
        return self.emit("step", metrics)

    def log_exception(
        self,
        exc: BaseException,
        *,
        header: str = "Trainer crashed — traceback follows:",
    ) -> None:
        """Forward a full exception traceback as individual ``log`` lines.

        The trainer runs as a detached subprocess whose stderr only reaches
        ``trainer_stdout.log`` on disk; the UI reads this job log. Sending the
        full traceback here (not just ``str(exc)`` in the exit message) makes
        the entire stack visible in the UI log viewer. Each physical line is a
        separate entry so the viewer renders normal rows.
        """
        import traceback

        self.log(header)
        formatted = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__),
        ).rstrip()
        for line in formatted.splitlines():
            self.log(line)

    def exit(self, code: int, error: str | None = None) -> None:
        """Write a terminal exit message exactly once, then close the file.

        Idempotent: a second call is a no-op (with a stderr warning so the
        offending site shows up in trainer_stdout.log).  The exit message
        is what tells the backend JobManager to transition the job to
        ``COMPLETED`` / ``FAILED`` and remove it from the active queue —
        writing it more than once, or writing it while real work is still
        in flight, makes the UI lose track of a live job.
        """
        if self._exit_emitted:
            # The writer can't be trusted (double-exit corner); the log channel is
            # already closed, so stderr is the only way to surface the bug.
            note(
                f"WARNING: JobLogWriter.exit() called twice (code={code}); "
                "ignoring second call to avoid spurious terminal event.",
            )
            return
        self._exit_emitted = True
        payload: dict[str, Any] = {"code": code}
        if error:
            payload["error"] = error
        def exit_and_close() -> None:
            # One step: the exit line is the last in the file, whichever thread writes
            # next. It waits its full bound even on a stuck log: it is what ends the job
            # for the backend.
            if not self._enter(LOCK_WAIT_S):
                return
            try:
                self.emit("exit", payload)
                self.close()
            finally:
                self._leave()

        self._within_bound(exit_and_close)

    def close(self) -> None:
        """Flush and close the backing file (idempotent). Left open on a stuck log."""
        if not self._file.closed:
            self._within_bound(self._close_file)

    # ── Internals ─────────────────────────────────────────────────────

    def _write_line(self, line: str) -> bool:
        if not self._enter():
            self.last_error = "the job log is stuck on another write"
            return False
        try:
            if self._file.closed:
                return True
            self._file.write(line)
            self._file.flush()
        except (OSError, ValueError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False
        finally:
            self._leave()
        self._stuck = False
        return True

    def _close_file(self) -> None:
        if not self._enter():
            return
        try:
            if not self._file.closed:
                self._file.flush()
                self._file.close()
        except (OSError, ValueError):
            pass
        finally:
            self._leave()

    def _within_bound(self, work: Callable[[], Any]) -> Any:
        """``work()``. On the way out it runs on a side thread, waited on for a bound.

        A call from inside a write (logging from a finaliser, or the exit line's own
        write and close) runs in place: it holds the lock, which a side thread would wait on.
        """
        if not self._bounded or getattr(self._local, "depth", 0) > 0:
            return work()
        result: list[Any] = []
        thread = threading.Thread(
            target=lambda: result.append(work()), name="job-log-write", daemon=True,
        )
        try:
            thread.start()
        except RuntimeError:
            self.last_error = "no thread for a write to the job log"
            return False
        thread.join(LOCK_WAIT_S)
        if thread.is_alive():
            self.last_error = "a write to the job log did not return"
            self._mark_stuck()
            return False
        return result[0] if result else False

    def _enter(self, wait_s: float | None = None) -> bool:
        """The lock, within a bound, counted for this thread. A stuck log is waited on no
        more, unless told so."""
        if wait_s is None:
            wait_s = 0.0 if self._stuck else LOCK_WAIT_S
        if not self._lock.acquire(timeout=wait_s):
            self._mark_stuck()
            return False
        self._local.depth = getattr(self._local, "depth", 0) + 1
        return True

    def _leave(self) -> None:
        self._local.depth -= 1
        self._lock.release()

    def _mark_stuck(self) -> None:
        if not self._stuck:
            self._stuck = True
            # The job log is what is stuck; stderr reaches trainer_stdout.log, on the same
            # disk, so this too has a bound.
            note("WARNING: job log is stuck, going on without it")
