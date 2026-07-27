"""LogTailer — file-based log reader for backend ← trainer communication.

Tails a ``job_log.jsonl`` file written by :class:`JobLogWriter` in the
trainer subprocess and dispatches parsed entries to a callback.

Features:
    - Persistent offset tracking (survives backend restarts)
    - File rotation / truncation detection
    - Configurable poll interval (default 150 ms — keeps live sampling /
      training progress streaming near real-time instead of in 2 Hz bursts)
    - Graceful shutdown via stop event
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable

import structlog

logger = structlog.get_logger(__name__)

LOG_FILENAME = "job_log.jsonl"
OFFSET_SUFFIX = ".offset"

# Batching thresholds for offset persistence: save every N dispatched lines
# OR every S seconds, whichever comes first. Per-step trainer logging at high
# rates made a full open/write/close per dispatched line constant WDDM-disk
# churn alongside training I/O; this bounds it without meaningfully widening
# the re-dispatch window on an unclean shutdown. stop() always force-saves
# regardless of these thresholds.
_SAVE_EVERY_N_LINES = 20
_SAVE_INTERVAL_S = 1.0


class LogTailer:
    """Polls a JSON-Lines log file and dispatches entries to the backend.

    The tailer persists its byte-offset in a sibling ``.offset`` file so
    that on backend restart it can resume from exactly where it left off,
    avoiding duplicate or missed messages.

    Args:
        job_id: Owning job identifier (for dispatcher context).
        log_path: Absolute path to the ``job_log.jsonl`` file.
        dispatcher: Callback ``(job_id, entry_dict) → None``.
        poll_interval: Seconds between file-size checks.
    """

    def __init__(
        self,
        job_id: str,
        log_path: str,
        dispatcher: Callable[[str, dict[str, Any]], None],
        poll_interval: float = 0.15,
    ) -> None:
        self.job_id = job_id
        self.log_path = log_path
        self.offset_path = log_path + OFFSET_SUFFIX
        self.dispatcher = dispatcher
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._offset: int = self._load_offset()
        self._thread: threading.Thread | None = None
        self._lines_since_save = 0
        self._last_save_monotonic = time.monotonic()

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> threading.Thread:
        """Begin tailing in a daemon thread. Returns the thread handle."""
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"log_tailer_{self.job_id[:8]}",
        )
        self._thread.start()
        logger.debug(
            "log_tailer_started",
            job_id=self.job_id,
            log_path=self.log_path,
            resume_offset=self._offset,
        )
        return self._thread

    def stop(self) -> None:
        """Signal the tailer to stop and persist the final offset.

        Safe to call from the tailer's own thread — when a dispatched
        handler (e.g. ``exit``) triggers a stop, joining self would raise
        and abort the polling loop before its offset write, leaving the
        last-processed line eligible for re-dispatch on the next start.
        """
        self._stop_event.set()
        if (
            self._thread
            and self._thread.is_alive()
            and self._thread is not threading.current_thread()
        ):
            self._thread.join(timeout=2.0)
        try:
            self._save_offset(force=True)
        except Exception:
            pass

    # ── Main loop ─────────────────────────────────────────────────────

    def _run(self) -> None:
        """Poll the log file and dispatch new entries."""
        # Wait for file to appear
        while not self._stop_event.is_set() and not os.path.exists(self.log_path):
            self._stop_event.wait(self.poll_interval)

        if self._stop_event.is_set():
            return

        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                f.seek(0, os.SEEK_END)
                file_len = f.tell()
                
                # Truncation / Rotation Check
                if file_len < self._offset:
                    logger.info(
                        "log_tailer_file_truncated",
                        job_id=self.job_id,
                        old_offset=self._offset,
                        new_size=file_len,
                    )
                    self._offset = 0

                f.seek(self._offset)

                while not self._stop_event.is_set():
                    current_pos = f.tell()
                    line = f.readline()

                    if not line:
                        # Clear EOF flag for next read
                        f.seek(current_pos)
                        
                        # Occasionally check for file rotation if stuck at EOF
                        try:
                            if os.path.exists(self.log_path) and os.path.getsize(self.log_path) < current_pos:
                                f.seek(0)
                        except OSError:
                            pass
                            
                        self._stop_event.wait(self.poll_interval)
                        continue

                    line = line.strip()
                    if not line:
                        continue

                    # Advance the tracked offset BEFORE dispatching: a
                    # handler that calls stop() synchronously from inside
                    # the callback (e.g. JobManager's exit-message path)
                    # must see the post-line offset when stop()'s forced
                    # save runs, not the pre-line one.
                    self._offset = f.tell()
                    try:
                        entry = json.loads(line)
                        self.dispatcher(self.job_id, entry)
                    except json.JSONDecodeError:
                        logger.debug("log_tailer_bad_json", job_id=self.job_id)

                    self._lines_since_save += 1
                    self._save_offset()

        except Exception as exc:
            logger.warning("log_tailer_loop_error", job_id=self.job_id, error=str(exc))

    # ── Offset persistence ────────────────────────────────────────────

    def _load_offset(self) -> int:
        """Load the last-read byte offset from disk."""
        if os.path.exists(self.offset_path):
            try:
                with open(self.offset_path, "r", encoding="utf-8") as f:
                    return int(f.read().strip())
            except (ValueError, OSError):
                pass
        return 0

    def _save_offset(self, *, force: bool = False) -> None:
        """Persist the current byte offset to disk, batched.

        A full open/write/close per dispatched line is constant disk churn
        at high-rate per-step trainer logging. Unless *force* (used by
        stop() so a shutdown never loses more than the current in-flight
        batch), this only actually writes every _SAVE_EVERY_N_LINES lines
        or _SAVE_INTERVAL_S seconds, whichever comes first.

        The write itself goes through a sibling ``.tmp`` file + os.replace
        so a crash mid-write can never leave a partial int on disk — a
        truncated value would fail ``int()`` in _load_offset and reset to
        0, re-dispatching the entire log on reattach.
        """
        if not force:
            elapsed = time.monotonic() - self._last_save_monotonic
            if (
                self._lines_since_save < _SAVE_EVERY_N_LINES
                and elapsed < _SAVE_INTERVAL_S
            ):
                return

        self._lines_since_save = 0
        self._last_save_monotonic = time.monotonic()
        tmp_path = self.offset_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(str(self._offset))
            os.replace(tmp_path, self.offset_path)
        except OSError:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
