"""LogTailer — file-based log reader for backend ← trainer communication.

Tails a ``job_log.jsonl`` file written by :class:`JobLogWriter` in the
trainer subprocess and dispatches parsed entries to a callback.

Features:
    - Persistent offset tracking (survives backend restarts)
    - File rotation / truncation detection
    - Configurable poll interval (default 500 ms)
    - Graceful shutdown via stop event
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Callable

import structlog

logger = structlog.get_logger(__name__)

LOG_FILENAME = "job_log.jsonl"
OFFSET_SUFFIX = ".offset"


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
        poll_interval: float = 0.5,
    ) -> None:
        self.job_id = job_id
        self.log_path = log_path
        self.offset_path = log_path + OFFSET_SUFFIX
        self.dispatcher = dispatcher
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._offset: int = self._load_offset()
        self._thread: threading.Thread | None = None

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
        """Signal the tailer to stop and do a final drain."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        # Final drain: read anything remaining
        try:
            self._read_new_lines()
            self._save_offset()
        except Exception:
            pass

    # ── Main loop ─────────────────────────────────────────────────────

    def _run(self) -> None:
        """Poll the log file and dispatch new entries."""
        while not self._stop_event.is_set():
            try:
                if not os.path.exists(self.log_path):
                    # File doesn't exist yet — trainer may still be
                    # initialising the output directory.
                    self._stop_event.wait(self.poll_interval)
                    continue

                file_size = os.path.getsize(self.log_path)

                # Rotation / truncation detection
                if file_size < self._offset:
                    logger.info(
                        "log_tailer_file_truncated",
                        job_id=self.job_id,
                        old_offset=self._offset,
                        new_size=file_size,
                    )
                    self._offset = 0

                if file_size > self._offset:
                    self._read_new_lines()
                    self._save_offset()

            except Exception as exc:
                logger.warning(
                    "log_tailer_poll_error",
                    job_id=self.job_id,
                    error=str(exc),
                )

            self._stop_event.wait(self.poll_interval)

    # ── File I/O ──────────────────────────────────────────────────────

    def _read_new_lines(self) -> None:
        """Read from the current offset to EOF, dispatching each line."""
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                f.seek(self._offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        self.dispatcher(self.job_id, entry)
                    except json.JSONDecodeError:
                        # Malformed line (partial write?) — skip but keep going
                        logger.debug(
                            "log_tailer_bad_json",
                            job_id=self.job_id,
                            line=line[:120],
                        )
                self._offset = f.tell()
        except OSError as exc:
            logger.warning(
                "log_tailer_read_error",
                job_id=self.job_id,
                error=str(exc),
            )

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

    def _save_offset(self) -> None:
        """Persist the current byte offset to disk."""
        try:
            with open(self.offset_path, "w", encoding="utf-8") as f:
                f.write(str(self._offset))
        except OSError:
            pass
