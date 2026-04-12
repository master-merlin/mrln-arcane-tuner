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
    exit         – Trainer process exit (code + optional error)
"""

from __future__ import annotations

import atexit
import json
import os
import time
from typing import Any

LOG_FILENAME = "job_log.jsonl"


class JobLogWriter:
    """Append-only JSON-Lines writer for the trainer subprocess.

    Each call to :meth:`emit` writes a single JSON object on its own
    line, terminated by ``\\n``.  The file is opened in append mode with
    line buffering so every ``emit`` is immediately flushed to disk.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.log_path = os.path.join(output_dir, LOG_FILENAME)
        os.makedirs(output_dir, exist_ok=True)
        # Append mode + line-buffered (buffering=1 in text mode)
        self._file = open(  # noqa: SIM115
            self.log_path, "a", encoding="utf-8", buffering=1,
        )
        # Ensure we close cleanly on process exit
        atexit.register(self.close)

    # ── Public API ────────────────────────────────────────────────────

    def emit(self, msg_type: str, data: Any) -> None:
        """Write a single structured JSON line.

        Args:
            msg_type: One of ``log``, ``status``, ``cache_ready``,
                      ``warning``, ``step``, ``exit``.
            data: Arbitrary JSON-serialisable payload.
        """
        if self._file.closed:
            return
        entry = {
            "t": time.time(),
            "type": msg_type,
            "data": data,
        }
        try:
            self._file.write(
                json.dumps(entry, separators=(",", ":"), default=str) + "\n",
            )
        except (OSError, ValueError):
            pass  # Non-fatal — never crash the trainer for a log write

    def log(self, message: str) -> None:
        """Shorthand for ``emit("log", message)``."""
        self.emit("log", message)

    def status(self, label: str) -> None:
        """Shorthand for ``emit("status", label)``."""
        self.emit("status", label)

    def warning(self, message: str) -> None:
        """Shorthand for ``emit("warning", message)``."""
        self.emit("warning", message)

    def step(self, metrics: dict[str, Any]) -> None:
        """Shorthand for ``emit("step", metrics)``."""
        self.emit("step", metrics)

    def exit(self, code: int, error: str | None = None) -> None:
        """Write a terminal exit message, then close the file."""
        payload: dict[str, Any] = {"code": code}
        if error:
            payload["error"] = error
        self.emit("exit", payload)
        self.close()

    def close(self) -> None:
        """Flush and close the backing file (idempotent)."""
        if not self._file.closed:
            try:
                self._file.flush()
                self._file.close()
            except (OSError, ValueError):
                pass
