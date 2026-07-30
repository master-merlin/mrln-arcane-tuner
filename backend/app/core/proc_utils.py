"""Subprocess plumbing shared by the guarded-download and ffmpeg runners.

``PipeTail`` exists because a child process whose stdout/stderr is a pipe will
BLOCK on write once the OS pipe buffer (~4-64 KB) fills, and a parent that only
reads those pipes after the child exits deadlocks: the child is waiting for the
parent to read, the parent is waiting for the child to finish.

That is not hypothetical for either caller. ffmpeg emits stream dumps and
per-packet warnings (non-monotonic DTS on stream-copy splits is routine) even
with ``-nostats``; huggingface_hub's downloader emits tqdm bars and urllib3
retry warnings on exactly the flaky networks its stall guard targets. Draining
both pipes on daemon threads decouples the child's verbosity from the parent's
liveness, and the bounded tail keeps memory flat while preserving the lines
that actually matter (the newest stderr carries the real error).
"""

from __future__ import annotations

import threading
from collections import deque


class PipeTail:
    """Continuously drain one child pipe on a daemon thread, keeping only the
    last ``max_chars`` of output."""

    def __init__(self, stream, max_chars: int = 65536):
        self._chunks: deque[str] = deque()
        self._size = 0
        self._max = max_chars
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._drain, args=(stream,), daemon=True, name="pipe_tail",
        )
        self._thread.start()

    def _drain(self, stream) -> None:
        try:
            for line in stream:
                with self._lock:
                    self._chunks.append(line)
                    self._size += len(line)
                    while self._size > self._max and len(self._chunks) > 1:
                        self._size -= len(self._chunks.popleft())
        except Exception:
            # Pipe closed mid-read (child killed) — the tail so far stands.
            pass

    def text(self, join_timeout_s: float = 5.0) -> str:
        """Join the drain thread (EOF arrives when the child exits or is
        killed) and return the retained tail."""
        self._thread.join(timeout=join_timeout_s)
        with self._lock:
            return "".join(self._chunks)


def drain_pipes(proc) -> tuple[PipeTail | None, PipeTail | None]:
    """Attach tail-drainers to whichever of stdout/stderr are pipes."""
    out = PipeTail(proc.stdout) if proc.stdout is not None else None
    err = PipeTail(proc.stderr) if proc.stderr is not None else None
    return out, err


def kill_process(proc) -> None:
    """Terminate then kill. On Windows both calls invoke ``TerminateProcess``
    (releasing file locks immediately); on POSIX ``terminate`` (SIGTERM) is
    tried first and ``kill`` (SIGKILL) follows only if the process is still
    alive after a short grace wait."""
    for action in (proc.terminate, proc.kill):
        try:
            action()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
            return
        except Exception:
            continue
