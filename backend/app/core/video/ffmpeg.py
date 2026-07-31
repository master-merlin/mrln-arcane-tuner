"""ffmpeg binary resolution + invocation helpers for the video curation layer.

The split/scene-detect workers shell out to ffmpeg for lossless stream-copy
and accurate-seek re-encode. We never run ffmpeg through a shell (``shell=True``
is a Windows quoting/injection footgun); every call passes an explicit argv
list.

Binary resolution order (first hit wins):
  1. ``FFMPEG_PATH`` env var (operator override / container pin)
  2. ``shutil.which("ffmpeg")`` (system install — e.g. ``C:\\Program Files\\ffmpeg``)
  3. ``imageio_ffmpeg.get_ffmpeg_exe()`` (bundled static binary; the Docker /
     no-system-ffmpeg fallback)

``nearest_keyframe_before(path, t)`` uses PyAV (no decode) to find the keyframe
PTS at-or-before ``t`` — the split worker uses it to decide stream-copy vs
re-encode per segment.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from functools import lru_cache
from pathlib import Path

import structlog

from app.core.proc_utils import PipeTail, kill_process

logger = structlog.get_logger(__name__)


class FFmpegError(RuntimeError):
    """Raised when ffmpeg cannot be resolved or a run exits non-zero."""


@lru_cache(maxsize=1)
def resolve_ffmpeg() -> str:
    """Return the path to an ffmpeg executable.

    Order: ``FFMPEG_PATH`` env → ``shutil.which`` → bundled imageio-ffmpeg.
    Raises :class:`FFmpegError` if none resolve.
    """
    env = os.environ.get("FFMPEG_PATH")
    if env and Path(env).exists():
        return env

    which = shutil.which("ffmpeg")
    if which:
        return which

    try:  # pragma: no cover - import guard for the bundled fallback
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception as exc:  # noqa: BLE001
        logger.warning("imageio_ffmpeg_unavailable", error=str(exc))

    raise FFmpegError(
        "ffmpeg not found: set FFMPEG_PATH, install ffmpeg on PATH, "
        "or install imageio-ffmpeg."
    )


def _parse_progress_line(line: str) -> tuple[str, str] | None:
    """Parse one ``key=value`` line emitted by ``-progress pipe:1``."""
    line = line.strip()
    if "=" not in line:
        return None
    key, _, value = line.partition("=")
    return key.strip(), value.strip()


#: Default wall-clock ceiling for one ffmpeg invocation. Generous — a long
#: re-encode is legitimately slow — but bounded, because an unbounded
#: subprocess.run() on a wedged ffmpeg parks the cpu-lane worker thread
#: permanently and every later split / scene-detect task queues behind it.
DEFAULT_TIMEOUT_S = 1800.0

#: How often the run loop checks liveness, the timeout, and the abort flag.
_POLL_INTERVAL_S = 0.25


class FFmpegAborted(FFmpegError):
    """Raised when ``should_abort`` asked the run to stop (task cancelled)."""


def run_ffmpeg(
    args: list[str],
    progress_cb=None,
    *,
    timeout: float | None = DEFAULT_TIMEOUT_S,
    should_abort=None,
) -> int:
    """Run ffmpeg with an explicit argv list (never via a shell).

    The resolved ffmpeg binary is prepended automatically — ``args`` should be
    everything *after* the executable (e.g. ``["-y", "-i", src, ...]``).

    If ``progress_cb`` is given, ``-progress pipe:1 -nostats`` is appended and
    stdout is streamed line-by-line; ``progress_cb(stats: dict)`` is invoked with
    the accumulated key/value block each time ffmpeg emits a ``progress=...``
    marker (``stats["progress"]`` is ``"continue"`` or ``"end"``; ``out_time_ms``
    / ``out_time_us`` carry the encoded position).

    ``should_abort`` is polled roughly every 250 ms; returning True kills the
    child and raises :class:`FFmpegAborted`. Without it a cancelled batch could
    only take effect BETWEEN segments — the worker thread was blocked inside a
    single un-interruptible ``subprocess.run``, so pressing Cancel did nothing
    until the current segment finished (or forever, if it never did).

    Returns the process exit code. Raises :class:`FFmpegError` on non-zero exit
    (with captured stderr tail) or timeout.
    """
    exe = resolve_ffmpeg()
    argv = [exe, *args]

    if progress_cb is not None:
        argv += ["-progress", "pipe:1", "-nostats"]

    logger.debug("ffmpeg_run", argv=argv)

    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    # BOTH pipes are drained on threads for the whole run. ffmpeg writes stream
    # dumps and per-packet warnings to stderr (non-monotonic DTS on stream-copy
    # splits is routine); with no reader it blocks on a full pipe buffer and the
    # parent waits forever for a process that is waiting for the parent.
    stdout_tail = PipeTail(proc.stdout) if progress_cb is None else None
    stderr_tail = PipeTail(proc.stderr)

    progress_thread = None
    if progress_cb is not None:
        progress_thread = threading.Thread(
            target=_pump_progress, args=(proc.stdout, progress_cb),
            daemon=True, name="ffmpeg_progress",
        )
        progress_thread.start()

    deadline = None if timeout is None else time.monotonic() + timeout
    try:
        while proc.poll() is None:
            if should_abort is not None and should_abort():
                kill_process(proc)
                raise FFmpegAborted("ffmpeg run aborted (task cancelled)")
            if deadline is not None and time.monotonic() > deadline:
                kill_process(proc)
                raise FFmpegError(f"ffmpeg timed out after {timeout}s")
            time.sleep(_POLL_INTERVAL_S)
        ret = proc.returncode
    finally:
        if proc.poll() is None:  # pragma: no cover - defensive
            kill_process(proc)

    if progress_thread is not None:
        progress_thread.join(timeout=5.0)

    if ret != 0:
        tail = stderr_tail.text() or (stdout_tail.text() if stdout_tail else "")
        raise FFmpegError(f"ffmpeg exited {ret}: {tail[-2000:]}")
    return ret


def _pump_progress(stream, progress_cb) -> None:
    """Read ``-progress pipe:1`` output and fire *progress_cb* per block."""
    stats: dict[str, str] = {}
    try:
        for raw in stream:
            parsed = _parse_progress_line(raw)
            if parsed is None:
                continue
            key, value = parsed
            stats[key] = value
            if key == "progress":
                try:
                    progress_cb(dict(stats))
                except Exception as exc:  # noqa: BLE001 - cb must not kill the run
                    logger.debug("ffmpeg_progress_cb_failed", error=str(exc))
                stats.clear()
    except Exception:  # noqa: BLE001 - pipe closed when the child was killed
        pass


def nearest_keyframe_before(path: str | Path, t: float) -> float:
    """Return the timestamp (seconds) of the keyframe at-or-before ``t``.

    Demuxes the first video stream's packets (no pixel decode) and returns the
    PTS of the latest keyframe whose presentation time is ``<= t``. Falls back
    to ``0.0`` when no keyframe precedes ``t`` (the clip starts on a keyframe) or
    the file cannot be opened — both safe defaults for the copy-vs-reencode
    decision (a 0.0 result simply means "not near this segment start").
    """
    import av  # lazy import — keeps non-video code paths ffmpeg-free

    path = Path(path)
    if t <= 0:
        return 0.0

    try:
        container = av.open(str(path))
    except Exception as exc:  # noqa: BLE001
        logger.debug("keyframe_open_failed", path=str(path), error=str(exc))
        return 0.0

    try:
        if not container.streams.video:
            return 0.0
        stream = container.streams.video[0]
        time_base = stream.time_base
        best = 0.0
        for packet in container.demux(stream):
            if not packet.is_keyframe or packet.pts is None or time_base is None:
                continue
            pkt_t = float(packet.pts * time_base)
            if pkt_t <= t + 1e-6:
                if pkt_t > best:
                    best = pkt_t
            else:
                # Packets are demuxed in roughly increasing PTS; once we pass t
                # we can stop scanning (keyframe spacing only grows from here).
                break
        return best
    except Exception as exc:  # noqa: BLE001
        logger.debug("keyframe_scan_failed", path=str(path), error=str(exc))
        return 0.0
    finally:
        try:
            container.close()
        except Exception:  # noqa: BLE001
            pass
