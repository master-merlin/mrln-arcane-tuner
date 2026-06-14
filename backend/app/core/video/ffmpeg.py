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
from functools import lru_cache
from pathlib import Path

import structlog

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


def run_ffmpeg(
    args: list[str],
    progress_cb=None,
    *,
    timeout: float | None = None,
) -> int:
    """Run ffmpeg with an explicit argv list (never via a shell).

    The resolved ffmpeg binary is prepended automatically — ``args`` should be
    everything *after* the executable (e.g. ``["-y", "-i", src, ...]``).

    If ``progress_cb`` is given, ``-progress pipe:1 -nostats`` is appended and
    stdout is streamed line-by-line; ``progress_cb(stats: dict)`` is invoked with
    the accumulated key/value block each time ffmpeg emits a ``progress=...``
    marker (``stats["progress"]`` is ``"continue"`` or ``"end"``; ``out_time_ms``
    / ``out_time_us`` carry the encoded position).

    Returns the process exit code. Raises :class:`FFmpegError` on non-zero exit
    (with captured stderr tail) or timeout.
    """
    exe = resolve_ffmpeg()
    argv = [exe, *args]

    if progress_cb is not None:
        argv += ["-progress", "pipe:1", "-nostats"]

    logger.debug("ffmpeg_run", argv=argv)

    if progress_cb is None:
        try:
            proc = subprocess.run(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - timing
            raise FFmpegError(f"ffmpeg timed out after {timeout}s") from exc
        if proc.returncode != 0:
            tail = (proc.stderr or "")[-2000:]
            raise FFmpegError(f"ffmpeg exited {proc.returncode}: {tail}")
        return proc.returncode

    # Streaming-progress mode.
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stats: dict[str, str] = {}
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
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
        ret = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - timing
        proc.kill()
        raise FFmpegError(f"ffmpeg timed out after {timeout}s") from exc
    finally:
        stderr_tail = ""
        if proc.stderr is not None:
            stderr_tail = proc.stderr.read()[-2000:]

    if ret != 0:
        raise FFmpegError(f"ffmpeg exited {ret}: {stderr_tail}")
    return ret


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
