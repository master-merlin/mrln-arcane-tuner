"""PyAV-backed video probing.

``probe_video(path)`` opens a clip and extracts the metadata the video-LoRA
ingest layer needs — dimensions, framerate, duration, frame count (exact or
estimated), audio presence and codec — without decoding any pixels. PyAV
bundles its own ffmpeg, so this behaves the same on Windows and in Docker.

The probe is deliberately tolerant: a malformed file raises
:class:`VideoProbeError` (logged), which callers catch to fall back to
best-effort metadata rather than failing a whole dataset scan.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import structlog

try:  # pragma: no cover - import guard
    from pydantic import BaseModel, ConfigDict
except Exception:  # pragma: no cover
    BaseModel = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)

# PyAV reports ``container.duration`` in units of this constant (microseconds).
_AV_TIME_BASE = 1_000_000


class VideoProbeError(RuntimeError):
    """Raised when a video file cannot be opened or probed."""


class VideoProbe(BaseModel):  # type: ignore[misc]
    """Immutable per-clip probe result.

    ``frame_count_estimated`` is ``True`` when the container exposed no exact
    frame count and we derived it from ``round(duration * fps)``.
    """

    model_config = ConfigDict(frozen=True)

    width: int
    height: int
    fps: float
    duration_s: float
    frame_count: int
    frame_count_estimated: bool
    has_audio: bool
    video_codec: str | None


def _stream_duration_s(container, stream) -> float:
    """Best-effort clip duration in seconds.

    Prefers the video stream's own ``duration`` (scaled by its ``time_base``)
    and falls back to the container duration (microseconds). Returns ``0.0``
    when neither is available.
    """
    if stream.duration is not None and stream.time_base is not None:
        try:
            return float(stream.duration * stream.time_base)
        except (TypeError, ValueError):
            pass
    if container.duration is not None:
        try:
            return float(container.duration) / _AV_TIME_BASE
        except (TypeError, ValueError):
            pass
    return 0.0


def probe_video(path: Path) -> VideoProbe:
    """Probe *path* and return a :class:`VideoProbe`.

    Raises:
        VideoProbeError: the file is missing, has no video stream, or cannot
            be decoded. The original error is chained and logged.
    """
    import av  # imported lazily so non-video code paths never pull in ffmpeg

    path = Path(path)
    try:
        container = av.open(str(path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("video_probe_open_failed", path=str(path), error=str(exc))
        raise VideoProbeError(f"could not open video: {path}") from exc

    try:
        if not container.streams.video:
            raise VideoProbeError(f"no video stream in {path}")
        stream = container.streams.video[0]

        # fps — average_rate is a Fraction; guard against a missing/zero rate.
        avg_rate = stream.average_rate
        fps = float(avg_rate) if avg_rate and avg_rate != Fraction(0) else 0.0

        duration_s = _stream_duration_s(container, stream)

        cc = stream.codec_context
        width = int(cc.width or stream.width or 0)
        height = int(cc.height or stream.height or 0)
        video_codec = cc.name or None

        # frame_count — prefer the container's exact count; estimate otherwise.
        raw_frames = stream.frames or 0
        if raw_frames > 0:
            frame_count = int(raw_frames)
            frame_count_estimated = False
        elif fps > 0 and duration_s > 0:
            frame_count = int(round(duration_s * fps))
            frame_count_estimated = True
        else:
            frame_count = 0
            frame_count_estimated = True

        has_audio = bool(container.streams.audio)

        return VideoProbe(
            width=width,
            height=height,
            fps=fps,
            duration_s=duration_s,
            frame_count=frame_count,
            frame_count_estimated=frame_count_estimated,
            has_audio=has_audio,
            video_codec=video_codec,
        )
    except VideoProbeError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("video_probe_failed", path=str(path), error=str(exc))
        raise VideoProbeError(f"failed to probe video: {path}") from exc
    finally:
        try:
            container.close()
        except Exception:  # noqa: BLE001
            pass
