"""PyAV-backed frame sampling for video captioning.

``sample_frames(path, n, start_s, end_s)`` extracts ``n`` evenly-spaced RGB
frames from a clip — the input the video-captioning VLM lane consumes. It seeks
to each target timestamp and decodes the nearest frame, returning PIL images.

The sampler is deliberately tolerant: clips shorter than ``n`` frames yield
whatever is available (deduplicated), and a clip that cannot be decoded raises
:class:`FrameSampleError`, which the caption service catches to fail a single
item rather than the whole batch. PyAV bundles its own ffmpeg, so this behaves
identically on Windows and inside the Docker image.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import structlog
from PIL import Image

logger = structlog.get_logger(__name__)


class FrameSampleError(RuntimeError):
    """Raised when a video file cannot be opened or no frame can be sampled."""


def _clip_duration_s(container, stream) -> float:
    """Best-effort clip duration in seconds (mirrors probe._stream_duration_s)."""
    _AV_TIME_BASE = 1_000_000
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


def _target_timestamps(
    duration_s: float, n: int, start_s: float | None, end_s: float | None
) -> list[float]:
    """Compute up to *n* evenly-spaced sample timestamps within the window.

    The window defaults to the whole clip ``[0, duration_s]``; ``start_s`` /
    ``end_s`` clamp it. Samples are placed at the centre of *n* equal sub-bins
    so the first/last frames aren't pinned to the exact boundaries (which can
    decode the same edge frame twice). Returns a sorted, ascending list.
    """
    lo = max(0.0, float(start_s)) if start_s is not None else 0.0
    hi = float(end_s) if end_s is not None else duration_s
    if hi <= lo or duration_s <= 0:
        # Degenerate window (zero/negative span or unknown duration): sample
        # from the start; the decode loop returns whatever frames exist.
        return [0.0] * max(1, n)
    span = hi - lo
    if n <= 1:
        return [lo + span / 2.0]
    # Centre-of-bin placement: (i + 0.5) / n across the span.
    return [lo + span * (i + 0.5) / n for i in range(n)]


def sample_frames(
    path: str | Path,
    n: int = 8,
    start_s: float | None = None,
    end_s: float | None = None,
) -> list[Image.Image]:
    """Return up to *n* evenly-spaced RGB frames from the clip at *path*.

    Frames are sampled within ``[start_s, end_s]`` (defaults to the whole
    clip). For each target timestamp the decoder seeks to the nearest preceding
    keyframe and decodes forward to the first frame at/after the timestamp.

    Robustness:
    - ``n`` larger than the available frame count returns only what exists
      (duplicate decodes for the same source frame are collapsed).
    - At least one frame is always attempted; a clip with no decodable frame
      raises :class:`FrameSampleError`.

    Args:
        path: Path to a video clip (mp4/webm/mkv/avi).
        n: Number of frames to sample (>= 1).
        start_s: Window start in seconds (default 0).
        end_s: Window end in seconds (default clip duration).

    Returns:
        A list of RGB :class:`PIL.Image.Image` in ascending time order.

    Raises:
        FrameSampleError: the file is missing/unreadable or yields no frame.
    """
    import av  # imported lazily so non-video paths never pull in ffmpeg

    n = max(1, int(n))
    path = Path(path)

    try:
        container = av.open(str(path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("frame_sample_open_failed", path=str(path), error=str(exc))
        raise FrameSampleError(f"could not open video: {path}") from exc

    try:
        if not container.streams.video:
            raise FrameSampleError(f"no video stream in {path}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        duration_s = _clip_duration_s(container, stream)
        timestamps = _target_timestamps(duration_s, n, start_s, end_s)

        time_base = stream.time_base
        frames: list[Image.Image] = []
        seen_pts: set[int] = set()

        for ts in timestamps:
            try:
                if time_base is not None and duration_s > 0:
                    seek_pts = int(ts / float(time_base))
                    container.seek(seek_pts, stream=stream, any_frame=False)
                else:
                    container.seek(0)
            except Exception as exc:  # noqa: BLE001
                logger.debug("frame_seek_failed", path=str(path), ts=ts, error=str(exc))

            picked = None
            for frame in container.decode(stream):
                picked = frame
                frame_t = (
                    float(frame.pts * time_base)
                    if frame.pts is not None and time_base is not None
                    else None
                )
                if frame_t is None or frame_t >= ts:
                    break
            if picked is None:
                continue
            # Collapse duplicate source frames (short clips / coarse seeks).
            if picked.pts is not None:
                if picked.pts in seen_pts:
                    continue
                seen_pts.add(picked.pts)
            img = picked.to_image()  # RGB PIL image
            if img.mode != "RGB":
                img = img.convert("RGB")
            frames.append(img)

        if not frames:
            raise FrameSampleError(f"no decodable frames in {path}")

        logger.debug(
            "frames_sampled", path=str(path), requested=n, returned=len(frames)
        )
        return frames
    except FrameSampleError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("frame_sample_failed", path=str(path), error=str(exc))
        raise FrameSampleError(f"failed to sample frames: {path}") from exc
    finally:
        try:
            container.close()
        except Exception:  # noqa: BLE001
            pass


def frame_to_data_url(
    image: Image.Image, max_long_side: int | None = None, quality: int = 90
) -> str:
    """JPEG-encode a PIL frame and wrap it as a ``data:image/jpeg`` base64 URL."""
    img = image
    if max_long_side and max(img.size) > max_long_side:
        img = img.copy()
        img.thumbnail((max_long_side, max_long_side), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def sample_frames_as_data_urls(
    path: str | Path,
    n: int = 8,
    start_s: float | None = None,
    end_s: float | None = None,
    max_long_side: int | None = 1024,
    quality: int = 90,
) -> list[str]:
    """Sample frames and return them as base64 ``data:image/jpeg`` URLs.

    Convenience wrapper over :func:`sample_frames` for the OpenAI-compatible
    API lane, which embeds frames as ``image_url`` data URLs.
    """
    return [
        frame_to_data_url(frame, max_long_side=max_long_side, quality=quality)
        for frame in sample_frames(path, n=n, start_s=start_s, end_s=end_s)
    ]
