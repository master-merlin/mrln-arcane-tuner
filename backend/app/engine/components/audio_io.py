"""Training-side audio decode: a clip's trim window -> fixed-length stereo waveform.

Engine-shared (``components/``) so a family never imports a sibling family's
module for it (the layering rule): ``ltx2/audio_io.py`` carries the same
contract for LTX-2 and can adopt this one later. Decoding is PyAV (Class A,
``av==16.1.0``): the whole (short) audio stream is decoded and sliced by
sample index, which sidesteps keyframe-seek offsets that would misalign the
window against the video frames.

Contract: ``([channels, N], target_sr)`` with ``N = round(duration_s *
target_sr)`` EXACTLY — zero-padded when the source is shorter, truncated when
longer — so equal-duration clips give equal-length waveforms (stackable
latents). A mono source is up-mixed, more channels are truncated. ``None``
when the file has no audio stream (the caller masks that clip's audio loss).
"""

from __future__ import annotations

import structlog
import torch
from torch import Tensor

logger = structlog.get_logger(__name__)


def _as_list(resampled) -> list:
    """``AudioResampler.resample`` returns a list on PyAV >= 9, a frame or
    ``None`` on older builds — iterate uniformly."""
    if resampled is None:
        return []
    return resampled if isinstance(resampled, list) else [resampled]


def _planar(arr, channels: int):
    import numpy as np

    arr = np.atleast_2d(arr)
    if arr.shape[0] == 1 and channels > 1:
        arr = np.repeat(arr, channels, axis=0)
    return arr[:channels]


def load_audio_waveform(
    path: str,
    *,
    trim_start_s: float,
    duration_s: float,
    target_sr: int,
    channels: int = 2,
) -> tuple[Tensor, int] | None:
    """Decode ``path``'s audio from ``trim_start_s`` for ``duration_s`` seconds
    at ``target_sr`` -> ``([channels, N], target_sr)`` in ``[-1, 1]``, or
    ``None`` when there is no audio stream."""
    import av
    import numpy as np

    if duration_s <= 0 or target_sr <= 0 or channels not in (1, 2):
        return None
    try:
        container = av.open(str(path))
    except (OSError, ValueError) as e:
        logger.warning("audio_open_failed", path=str(path), error=str(e))
        return None
    try:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout="stereo" if channels == 2 else "mono", rate=int(target_sr)
        )
        parts: list = []
        for frame in container.decode(stream):
            for rframe in _as_list(resampler.resample(frame)):
                parts.append(_planar(rframe.to_ndarray(), channels))
        for rframe in _as_list(resampler.resample(None)):  # flush
            parts.append(_planar(rframe.to_ndarray(), channels))
        parts = [p for p in parts if p.size]
        if not parts:
            return None
        wav = np.concatenate(parts, axis=1).astype("float32")
    finally:
        try:
            container.close()
        except Exception:  # noqa: BLE001 - closing is best effort
            pass

    n = int(round(duration_s * target_sr))
    start = max(int(round(max(trim_start_s, 0.0) * target_sr)), 0)
    segment = wav[:, start : start + n]
    out = torch.zeros(channels, n, dtype=torch.float32)
    if segment.shape[1]:
        out[:, : segment.shape[1]] = torch.from_numpy(segment[:, :n])
    return out.clamp_(-1.0, 1.0), int(target_sr)
