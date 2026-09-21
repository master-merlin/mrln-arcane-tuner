"""Training-side audio decode: a clip's trim window -> fixed-length stereo waveform.

Engine-shared (``components/``) so a family never imports a sibling family's
module for it (the layering rule); every family that trains a clip's
soundtrack decodes it here. Decoding is PyAV (Class A,
``av==16.1.0``): the whole (short) audio stream is decoded without a seek,
which sidesteps keyframe-seek offsets.

ONE timeline. ``trim_start_s`` is a time on the file's presentation clock —
the seconds ``frame.time`` reports, the clock ``VideoFrameLoader.load_clip``
selects its frames on — NOT an index into the decoded samples. The decoded
audio is PLACED by its timestamps before the window is cut: a soundtrack that
starts late is silence until it starts, samples presented before zero (encoder
priming kept by some containers) are dropped, a hole in the timestamps stays a
hole (and is reported), and a stream without timestamps is its own clock from
zero. A stream that starts at zero and runs without a hole decodes to exactly
the samples it always did.

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

# What a cache of anything derived from this decode folds into its key: bump
# it whenever the SAMPLES a given (file, window) decodes to can change.
#   1 (implicit, never written) - concatenate the decoded frames, slice from 0
#   2 - place by presentation timestamp, then cut the window
AUDIO_DECODE_VERSION = 2

# Timestamps closer than this to where the previous frame ended are the same
# run. Containers that store milliseconds jitter by under a millisecond per
# frame; one video frame at 60 fps is 16.7 ms, so a slip a viewer could see is
# always a jump.
TIMELINE_GAP_TOLERANCE_S = 0.010


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


def _flush(resampler, channels: int) -> list:
    return [_planar(r.to_ndarray(), channels) for r in _as_list(resampler.resample(None))]


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
        # A RUN is a stretch of frames whose timestamps follow each other; it is
        # anchored at the presentation time of its first frame. A new run (and
        # a fresh resampler, so no sample leaks across the hole) starts wherever
        # the timestamps jump by more than the tolerance.
        runs: list[tuple[float, list]] = []
        jumps: list[tuple[float, float]] = []
        expected: float | None = None
        for frame in container.decode(stream):
            at = float(frame.time) if frame.time is not None else expected
            if at is None:
                at = 0.0  # no timestamps at all: the stream is its own clock
            if expected is None or abs(at - expected) > TIMELINE_GAP_TOLERANCE_S:
                if runs:
                    jumps.append((expected, at - expected))
                    runs[-1][1].extend(_flush(resampler, channels))
                    resampler = av.audio.resampler.AudioResampler(
                        format="fltp", layout="stereo" if channels == 2 else "mono", rate=int(target_sr)
                    )
                runs.append((at, []))
                expected = at
            for rframe in _as_list(resampler.resample(frame)):
                runs[-1][1].append(_planar(rframe.to_ndarray(), channels))
            expected += frame.samples / float(frame.sample_rate or target_sr)
        if runs:
            runs[-1][1].extend(_flush(resampler, channels))
        placed = [
            (anchor, np.concatenate(parts, axis=1).astype("float32"))
            for anchor, parts in ((a, [p for p in ps if p.size]) for a, ps in runs)
            if parts
        ]
        if not placed:
            return None
    finally:
        try:
            container.close()
        except Exception:  # noqa: BLE001 - closing is best effort
            pass

    if jumps:
        logger.warning(
            "audio_timeline_discontinuity",
            path=str(path),
            count=len(jumps),
            first_at_s=round(jumps[0][0], 4),
            first_jump_s=round(jumps[0][1], 4),
            hint="the timestamps are kept: a hole is silence, an overlap is overwritten by the later samples",
        )

    n = int(round(duration_s * target_sr))
    start = max(int(round(max(trim_start_s, 0.0) * target_sr)), 0)
    out = torch.zeros(channels, n, dtype=torch.float32)
    for anchor, wav in placed:
        # Sample index of this run on the file's timeline, then its overlap
        # with the window [start, start + n).
        origin = int(round(anchor * target_sr))
        lo, hi = max(origin, start), min(origin + wav.shape[1], start + n)
        if hi > lo:
            out[:, lo - start : hi - start] = torch.from_numpy(wav[:, lo - origin : hi - origin])
    return out.clamp_(-1.0, 1.0), int(target_sr)
