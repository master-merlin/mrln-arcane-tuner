"""LTX-2 training-side audio decode: clip → fixed-length STEREO waveform.

Pairs with :mod:`.audio_mel` (waveform → log-mel → clean audio latents). This
module owns the *I/O* half: pull the audio of a video clip's trim window via
PyAV and return a DETERMINISTIC-length stereo waveform so a batch of
equal-duration clips yields equal-length audio latents (required for batched
collation).

Stereo, not mono
~~~~~~~~~~~~~~~~
The LTX-2 audio VAE encoder (``AutoencoderKLLTX2Audio``) is **2-channel**
(``in_channels=2``): its first conv weight is ``[base, 2, 3, 3]``. Feeding a
mono (1-channel) mel raised "expected input to have 2 channels, but got 1". A
mono source is up-mixed to stereo (both channels identical) so the channel count
always matches the model.

Why a fixed length matters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The audio latent length ``L`` is a function of the waveform sample count (mel
time frames → VAE temporal compression). Two clips in the same temporal bucket
share ``target_frames`` / ``target_fps`` → the same *duration*, so emitting
exactly ``round(duration_s * target_sr)`` samples per channel (zero-padded if
the source is short, truncated if long) guarantees they encode to the same ``L``
and ``torch.stack`` succeeds. A clip with no audio stream returns ``None`` — an
"absent audio" item whose loss is masked to zero downstream.
"""

from __future__ import annotations

import structlog
import torch
from torch import Tensor

logger = structlog.get_logger(__name__)

# The LTX-2 audio VAE encoder consumes a 2-channel (stereo) mel.
_TARGET_CHANNELS = 2


def _resample_frames(resampler, frame) -> list:
    """Normalize ``AudioResampler.resample`` to a list across PyAV versions.

    PyAV ≥ 9 returns a list of frames; older builds return a single frame or
    ``None``. Returning a uniform list lets the caller iterate unconditionally.
    """
    out = resampler.resample(frame)
    if out is None:
        return []
    return out if isinstance(out, list) else [out]


def _to_stereo(arr) -> "object":
    """Normalize a decoded audio frame to a planar ``[2, samples]`` array."""
    import numpy as np

    arr = np.atleast_2d(arr)
    if arr.shape[0] == 1:  # mono → duplicate into both channels
        arr = np.repeat(arr, _TARGET_CHANNELS, axis=0)
    return arr[:_TARGET_CHANNELS]


def load_audio_waveform(
    path: str,
    *,
    trim_start_s: float,
    duration_s: float,
    target_sr: int,
) -> tuple[Tensor, int] | None:
    """Decode a clip's audio trim window → ``([2, N], target_sr)`` stereo in [-1, 1].

    ``N = round(duration_s * target_sr)`` exactly per channel — zero-padded when
    the source is shorter, truncated when longer — so equal-duration clips
    produce equal-length waveforms (stackable audio latents). Output is always
    2-channel (the LTX-2 audio VAE is stereo): a mono source is up-mixed, >2
    channels are truncated. Resampled to ``target_sr`` via PyAV's planar
    ``AudioResampler``.

    Returns ``None`` when the file has no audio stream.

    Note: training clips are short (a few seconds), so the whole audio stream is
    decoded and then sliced by sample index — this sidesteps keyframe-seek offset
    bugs that would misalign the trim window.
    """
    import av
    import numpy as np

    if duration_s <= 0 or target_sr <= 0:
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

        # Planar stereo: to_ndarray() → [2, samples] per frame; a mono source is
        # up-mixed to stereo by the resampler (both channels carry the signal).
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout="stereo", rate=int(target_sr),
        )
        parts: list[np.ndarray] = []
        for frame in container.decode(stream):
            for rframe in _resample_frames(resampler, frame):
                parts.append(_to_stereo(rframe.to_ndarray()))
        # Flush any buffered samples held by the resampler.
        for rframe in _resample_frames(resampler, None):
            parts.append(_to_stereo(rframe.to_ndarray()))

        parts = [p for p in parts if p.size]
        if not parts:
            return None
        wav = np.concatenate(parts, axis=1).astype("float32")  # [2, total]
    finally:
        try:
            container.close()
        except Exception:  # noqa: BLE001
            pass

    n = int(round(duration_s * target_sr))
    start_i = max(int(round(max(trim_start_s, 0.0) * target_sr)), 0)
    seg = wav[:, start_i : start_i + n]

    out = torch.zeros(_TARGET_CHANNELS, n, dtype=torch.float32)
    if seg.shape[1]:
        out[:, : seg.shape[1]] = torch.from_numpy(seg[:, :n])
    return out.clamp_(-1.0, 1.0), int(target_sr)  # [2, N] stereo
