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

from torch import Tensor

# The LTX-2 audio VAE encoder consumes a 2-channel (stereo) mel.
_TARGET_CHANNELS = 2


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

    The decode itself is the engine-shared one
    (:func:`app.engine.components.audio_io.load_audio_waveform`): the window is
    a stretch of the file's presentation clock — the clock the video frames are
    selected on — so a soundtrack that starts late, early or has a hole in it
    lands where the file says it plays. This module's own copy sliced the
    decoded samples from index zero and paired such files with the wrong
    second of sound; a cache of its output is re-keyed by
    ``AUDIO_DECODE_VERSION`` (see ``trainer._audio_cache_version``).
    """
    from app.engine.components.audio_io import load_audio_waveform as _shared

    return _shared(
        path,
        trim_start_s=trim_start_s,
        duration_s=duration_s,
        target_sr=target_sr,
        channels=_TARGET_CHANNELS,
    )
