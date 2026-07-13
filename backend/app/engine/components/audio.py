"""Audio waveform loading — the audio-modality analogue of ``components/video.py``.

``AudioClipLoader.load_clip`` returns a FIXED-shape waveform tensor regardless
of the source file's actual length (trim for long clips, zero-pad silence for
short ones) — mirroring ``VideoFrameLoader.load_clip``'s fixed-output-shape
contract so a batch of items always ``torch.stack``s cleanly. Output is
``[C, T]`` float32 in ``[-1, 1]``, resampled + channel-matched to the family's
VAE contract (ACE-Step 1.5: 48 kHz stereo).

No new pip dependency: ``soundfile`` (already used by ``app.core.audio.probe``)
does the file I/O, ``torchaudio.functional.resample`` (already a transitive dep
of the torch/torchaudio stack pinned in requirements.txt) does resampling.
"""

from __future__ import annotations

import math

import torch


class AudioClipLoader:
    """Loads one fixed-length audio window as a normalized waveform tensor."""

    def load_clip(
        self,
        path: str,
        *,
        target_sample_rate: int,
        target_channels: int,
        window_s: float,
        trim_start_s: float = 0.0,
    ) -> torch.Tensor:
        """Read *path*, resample/channel-match, and fit to ``window_s`` seconds.

        Args:
            path: Source audio file path (any libsndfile-readable format).
            target_sample_rate: Output sample rate (Hz) — the VAE's contract.
            target_channels: Output channel count — the VAE's contract.
            window_s: Output window length in seconds. Longer sources are
                trimmed (from ``trim_start_s``); shorter sources are
                zero-padded (silence) at the tail.
            trim_start_s: Offset into the source file to start reading, in
                SOURCE seconds (before resampling).

        Returns:
            ``[C, T]`` float32 tensor in ``[-1, 1]``, ``T ==
            round(window_s * target_sample_rate)`` exactly.
        """
        import soundfile as sf

        info = sf.info(str(path))
        source_sr = int(info.samplerate) or target_sample_rate
        source_channels = max(int(info.channels), 1)
        window_samples = max(int(round(window_s * target_sample_rate)), 1)

        start_frame = max(int(round(trim_start_s * source_sr)), 0)
        # Read a small margin beyond the exact window (rounding/resample slop);
        # _fit_window trims to the exact sample count afterwards.
        read_frames = int(math.ceil(window_s * source_sr)) + max(source_sr // 10, 1)
        total_frames = int(info.frames)
        stop_frame = min(start_frame + read_frames, total_frames)
        n = max(stop_frame - start_frame, 0)

        if n <= 0:
            wav = torch.zeros((source_channels, 0), dtype=torch.float32)
        else:
            raw, _ = sf.read(
                str(path),
                start=start_frame,
                frames=n,
                dtype="float32",
                always_2d=True,
            )  # [T, C]
            wav = torch.from_numpy(raw).T.contiguous().float()  # [C, T]

        wav = self._match_channels(wav, target_channels)
        if source_sr != target_sample_rate and wav.shape[-1] > 0:
            wav = self._resample(wav, source_sr, target_sample_rate)

        return self._fit_window(wav, window_samples)

    # ── helpers (unit-tested independently) ────────────────────────────────

    @staticmethod
    def _match_channels(wav: torch.Tensor, target_channels: int) -> torch.Tensor:
        """Mono<->stereo (or N-channel) conversion by averaging/repeating."""
        c = wav.shape[0]
        if c == target_channels:
            return wav
        if c == 0 or wav.shape[-1] == 0:
            return torch.zeros((target_channels, wav.shape[-1]), dtype=wav.dtype)
        if target_channels == 1:
            return wav.mean(dim=0, keepdim=True)
        # Upmix mono -> N (repeat), or downmix multi-channel -> N (average then
        # broadcast) — audio datasets are overwhelmingly mono or stereo, so a
        # single averaging fallback covers every other source-channel count.
        mono = wav.mean(dim=0, keepdim=True)
        return mono.expand(target_channels, -1).contiguous()

    @staticmethod
    def _resample(wav: torch.Tensor, source_sr: int, target_sr: int) -> torch.Tensor:
        import torchaudio

        return torchaudio.functional.resample(wav, source_sr, target_sr)

    @staticmethod
    def _fit_window(wav: torch.Tensor, window_samples: int) -> torch.Tensor:
        c, t = wav.shape
        if t == window_samples:
            return wav.clamp(-1.0, 1.0)
        if t > window_samples:
            return wav[:, :window_samples].clamp(-1.0, 1.0)
        pad = torch.zeros((c, window_samples - t), dtype=wav.dtype)
        return torch.cat([wav, pad], dim=1).clamp(-1.0, 1.0)


def round_duration_bucket(duration_s: float, cap_s: float, step_s: float = 5.0) -> float:
    """Round a clip's usable duration into a coarse bucket (seconds).

    Mirrors the spirit of the video frame ladder: clips within the same
    ``step_s``-wide bucket (capped at ``cap_s``) share a batch/cache bucket so
    ``torch.stack`` works at ``train_batch_size > 1`` without ragged tensors.
    Always returns a value in ``(0, cap_s]``, rounded UP to the nearest
    ``step_s`` (so a clip is never truncated below its rounded bucket).
    """
    d = max(min(float(duration_s), float(cap_s)), 0.0)
    if d <= 0.0:
        return min(step_s, cap_s)
    bucketed = math.ceil(d / step_s) * step_s
    return min(bucketed, float(cap_s))
