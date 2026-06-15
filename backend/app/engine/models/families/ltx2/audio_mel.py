"""LTX-2 audio feature extraction + latent packing (training-side).

The diffusers ``AutoencoderKLLTX2Audio`` encoder consumes a **log-mel
spectrogram** ``(B, C, time, mel_bins)`` — there is NO waveform→mel transform in
diffusers, so to produce audio *training targets* we replicate the exact
transform from Lightricks' own trainer package (``ltx-core`` ``audio_vae/ops.py``):

    MelSpectrogram(sample_rate=16000, n_fft=1024, win_length=1024,
                   hop_length=160, f_min=0, f_max=8000, n_mels=64,
                   window_fn=hann, center=True, pad_mode="reflect",
                   power=1.0, mel_scale="slaney", norm="slaney")
    mel = log(clamp(mel, 1e-5))            # log-mel
    mel = mel.permute(0, 1, 3, 2)          # → (B, C, time, n_mels)

Getting these parameters wrong silently corrupts the target latents, so they are
pinned here verbatim and asserted in tests.

The clean training target is then::

    latent  = audio_vae.encode(mel).latent_dist.mode()   # [B, 8, L, 16]
    packed  = pack_audio_latents(latent)                 # [B, L, 8*16=128]
    clean   = normalize_audio_latents(packed, mean, std) # per-128-feature

matching ``LTX2Pipeline.prepare_audio_latents`` (pack THEN normalize; the VAE's
``latents_mean``/``latents_std`` buffers are 128-wide, i.e. per packed feature).
"""

from __future__ import annotations

import torch
from torch import Tensor

from .audio import extract_audio_latents

# ── Exact mel contract (Lightricks ltx-core audio_vae/ops.py) ──────────────
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_N_FFT = 1024
DEFAULT_MEL_HOP = 160
DEFAULT_MEL_BINS = 64


class AudioMelExtractor:
    """Waveform → log-mel spectrogram, matching the LTX-2 audio VAE's training.

    Lazily builds a ``torchaudio`` ``MelSpectrogram`` so importing this module
    never requires torchaudio (only constructing the extractor does).
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_fft: int = DEFAULT_N_FFT,
        mel_hop_length: int = DEFAULT_MEL_HOP,
        mel_bins: int = DEFAULT_MEL_BINS,
    ) -> None:
        import torchaudio

        self.target_sample_rate = int(sample_rate)
        self._resample = torchaudio.functional.resample
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=mel_hop_length,
            f_min=0.0,
            f_max=sample_rate / 2.0,
            n_mels=mel_bins,
            window_fn=torch.hann_window,
            center=True,
            pad_mode="reflect",
            power=1.0,
            mel_scale="slaney",
            norm="slaney",
        )

    def to(self, device) -> "AudioMelExtractor":
        self.mel = self.mel.to(device)
        return self

    def waveform_to_mel(self, waveform: Tensor, sample_rate: int) -> Tensor:
        """``[B, C, samples]`` waveform in [-1, 1] → log-mel ``[B, C, time, n_mels]``.

        Resamples to the model's 16 kHz first when needed.
        """
        if int(sample_rate) != self.target_sample_rate:
            waveform = self._resample(waveform, sample_rate, self.target_sample_rate)
        mel = self.mel(waveform)  # [B, C, n_mels, time]
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.permute(0, 1, 3, 2).contiguous()  # [B, C, time, n_mels]


# ── Latent packing / normalization (verbatim from LTX2Pipeline) ────────────


def pack_audio_latents(latents: Tensor) -> Tensor:
    """``[B, C, L, M]`` → ``[B, L, C*M]`` (mel patch = all bins, temporal patch = 1)."""
    return latents.transpose(1, 2).flatten(2, 3)


def unpack_audio_latents(latents: Tensor, num_mel_bins: int) -> Tensor:
    """``[B, L, C*M]`` → ``[B, C, L, M]`` (inverse of :func:`pack_audio_latents`)."""
    return latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)


def normalize_audio_latents(latents: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Per-feature ``(x - mean) / std`` (applied to the PACKED 128-dim latent)."""
    mean = mean.to(latents.device, latents.dtype)
    std = std.to(latents.device, latents.dtype)
    return (latents - mean) / std


def denormalize_audio_latents(latents: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Inverse of :func:`normalize_audio_latents` (for the sampler's decode)."""
    mean = mean.to(latents.device, latents.dtype)
    std = std.to(latents.device, latents.dtype)
    return latents * std + mean


def encode_clean_audio_latents(audio_vae, mel: Tensor) -> Tensor:
    """Mel ``[B, C, T, 64]`` → clean (packed + normalized) audio latents ``[B, L, 128]``.

    The flow-match *training target*: VAE-encode (mode), pack the mel bins into
    the feature axis, then per-feature normalize with the VAE's stored stats.
    Mirrors ``LTX2Pipeline.prepare_audio_latents`` (pack THEN normalize).
    """
    latent = extract_audio_latents(audio_vae, mel)  # [B, 8, L, 16]
    packed = pack_audio_latents(latent)  # [B, L, 128]
    mean = getattr(audio_vae, "latents_mean", None)
    std = getattr(audio_vae, "latents_std", None)
    if mean is not None and std is not None:
        packed = normalize_audio_latents(packed, mean, std)
    return packed
