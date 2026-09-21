"""MiniMax-H3 stereo audio latents — the mono VAE run twice (plan row 2.2).

``AutoencoderKLMiniMaxH3Audio`` (diffusers 0.40.0) is MONO: ``encode`` takes
``[batch, 1, samples]`` and ``decode`` returns ``[batch, 1, frames * 800]``;
the reference pipeline carries stereo as ``batch = 2`` and consumes the
posterior MEAN (``latent_dist.mode()``), never a sample. Latents are
normalised per channel with the config's ``latents_mean`` / ``latents_std``
(32 floats each) — the VAE applies neither, the caller does (its module
docstring). This module is that caller for training:

- :func:`encode_stereo` ``(B, 2, N) → (B, 2, C, T)`` normalised, ``T = N / 800``
  (40 latents per second at 32 kHz — ``audio.latent_rate`` in the YAML).
- :func:`decode_stereo` — the inverse, denormalising first.
- :func:`stereo_to_rows` / :func:`rows_to_stereo` — the packed-sequence
  layout, CHANNEL-MAJOR along time: ``(B, 2, C, T) → (B, 2·T, C)``, all T
  latents of L then all of R (A-5, CONFIRMED by ai-toolkit
  ``pack_audio_latents``, research §3.3; re-implemented in ``packing.py``,
  not copied). Concatenating along channels instead would produce
  64-wide rows that the transformer's ``audio_in_channels: 32`` rejects.

Production callers (owned by row 2.5): the encode side runs from
``_pre_cache_aux`` while the audio VAE is resident; the train side from
``build_batch_extra``. This module ships the arithmetic and its proof only.
"""

from __future__ import annotations

from typing import Any

import torch

from .packing import AUDIO_CHANNELS, pack_audio, unpack_audio

AUDIO_LATENT_VERSION = 1


def _stats(audio_vae: Any, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = audio_vae.config
    mean = torch.as_tensor(cfg.latents_mean, dtype=like.dtype, device=like.device)
    std = torch.as_tensor(cfg.latents_std, dtype=like.dtype, device=like.device)
    return mean.view(1, -1, 1), std.view(1, -1, 1)


def normalize(audio_vae: Any, raw: torch.Tensor) -> torch.Tensor:
    """``(x − latents_mean) / latents_std`` per latent channel on ``(.., C, T)``."""
    mean, std = _stats(audio_vae, raw)
    return (raw - mean) / std


def denormalize(audio_vae: Any, latents: torch.Tensor) -> torch.Tensor:
    mean, std = _stats(audio_vae, latents)
    return latents * std + mean


def encode_stereo(audio_vae: Any, waveform: torch.Tensor) -> torch.Tensor:
    """``(B, 2, N)`` waveform in ``[-1, 1]`` → ``(B, 2, C, T)`` normalised latents.

    Both channels of every clip go through ONE encoder call as ``2·B`` mono
    items (the pipeline's own batching), so L and R are encoded identically
    and independently; the posterior mean is taken, as the pipeline does.
    """
    if waveform.ndim != 3 or waveform.shape[1] != AUDIO_CHANNELS:
        raise ValueError(
            f"stereo waveform must be (B, {AUDIO_CHANNELS}, samples), got {tuple(waveform.shape)}"
        )
    b = waveform.shape[0]
    mono = waveform.reshape(b * AUDIO_CHANNELS, 1, waveform.shape[-1])
    raw = audio_vae.encode(mono).latent_dist.mode()  # (2B, C, T)
    latents = normalize(audio_vae, raw)
    return latents.reshape(b, AUDIO_CHANNELS, latents.shape[-2], latents.shape[-1])


def decode_stereo(audio_vae: Any, latents: torch.Tensor) -> torch.Tensor:
    """``(B, 2, C, T)`` normalised latents → ``(B, 2, T·800)`` waveform."""
    if latents.ndim != 4 or latents.shape[1] != AUDIO_CHANNELS:
        raise ValueError(
            f"stereo latents must be (B, {AUDIO_CHANNELS}, C, T), got {tuple(latents.shape)}"
        )
    b, _, c, t = latents.shape
    raw = denormalize(audio_vae, latents.reshape(b * AUDIO_CHANNELS, c, t))
    decoded = audio_vae.decode(raw).sample  # (2B, 1, samples)
    return decoded.reshape(b, AUDIO_CHANNELS, decoded.shape[-1])


def stereo_to_rows(latents: torch.Tensor) -> torch.Tensor:
    """``(B, 2, C, T) → (B, 2·T, C)`` — channel-major along time (A-5)."""
    return pack_audio(latents)


def rows_to_stereo(rows: torch.Tensor, audio_latents: int) -> torch.Tensor:
    """Inverse of :func:`stereo_to_rows`."""
    return unpack_audio(rows, audio_latents)
