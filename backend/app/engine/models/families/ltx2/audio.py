"""LTX 2.3 audio latent extraction, caching, and masking helpers.

This module is deliberately *pure-ish*: every function takes plain tensors (or a
VAE-like object exposing ``.encode``) and returns plain tensors, so the masking
math — the novel correctness point of joint audio+video training — is unit
testable with fakes and tiny tensors, no weights and no GPU.

The masking contract
~~~~~~~~~~~~~~~~~~~~~
Joint training shares ONE timestep ``t`` across both modalities and sums::

    loss = video_fm_loss + audio_weight * masked_audio_fm_loss

For the audio term, each batch item carries an ``audio_mask`` scalar in
``{0, 1}``:

- A video clip WITH audio          → mask = 1 (its audio loss flows).
- A video clip WITHOUT audio       → mask = 0 (audio_latents = zeros; the
  audio loss term is forced to zero — never train on silence-as-target).
- An image (F = 1)                 → mask = 0 (no temporal audio at all).

:func:`masked_audio_loss` computes the per-element flow-match MSE, reduces it
per item, multiplies by the per-item mask, and divides by the number of
*present* (mask=1) items — so absent-audio and image items contribute exactly
zero to both the numerator and the denominator.  When the whole batch lacks
audio the term is a hard zero (no division by zero).
"""

from __future__ import annotations

import torch
from torch import Tensor


def extract_audio_latents(
    audio_vae,
    waveform: Tensor,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Encode a waveform to audio latents via the LTX-2 audio VAE.

    Pure wrapper around ``audio_vae.encode`` that tolerates both the diffusers
    ``AutoencoderKLOutput`` (``.latent_dist.mode()``) and a raw-tensor return
    (fakes / future VAEs).  No caching here — the caller persists the result
    alongside the video latent.

    Args:
        audio_vae: An object exposing ``.encode(waveform)``.
        waveform: ``[B, C, N]`` (or ``[B, N]``) float waveform in ``[-1, 1]``.
        device: Optional device to move the input onto first.

    Returns:
        Audio latent tensor as returned by the VAE (channel-first).
    """
    if device is not None:
        waveform = waveform.to(device)
    with torch.no_grad():
        encoded = audio_vae.encode(waveform)
    if isinstance(encoded, Tensor):
        return encoded
    dist = getattr(encoded, "latent_dist", None)
    if dist is not None:
        return dist.mode()
    # Some VAEs return a tuple / object with ``.sample`` — fall back gracefully.
    sample = getattr(encoded, "sample", None)
    return sample if isinstance(sample, Tensor) else encoded


def masked_audio_loss(
    pred: Tensor,
    target: Tensor,
    audio_mask: Tensor,
) -> Tensor:
    """Mask-weighted mean flow-match MSE for the audio stream.

    Computes the per-item MSE (mean over every non-batch dim), multiplies by the
    per-item ``audio_mask`` (``{0, 1}``), and averages over the number of
    PRESENT items only.  Absent-audio and image items (mask=0) contribute zero
    to numerator and denominator, so they neither dilute nor inflate the loss.

    Args:
        pred: Audio velocity prediction ``[B, ...]``.
        target: Audio velocity target ``[B, ...]`` (``noise - latents``).
        audio_mask: ``[B]`` mask in ``{0, 1}``.

    Returns:
        Scalar tensor.  Exactly ``0.0`` (no grad path) when no item has audio.
    """
    pred = pred.float()
    target = target.float()
    mask = audio_mask.float().reshape(-1)

    # Per-item MSE: mean over all dims except batch.
    per_elem = (pred - target) ** 2
    reduce_dims = list(range(1, per_elem.ndim))
    per_item = per_elem.mean(dim=reduce_dims) if reduce_dims else per_elem

    weighted = per_item * mask
    denom = mask.sum()
    if float(denom) == 0.0:
        # Hard zero — keep it a tensor (and on-device) but with no division.
        return weighted.sum() * 0.0
    return weighted.sum() / denom
