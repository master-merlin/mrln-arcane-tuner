"""WAN 2.1 I2V conditioning: first-frame latent + 4-channel temporal mask.

WAN 2.1 image-to-video conditions the transformer on the first frame. The
conditioning input fed to the transformer concatenates, along the channel axis:

    [ noisy(16) , mask(4) , cond(16) ]  →  36 in-channels

where:

- ``noisy(16)``  : the noised video latent (the diffusion variable).
- ``mask(4)``    : a temporal mask with 1.0 on the FIRST latent frame and 0.0
                   on the rest, broadcast to 4 channels (WAN's mask is encoded
                   as 4 channels because the VAE temporal factor is 4).
- ``cond(16)``   : the first input frame encoded through the VAE (the rest of
                   the temporal window zero-padded), giving a 16-channel latent
                   whose only non-zero temporal slot is the first latent frame.

CRITICAL: the noise + velocity target are computed ONLY over the 16 noise
channels. The ``mask`` + ``cond`` channels are conditioning, not diffusion
targets — so the concatenation must happen INSIDE ``forward_pass`` (from a
batch "extra"), never folded into the latent tensor the trainer noises.

All tensors are 5D ``[B, C, F, H, W]``. These builders are weight-free and
unit-test with tiny fake tensors.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Channel layout constants for WAN 2.1 I2V.
NOISE_CHANNELS = 16
MASK_CHANNELS = 4
COND_CHANNELS = 16
I2V_IN_CHANNELS = NOISE_CHANNELS + MASK_CHANNELS + COND_CHANNELS  # 36


def build_temporal_mask(
    latent_frames: int,
    batch: int,
    height: int,
    width: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build the 4-channel temporal mask ``[B, 4, F, H, W]``.

    The mask is 1.0 on the first latent frame and 0.0 elsewhere, broadcast
    across all 4 mask channels and the full spatial grid.
    """
    mask = torch.zeros(
        (batch, MASK_CHANNELS, latent_frames, height, width),
        device=device,
        dtype=dtype,
    )
    mask[:, :, 0, :, :] = 1.0
    return mask


def build_first_frame_cond(
    first_frame_latent: Tensor,
    latent_frames: int,
) -> Tensor:
    """Zero-pad a first-frame latent to ``F`` latent frames.

    Args:
        first_frame_latent: ``[B, 16, 1, H, W]`` (the VAE-encoded first frame),
            or ``[B, 16, H, W]`` which is treated as a single latent frame.
        latent_frames: Target latent-frame count ``F``.

    Returns:
        ``[B, 16, F, H, W]`` with the first-frame latent in temporal slot 0
        and zeros elsewhere.
    """
    if first_frame_latent.ndim == 4:
        first_frame_latent = first_frame_latent.unsqueeze(2)  # [B, C, 1, H, W]
    if first_frame_latent.shape[2] != 1:
        # Keep only the first latent frame if a multi-frame tensor is passed.
        first_frame_latent = first_frame_latent[:, :, :1, :, :]

    b, c, _, h, w = first_frame_latent.shape
    cond = torch.zeros(
        (b, c, latent_frames, h, w),
        device=first_frame_latent.device,
        dtype=first_frame_latent.dtype,
    )
    cond[:, :, 0, :, :] = first_frame_latent[:, :, 0, :, :]
    return cond


def build_i2v_conditioning(
    noisy_latents: Tensor,
    first_frame_latent: Tensor,
) -> Tensor:
    """Concatenate ``[noisy(16), mask(4), cond(16)]`` → ``[B, 36, F, H, W]``.

    Args:
        noisy_latents: ``[B, 16, F, H, W]`` — the noised video latent (the
            diffusion variable). Its temporal/spatial shape defines the mask
            and cond shapes.
        first_frame_latent: ``[B, 16, 1, H, W]`` or ``[B, 16, H, W]`` — the
            VAE-encoded first frame.

    Returns:
        ``[B, 36, F, H, W]`` transformer input. The first 16 channels are
        exactly ``noisy_latents`` (so the caller can recover the diffusion
        variable / compute the velocity target over those channels alone).
    """
    if noisy_latents.ndim != 5:
        raise ValueError(
            f"noisy_latents must be 5D [B, C, F, H, W], got {tuple(noisy_latents.shape)}"
        )
    if noisy_latents.shape[1] != NOISE_CHANNELS:
        raise ValueError(
            f"noisy_latents must have {NOISE_CHANNELS} channels, "
            f"got {noisy_latents.shape[1]}"
        )

    b, _, f, h, w = noisy_latents.shape
    mask = build_temporal_mask(
        f, b, h, w, device=noisy_latents.device, dtype=noisy_latents.dtype
    )
    cond = build_first_frame_cond(first_frame_latent, f)

    return torch.cat([noisy_latents, mask, cond], dim=1)


def build_still_t2v_input(noisy_latents: Tensor) -> Tensor:
    """36-ch input for an F=1 still on an i2v run — NO conditioning leak.

    A single still has no frame to predict beyond a conditioning frame, so it
    must train as t2v: the model denoises the one frame from full noise with no
    reference. The i2v transformer's ``patch_embedding`` is a 36-in-channel
    conv, so we still concat to 36 channels — but the ``mask(4)`` AND
    ``cond(16)`` channels are ZEROED (``mask=0`` ⇒ "denoise this frame",
    ``cond=0`` ⇒ "no reference frame"). This is what keeps the still's own clean
    latent from being handed back as the answer, which — because the flow-match
    target ``noise - latents`` is recoverable from ``noisy`` + ``cond`` — would
    otherwise be a degenerate, zero-information training step.

    Parity with ltx2/k5's ``_i2v_conditioning_engaged`` F>1 gate; WAN differs
    only in that the 36-ch architecture forces this zero-pad instead of a
    plain 16-channel t2v tensor.

    Args:
        noisy_latents: ``[B, 16, 1, H, W]`` — the noised single-frame latent.

    Returns:
        ``[B, 36, 1, H, W]`` = ``[noisy(16), zeros(4), zeros(16)]``. The first
        16 channels are exactly ``noisy_latents``.
    """
    if noisy_latents.ndim != 5:
        raise ValueError(
            f"noisy_latents must be 5D [B, C, F, H, W], got {tuple(noisy_latents.shape)}"
        )
    if noisy_latents.shape[1] != NOISE_CHANNELS:
        raise ValueError(
            f"noisy_latents must have {NOISE_CHANNELS} channels, "
            f"got {noisy_latents.shape[1]}"
        )

    b, _, f, h, w = noisy_latents.shape
    pad = torch.zeros(
        (b, MASK_CHANNELS + COND_CHANNELS, f, h, w),
        device=noisy_latents.device,
        dtype=noisy_latents.dtype,
    )
    return torch.cat([noisy_latents, pad], dim=1)
