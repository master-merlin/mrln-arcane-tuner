"""FLUX.2 latent packing utilities.

Converts between spatial latents [B, C, H, W] and the packed sequence
format [B, L, D] expected by diffusers ``Flux2Transformer2DModel``.

The transformer expects 2×2 spatial patches packed into the channel
dimension: [B, C, H, W] → [B, (H/2)*(W/2), C*4].  Position IDs use
4 columns ``(T, H, W, L)`` for the 4-axis RoPE embedding.

Latent normalization uses the VAE's pretrained ``BatchNorm2d``
running statistics (``running_mean`` / ``running_var``) instead of a
scalar ``scaling_factor``.  Normalization must be applied in the
patchified space ``[B, C*4, H/2, W/2]``, i.e. *after* pixel-unshuffle
but *before* sequence packing.
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import Tensor


# ---------------------------------------------------------------------------
# Patchify / Unpatchify (pixel-unshuffle / pixel-shuffle)
# ---------------------------------------------------------------------------

def patchify_latents(latents: Tensor) -> Tensor:
    """Pixel-unshuffle: ``[B, C, H, W]`` → ``[B, C*4, H/2, W/2]``.

    Groups every 2×2 spatial block into the channel dimension.
    Matches ``Flux2Pipeline._patchify_latents`` in diffusers.
    """
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(B, C * 4, H // 2, W // 2)
    return latents


def unpatchify_latents(latents: Tensor) -> Tensor:
    """Pixel-shuffle: ``[B, C*4, H/2, W/2]`` → ``[B, C, H, W]``.

    Reverses :func:`patchify_latents`.
    Matches ``Flux2Pipeline._unpatchify_latents`` in diffusers.
    """
    B, C4, H_half, W_half = latents.shape
    C = C4 // 4
    latents = latents.reshape(B, C, 2, 2, H_half, W_half)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(B, C, H_half * 2, W_half * 2)
    return latents


# ---------------------------------------------------------------------------
# BatchNorm normalize / denormalize
# ---------------------------------------------------------------------------

def bn_normalize(latents: Tensor, vae: torch.nn.Module) -> Tensor:
    """Normalize patchified latents using VAE BatchNorm running statistics.

    Applies ``(x - mean) / std`` where mean and std come from
    ``vae.bn.running_mean`` / ``vae.bn.running_var``.

    Must be called on *patchified* latents ``[B, C*4, H/2, W/2]``.
    Matches ``Flux2Pipeline._encode_vae_image`` normalization.

    Args:
        latents: Patchified latents ``[B, C*4, H/2, W/2]``.
        vae: ``AutoencoderKLFlux2`` instance with ``bn`` attribute.

    Returns:
        Normalized latents, same shape.
    """
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(
        device=latents.device, dtype=latents.dtype
    )
    bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1).to(
            device=latents.device, dtype=latents.dtype
        )
        + vae.config.batch_norm_eps
    )
    return (latents - bn_mean) / bn_std


def bn_denormalize(latents: Tensor, vae: torch.nn.Module) -> Tensor:
    """Denormalize patchified latents — inverse of :func:`bn_normalize`.

    Applies ``x * std + mean``.
    Matches ``Flux2Pipeline.__call__`` decode path normalization.

    Args:
        latents: Normalized patchified latents ``[B, C*4, H/2, W/2]``.
        vae: ``AutoencoderKLFlux2`` instance with ``bn`` attribute.

    Returns:
        Denormalized latents, same shape.
    """
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(
        device=latents.device, dtype=latents.dtype
    )
    bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1).to(
            device=latents.device, dtype=latents.dtype
        )
        + vae.config.batch_norm_eps
    )
    return latents * bn_std + bn_mean


# ---------------------------------------------------------------------------
# Pack / Unpack (spatial ↔ sequence)
# ---------------------------------------------------------------------------

def _pack_spatial(latents: Tensor) -> Tensor:
    """Flatten spatial dims to sequence: ``[B, C, H, W]`` → ``[B, H*W, C]``."""
    B, C, H, W = latents.shape
    return latents.reshape(B, C, H * W).permute(0, 2, 1)


def _make_img_ids(h: int, w: int, device: torch.device) -> Tensor:
    """Create 4-column RoPE position IDs ``[L, 4]`` for a grid ``h × w``.

    Columns ``(T, H, W, L)``; T=0 and L=0 for image tokens.
    """
    t = torch.arange(1, device=device)  # [0]
    hr = torch.arange(h, device=device)
    wr = torch.arange(w, device=device)
    l = torch.arange(1, device=device)  # [0]  # noqa: E741
    return torch.cartesian_prod(t, hr, wr, l)  # [h*w, 4]


def pack_latents(x: Tensor) -> tuple[Tensor, Tensor]:
    """Pack spatial latents via 2×2 patching for Flux2Transformer2DModel.

    Groups every 2×2 spatial block into the channel dimension so the
    transformer receives tokens of width ``C * 4``.

    Args:
        x: Image latents ``[B, C, H, W]`` (H, W must be even).

    Returns:
        Tuple of:
        - packed tokens ``[B, (H/2)*(W/2), C*4]``
        - position IDs ``[L, 4]`` with columns ``(T, H, W, L)``
    """
    B, C, H, W = x.shape

    # 2×2 patch packing: [B, C, H, W] → [B, (H/2)*(W/2), C*4]
    packed = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    # Position IDs over the patched grid (H/2 × W/2)
    img_ids = _make_img_ids(H // 2, W // 2, x.device)

    return packed, img_ids


def unpack_latents(
    packed: Tensor, height: int, width: int
) -> Tensor:
    """Unpack patched sequence format back to spatial latents.

    Args:
        packed: Packed tokens ``[B, (H/2)*(W/2), C*4]``.
        height: Spatial height of latents (H, before patching).
        width: Spatial width of latents (W, before patching).

    Returns:
        Spatial latents ``[B, C, H, W]``.
    """
    return rearrange(
        packed,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=height // 2,
        w=width // 2,
        ph=2,
        pw=2,
    )
