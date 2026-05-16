"""ERNIE-Image latent packing utilities.

The transformer operates in **patchified** space — 2x2 spatial patches
from the VAE latent are packed into the channel dimension before being
fed to the model:

    VAE latent [B, 32, H, W] --(2x2 patchify)--> [B, 128, H/2, W/2]

ERNIE reuses the FLUX.2 VAE (``AutoencoderKLFlux2``), which carries
``running_mean`` / ``running_var`` BatchNorm statistics that must be
applied to the patchified latents before noise is added.  This mirrors
the official ``ErnieImagePipeline`` decode path which applies
``latents * bn_std + bn_mean`` immediately before ``_unpatchify_latents``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def patchify_latents(latents: Tensor) -> Tensor:
    """Pixel-unshuffle: ``[B, C, H, W]`` → ``[B, C*4, H/2, W/2]``.

    Matches ``ErnieImagePipeline._patchify_latents``.
    """
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(b, c * 4, h // 2, w // 2)


def unpatchify_latents(latents: Tensor) -> Tensor:
    """Pixel-shuffle: ``[B, C*4, H/2, W/2]`` → ``[B, C, H, W]``.

    Matches ``ErnieImagePipeline._unpatchify_latents``.
    """
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c // 4, 2, 2, h, w)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(b, c // 4, h * 2, w * 2)


def bn_normalize(latents: Tensor, vae: torch.nn.Module) -> Tensor:
    """Normalize patchified latents with the VAE's BatchNorm running stats.

    Inverse of the ``latents * bn_std + bn_mean`` denormalize done in
    ``ErnieImagePipeline.__call__`` right before ``_unpatchify_latents``.
    Must be applied in patchified space ``[B, C*4, H/2, W/2]``.
    """
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(
        device=latents.device, dtype=latents.dtype,
    )
    eps = getattr(vae.config, "batch_norm_eps", 1e-5)
    bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1).to(
            device=latents.device, dtype=latents.dtype,
        )
        + eps,
    )
    return (latents - bn_mean) / bn_std


def bn_denormalize(latents: Tensor, vae: torch.nn.Module) -> Tensor:
    """Denormalize patchified latents — inverse of :func:`bn_normalize`."""
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(
        device=latents.device, dtype=latents.dtype,
    )
    eps = getattr(vae.config, "batch_norm_eps", 1e-5)
    bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1).to(
            device=latents.device, dtype=latents.dtype,
        )
        + eps,
    )
    return latents * bn_std + bn_mean
