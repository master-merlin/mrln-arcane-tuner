"""The MiniMax-H3 flow-match convention, shared by the video and image models.

H3 runs its clock INVERTED against the house default: ``t = 1`` is clean,
``t = 1 - sigma`` in ``[0, 1]``, UNSCALED (no x1000; the transformer's
``time_proj`` consumes ``[0, 1]`` directly). The closed oracle is diffusers
0.40.0 ``scheduling_minimax_h3.py``: ``:170-171`` ``t = 1 - sigma``, ``:225``
``x_t = t*x0 + (1-t)*noise``, ``:273`` ``x0_hat = x_t + sigma*v``. Hence

    x0 = x_t + sigma*v   =>   v = x0 - noise   (unique; no scale, no sign freedom)

ai-toolkit (``t_v = 1.0 - sigma_v``, ``return -noise_pred``) and
diffusion-pipe (``t_v = 1.0 - sigma_v``, ``-video_out``) train on the same
contract. The house default target (``noise - latents``) is the OPPOSITE sign
on these models.
"""

from __future__ import annotations

import torch


def sigma_to_t(sigma: torch.Tensor) -> torch.Tensor:
    """H3's clock: ``t = 1 - sigma``, ``t = 1`` clean."""
    return 1.0 - sigma


def t_to_sigma(t: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`sigma_to_t`: the noise weight of ``x_t``."""
    return 1.0 - t


def add_noise(latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """``x_t = t*x0 + sigma*noise`` with ``sigma = 1 - t``, one ``t`` per item
    broadcast over the trailing dims; both endpoints are exact."""
    t = timesteps.to(device=latents.device, dtype=latents.dtype)
    while t.ndim < latents.ndim:
        t = t.unsqueeze(-1)
    return t * latents + t_to_sigma(t) * noise


def velocity_target(latents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """The data-ward velocity ``v = x0 - noise``: the unique target the
    reference scheduler's ``step`` inverts."""
    return latents - noise
