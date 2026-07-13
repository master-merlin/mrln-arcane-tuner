"""Wan-VAE latent normalization + frame-rule helpers.

The Wan-VAE (``AutoencoderKLWan``) does NOT use a scalar ``scaling_factor``.
Instead it ships per-channel ``latents_mean`` / ``latents_std`` (16 channels)
in its config and normalizes latents as::

    z_norm = (z - mean) / std          # encode side
    z      = z_norm * std + mean       # decode side

It also temporally compresses video by 4× and spatially by 8×, so the input
frame count ``F`` must satisfy the ``4n+1`` rule (one latent frame per group of
four input frames, plus the leading frame): ``latent_f = (F - 1) / 4 + 1``.

These helpers are intentionally weight-free and operate on tensors + a config
object (or plain mean/std sequences) so they unit-test without loading a VAE.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

# Wan-VAE compression factors (constant across WAN 2.1 / 2.2 -- the A14B/
# 1.3B/14B checkpoints all share this 8x-spatial / 16-channel VAE).
#
# NOT valid for the wan22_ti2v_5b family: its all-new higher-compression VAE
# is spatial 16 / z_dim 48 (see families/wan22_ti2v_5b/definitions/
# wan22_ti2v_5b.yaml's provenance header). That family does NOT import these
# constants -- its bucket math reads `video.vae_spatial` from the definition
# via the video contract, and its latent normalization reads the real VAE's
# own `config.latents_mean`/`latents_std`. Do not import these constants
# generically for wan22_ti2v_5b; add a family-aware lookup instead if a
# future caller needs one.
WAN_VAE_SPATIAL = 8
WAN_VAE_TEMPORAL = 4
WAN_VAE_Z_DIM = 16


def wan_latent_stats(
    config_or_mean: Any,
    std: Sequence[float] | None = None,
) -> tuple[list[float], list[float]]:
    """Resolve ``(latents_mean, latents_std)`` from a VAE config or raw lists.

    Accepts either an ``AutoencoderKLWan``-style object exposing
    ``config.latents_mean`` / ``config.latents_std`` (or those attrs directly),
    or two explicit sequences. Returns plain Python lists.
    """
    if std is not None:
        return list(config_or_mean), list(std)

    cfg = getattr(config_or_mean, "config", config_or_mean)
    mean = getattr(cfg, "latents_mean", None)
    sd = getattr(cfg, "latents_std", None)
    if mean is None or sd is None:
        raise ValueError(
            "Wan-VAE latent stats unavailable: config has no "
            "latents_mean / latents_std."
        )
    return list(mean), list(sd)


def _reshape_stats(values: Sequence[float], latents: Tensor) -> Tensor:
    """Reshape a per-channel stat list to broadcast against ``latents``.

    Channel axis is dim 1 for both 4D ``[B, C, H, W]`` and 5D
    ``[B, C, F, H, W]`` latents.
    """
    t = torch.as_tensor(values, device=latents.device, dtype=latents.dtype)
    # [C] → [1, C, 1, ...] matching latents.ndim
    return t.view(1, -1, *([1] * (latents.ndim - 2)))


def normalize_wan_latents(
    latents: Tensor,
    config_or_mean: Any,
    std: Sequence[float] | None = None,
) -> Tensor:
    """Apply ``(z - mean) / std`` per-channel normalization.

    Use on the VAE *encode* output before it enters the diffusion process.
    """
    mean, sd = wan_latent_stats(config_or_mean, std)
    mean_t = _reshape_stats(mean, latents)
    std_t = _reshape_stats(sd, latents)
    return (latents - mean_t) / std_t


def denormalize_wan_latents(
    latents: Tensor,
    config_or_mean: Any,
    std: Sequence[float] | None = None,
) -> Tensor:
    """Invert normalization: ``z * std + mean`` per-channel.

    Use before the VAE *decode* call at sampling time.
    """
    mean, sd = wan_latent_stats(config_or_mean, std)
    mean_t = _reshape_stats(mean, latents)
    std_t = _reshape_stats(sd, latents)
    return latents * std_t + mean_t


def is_valid_frame_count(num_frames: int) -> bool:
    """Return True iff ``num_frames`` obeys the Wan ``4n+1`` rule (n >= 0)."""
    return num_frames >= 1 and (num_frames - 1) % WAN_VAE_TEMPORAL == 0


def assert_frame_rule(num_frames: int) -> None:
    """Raise ``ValueError`` if ``num_frames`` violates the ``4n+1`` rule.

    The Wan-VAE compresses time by 4×; a non-conforming frame count produces a
    fractional latent-frame request and a shape mismatch at encode time.
    """
    if not is_valid_frame_count(num_frames):
        raise ValueError(
            f"WAN frame count {num_frames} violates the 4n+1 rule "
            f"(temporal compression {WAN_VAE_TEMPORAL}×). Valid counts: "
            f"1, 5, 9, 13, 17, 21, 25, ... (F = 4n + 1)."
        )


def latent_frames_4x(num_frames: int) -> int:
    """Latent frame count for ``num_frames`` input frames: ``(F - 1)/4 + 1``."""
    assert_frame_rule(num_frames)
    return (num_frames - 1) // WAN_VAE_TEMPORAL + 1
