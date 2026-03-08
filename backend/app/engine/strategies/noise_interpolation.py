"""Noise interpolation strategies for diffusion training.

Provides a factory for computing `noisy_latents = interpolate(latents, noise, t)`
using different schedules.  All current model families use ``linear``
(rectified-flow / flow-matching), but ``ddpm`` and ``cosine`` are included
for forward-compatibility with SDXL/SD-style models and research schedules.

Usage::

    from app.engine.strategies.noise_interpolation import NoiseInterpolation

    interp = NoiseInterpolation("linear")       # or from config
    noisy = interp.add_noise(latents, noise, timesteps)
"""

import math
from typing import Any

import structlog
import torch

logger = structlog.get_logger(__name__)


class NoiseInterpolation:
    """Configurable noise–signal interpolation for the forward diffusion process.

    Supported modes:
        ``linear``
            Rectified-flow / flow-matching: ``(1-t)*x + t*noise``.
            Used by Flux1, Flux2, Qwen-Image, ZImage, SD3.

        ``ddpm``
            Classic DDPM scheduler-based: ``sqrt(ᾱ)*x + sqrt(1-ᾱ)*noise``.
            Requires a diffusers-style scheduler with ``alphas_cumprod``.
            Used by SDXL, SD1.5.

        ``cosine``
            Cosine-weighted blend: ``cos(πt/2)*x + sin(πt/2)*noise``
            where ``t ∈ [0, 1]``.  Slower noise onset than linear,
            more time in mid-SNR regime.
    """

    SUPPORTED_MODES = ("linear", "ddpm", "cosine")

    def __init__(
        self,
        mode: str = "linear",
        scheduler: Any | None = None,
    ) -> None:
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unknown noise interpolation mode '{mode}'. "
                f"Supported: {self.SUPPORTED_MODES}"
            )
        if mode == "ddpm" and scheduler is None:
            raise ValueError(
                "DDPM noise interpolation requires a scheduler with "
                "`alphas_cumprod`.  Pass the scheduler instance."
            )
        self.mode = mode
        self.scheduler = scheduler
        logger.info(
            "noise_interpolation_initialized",
            mode=mode,
            has_scheduler=scheduler is not None,
        )

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Blend signal and noise at the given timesteps.

        Args:
            latents: Clean latent signal ``[B, ...]``.
            noise: Noise tensor, same shape as *latents*.
            timesteps: Per-sample timesteps.
                       ``[0, 1000]`` for linear/cosine (scaled to ``[0, 1]``).
                       Discrete indices for ddpm.

        Returns:
            Noisy latents ``[B, ...]``.
        """
        if self.mode == "linear":
            return self._linear(latents, noise, timesteps)
        elif self.mode == "ddpm":
            return self._ddpm(latents, noise, timesteps)
        elif self.mode == "cosine":
            return self._cosine(latents, noise, timesteps)
        # Unreachable because __init__ validates, but satisfy type checkers
        raise ValueError(f"Unsupported mode: {self.mode}")  # pragma: no cover

    # ── Interpolation implementations ────────────────────────────────────

    @staticmethod
    def _linear(
        latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Rectified-flow: ``(1-t)*x + t*ε``."""
        shape = (-1,) + (1,) * (latents.ndim - 1)
        t_01 = (timesteps / 1000.0).view(*shape).to(
            latents.device, dtype=latents.dtype
        )
        return (1.0 - t_01) * latents + t_01 * noise

    def _ddpm(
        self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """DDPM / SDXL: ``√ᾱ·x + √(1-ᾱ)·ε``."""
        alphas_cumprod = self.scheduler.alphas_cumprod.to(
            device=latents.device, dtype=latents.dtype
        )
        alpha_t = alphas_cumprod[timesteps.long()]
        shape = (-1,) + (1,) * (latents.ndim - 1)
        sqrt_alpha = alpha_t.sqrt().view(*shape)
        sqrt_one_minus = (1.0 - alpha_t).sqrt().view(*shape)
        return sqrt_alpha * latents + sqrt_one_minus * noise

    @staticmethod
    def _cosine(
        latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Cosine blend: ``cos(πt/2)·x + sin(πt/2)·ε``."""
        shape = (-1,) + (1,) * (latents.ndim - 1)
        t_01 = (timesteps / 1000.0).view(*shape).to(
            latents.device, dtype=latents.dtype
        )
        angle = t_01 * (math.pi / 2.0)
        return torch.cos(angle) * latents + torch.sin(angle) * noise
