"""Shared Timestep Sampling — all modes available to every model family.

Follows the ``OptimizerFactory`` pattern: static factory with a single
``sample()`` entry-point.  Each model family delegates to this instead of
duplicating ~50 lines of mode-selection logic.

Modes
-----
uniform        – U(0,1)
logit_normal   – Sigmoid(N(mu, sigma))  (configurable ``logit_normal_mu/sigma``)
sigmoid        – Sigmoid(N(0,1))
cosmap         – 1 - cos(U * π/2)
mode           – Mode-weighted with configurable ``mode_scale``
flux_shift     – Resolution-dependent shifted logit-normal
radc           – Resolution-Aware Dynamic Curriculum (step-aware Gaussian
                 that shifts from high-noise to low-noise over training,
                 with a resolution × progress cross-function)
"""

from __future__ import annotations

import math
from typing import Any

import structlog
import torch

logger = structlog.get_logger(__name__)

# ── RADC PDF builder ──────────────────────────────────────────────────────


def _make_radc_pdf(center: float, width: float) -> torch.Tensor:
    """Build a Gaussian PDF over 1000 bins with configurable center/width.

    Args:
        center: Peak position in ``[0, 1]``.  0 = low noise (clean),
                1 = high noise (pure noise).
        width:  Gaussian width factor.  Maps to ``sigma = width * 500``
                index-space units.  0.1 = very focused, 1.0 = nearly uniform.

    Returns:
        Mean-normalized weight tensor ``[1000]`` on CPU (float32).
    """
    ts = torch.arange(1, 1001, dtype=torch.float32)
    center_idx = center * 1000.0
    sigma = max(width * 500.0, 1.0)
    pdf = torch.exp(-0.5 * ((ts - center_idx) / sigma) ** 2)
    pdf = pdf / pdf.sum()
    return pdf / pdf.mean()  # mean ≈ 1.0


def _radc_center(
    config: dict[str, Any],
    progress: float,
    latents: torch.Tensor | None,
) -> float:
    """Compute the dynamic RADC center for this step.

    Interpolates linearly from ``radc_start`` (high-noise focus at step 0)
    to ``radc_end`` (low-noise focus at final step), then applies a
    **resolution × progress cross-function** that makes high-resolution
    images shift further toward low-noise detail refinement at the end
    of training, while low-resolution images stay at higher noise.

    Cross-function:
        ``cross = res_influence * progress * (1 - 2 * res_norm)``

        * ``progress ≈ 0`` → cross ≈ 0 (all resolutions equal)
        * ``progress ≈ 1, high-res`` → negative shift → more detail focus
        * ``progress ≈ 1, low-res``  → positive shift → structure emphasis
    """
    start = float(config.get("radc_start", 0.8))
    end = float(config.get("radc_end", 0.2))
    res_influence = float(config.get("radc_res_influence", 0.15))

    base_center = start + (end - start) * progress

    # Resolution cross-function (only when latents are available)
    if res_influence > 0 and latents is not None and latents.ndim >= 4:
        h, w = latents.shape[2], latents.shape[3]
        pf = int(config.get("flux_shift_patchify_factor", 1))
        seq_len = (h // pf) * (w // pf)
        # Normalize seq_len to [0, 1] range (256..4096 → 0..1)
        res_norm = max(0.0, min(1.0, (seq_len - 256) / (4096 - 256)))
        cross = res_influence * progress * (1.0 - 2.0 * res_norm)
        base_center = base_center + cross

    return max(0.0, min(1.0, base_center))


# ── Public API ────────────────────────────────────────────────────────────


class TimestepSampler:
    """Static factory for timestep sampling — one implementation for all
    model families.

    Usage from a trainer::

        from app.engine.strategies.timestep_sampling import TimestepSampler

        def sample_timesteps(self, bs, latents=None):
            mode = self.config.get("timestep_sampling", "logit_normal")
            progress = self.global_step / self.max_train_steps
            return TimestepSampler.sample_scaled(
                mode, bs, self.device, self.config,
                latents=latents, progress=progress,
            )
    """

    SUPPORTED_MODES = [
        "uniform",
        "logit_normal",
        "sigmoid",
        "cosmap",
        "mode",
        "flux_shift",
        "radc",
    ]

    # ── Core sampler ──────────────────────────────────────────────────

    @staticmethod
    def sample(
        mode: str,
        bs: int,
        device: torch.device,
        config: dict[str, Any],
        *,
        latents: torch.Tensor | None = None,
        progress: float = 0.0,
    ) -> torch.Tensor:
        """Sample raw timesteps in ``[0, 1]``.

        Args:
            mode: One of :pyattr:`SUPPORTED_MODES`.
            bs: Batch size.
            device: Target device.
            config: Training config dict (for mode-specific params).
            latents: Optional latents for resolution-dependent modes.
            progress: Training progress in ``[0, 1]`` (step / total_steps).
                      Used by ``radc`` mode for curriculum shifting.

        Returns:
            Timestep tensor ``[bs]`` in ``[0, 1]``.
        """
        if mode == "uniform":
            return torch.rand((bs,), device=device)

        if mode == "logit_normal":
            # Accept both naming conventions for backward compat
            mu = float(
                config.get("logit_normal_mu",
                           config.get("logit_normal_mean", 0.0))
            )
            sigma = float(
                config.get("logit_normal_sigma",
                           config.get("logit_normal_std", 1.0))
            )
            return torch.sigmoid(
                torch.randn((bs,), device=device) * sigma + mu
            )

        if mode == "sigmoid":
            return torch.sigmoid(torch.randn((bs,), device=device))

        if mode == "cosmap":
            u = torch.rand((bs,), device=device)
            return 1.0 - torch.cos(u * math.pi / 2.0)

        if mode == "mode":
            scale = float(config.get("mode_scale", 1.5))
            u = torch.rand((bs,), device=device)
            return 1.0 - u / (scale + (1.0 - scale) * u)

        if mode == "flux_shift":
            base_shift = float(config.get("flux_shift_base", 0.5))
            max_shift = float(config.get("flux_shift_max", 1.16))
            if latents is not None:
                h, w = latents.shape[2], latents.shape[3]
                # Flux2 patchifies latents (2× down per spatial dim) before
                # the transformer, so the actual image sequence length the
                # model sees is (h/p)*(w/p).  Without this correction the mu
                # is far too high (3.27 instead of 1.16 for 1024px images),
                # heavily biasing training toward extreme-noise timesteps.
                pf = int(config.get("flux_shift_patchify_factor", 1))
                seq_len = (h // pf) * (w // pf)
                m = (max_shift - base_shift) / (4096 - 256)
                b = base_shift - m * 256
                mu = seq_len * m + b
            else:
                mu = (base_shift + max_shift) / 2.0
            return torch.sigmoid(torch.randn((bs,), device=device) + mu)

        if mode == "radc":
            center = _radc_center(config, progress, latents)
            width = float(config.get("radc_width", 0.5))
            pdf = _make_radc_pdf(center, width)
            u = torch.rand((bs,), device=device)
            cdf = torch.cumsum(pdf / pdf.sum(), dim=0).to(device)
            indices = torch.searchsorted(cdf, u)
            return (indices.float() + 1) / 1000.0

        # Unknown → fallback with warning
        logger.warning("unknown_timestep_mode", mode=mode, fallback="uniform")
        return torch.rand((bs,), device=device)

    # ── Convenience wrapper ───────────────────────────────────────────

    @staticmethod
    def sample_scaled(
        mode: str,
        bs: int,
        device: torch.device,
        config: dict[str, Any],
        *,
        scale: float = 1000.0,
        latents: torch.Tensor | None = None,
        progress: float = 0.0,
    ) -> torch.Tensor:
        """Sample timesteps and scale to ``[0, scale]``.

        Convenience wrapper around :meth:`sample` for the common
        ``t * 1000`` pattern used by most model families.

        Returns:
            Timestep tensor ``[bs]`` clamped to ``[0, scale]``.
        """
        t = TimestepSampler.sample(
            mode, bs, device, config, latents=latents, progress=progress,
        )
        return (t * scale).clamp(0, scale)
