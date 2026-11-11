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
model_shift    – Inference-matched flow-match shift in logit space.
                 LTX (dynamic): mu interpolates base_shift→max_shift with seq-len.
                 WAN (fixed):   mu = ln(flow_shift) (seq-len independent).
                 else (fallback): flux-style mid shift.
fixed_sigmas   – Uniform over the configured ``fixed_sigmas`` list (each value in
                 (0, 1]): a model distilled to a fixed schedule trains on its levels.

Every mode takes an optional local ``generator``; without one it draws from the
global RNG exactly as before.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Callable

import structlog
import torch

logger = structlog.get_logger(__name__)


# ── Shared seq-len helper ─────────────────────────────────────────────────


def _patchified_seq_len(latents: torch.Tensor | None, patchify_factor: int = 1) -> int | None:
    """Token count the transformer sees: F * (H/p) * (W/p).

    4D image latent [B,C,H,W] → F=1. 5D video latent [B,C,F,H,W] → real F.
    H,W are ALWAYS the last two dims (the prior code wrongly read shape[2],
    shape[3], which for 5D video are F,H). Returns None for unusable input.
    """
    if latents is None or latents.ndim not in (4, 5):
        return None
    if latents.ndim == 5:
        f, h, w = int(latents.shape[2]), int(latents.shape[3]), int(latents.shape[4])
    else:
        f, h, w = 1, int(latents.shape[2]), int(latents.shape[3])
    p = max(int(patchify_factor), 1)
    return f * (h // p) * (w // p)


# ── RADC PDF builder ──────────────────────────────────────────────────────


@lru_cache(maxsize=256)
def _radc_cdf(center_q: int, width_q: int) -> torch.Tensor:
    """Cached CPU CDF for a quantized ``(center, width)`` pair.

    ``center`` moves continuously with training progress, so the raw pair is a
    poor cache key; quantizing to 1e-3 collapses a whole run into a few hundred
    distinct curves while shifting the sampled distribution by less than one of
    the 1000 bins. Without this, every step rebuilt a 1000-bin Gaussian,
    normalized it twice, and re-derived the cumulative sum.
    """
    pdf = _make_radc_pdf(center_q / 1000.0, width_q / 1000.0)
    return torch.cumsum(pdf / pdf.sum(), dim=0)


@lru_cache(maxsize=256)
def _radc_pdf(center_q: int, width_q: int) -> torch.Tensor:
    """Cached CPU weight curve for the quantized pair :func:`_radc_cdf` draws from.

    The returned tensor is shared: index it, never modify it in place.
    """
    return _make_radc_pdf(center_q / 1000.0, width_q / 1000.0)


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
        seq_len = _patchified_seq_len(latents, int(config.get("flux_shift_patchify_factor", 1)))
        if seq_len is None:
            return max(0.0, min(1.0, base_center))
        # Normalize seq_len to [0, 1] range (256..4096 → 0..1)
        res_norm = max(0.0, min(1.0, (seq_len - 256) / (4096 - 256)))
        cross = res_influence * progress * (1.0 - 2.0 * res_norm)
        base_center = base_center + cross

    return max(0.0, min(1.0, base_center))


# ── Public API ────────────────────────────────────────────────────────────


def validate_fixed_sigmas(values: Any) -> list[float]:
    """The ``fixed_sigmas`` list as floats, or a ``ValueError`` naming it.

    It must be a non-empty list or tuple whose every value lies in ``(0, 1]``: sigma 0 is the clean
    endpoint, never a training level, and a value past 1 is no flow-matching sigma.
    """
    if not isinstance(values, (list, tuple)) or len(values) == 0:
        raise ValueError(f"fixed_sigmas must be a non-empty list of sigmas in (0, 1], got {values!r}")
    out: list[float] = []
    for value in values:
        try:
            sigma = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"fixed_sigmas holds a non-number: {value!r}") from None
        if not 0.0 < sigma <= 1.0:  # also refuses NaN
            raise ValueError(f"fixed_sigmas values must lie in (0, 1], got {value!r}")
        out.append(sigma)
    return out


def _rand(bs: int, device: torch.device, generator: torch.Generator | None) -> torch.Tensor:
    """``U(0,1)`` of shape ``[bs]`` on ``device``. With a generator, drawn on its device and moved,
    so the global RNG is untouched; without one, exactly the plain ``torch.rand`` call."""
    if generator is None:
        return torch.rand((bs,), device=device)
    return torch.rand((bs,), device=generator.device, generator=generator).to(device)


def _randn(bs: int, device: torch.device, generator: torch.Generator | None) -> torch.Tensor:
    """``N(0,1)`` counterpart of :func:`_rand`."""
    if generator is None:
        return torch.randn((bs,), device=device)
    return torch.randn((bs,), device=generator.device, generator=generator).to(device)


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
        "model_shift",
        "fixed_sigmas",
    ]

    # ``draw_in_band``'s limits. A batch of 4 spends them over 29 rounds, and fills on
    # average down to a band of 1 in 250,000 of the distribution.
    BAND_MAX_ROUND_CANDIDATES = 65536
    BAND_MAX_CANDIDATES = 1_000_000

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
        generator: torch.Generator | None = None,
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
            generator: Optional local RNG. Every mode draws from it instead of the
                      global RNG; ``None`` keeps the global draws exactly as before.

        Returns:
            Timestep tensor ``[bs]`` in ``[0, 1]``.
        """
        if mode == "fixed_sigmas":
            # A uniform draw from the configured list (a model distilled to a fixed
            # schedule trains on that schedule's levels only).
            sigmas = validate_fixed_sigmas(config.get("fixed_sigmas"))
            gen_device = generator.device if generator is not None else device
            idx = torch.randint(0, len(sigmas), (bs,), device=gen_device, generator=generator)
            return torch.tensor(sigmas, dtype=torch.float32, device=gen_device)[idx].to(device)

        if mode == "uniform":
            return _rand(bs, device, generator)

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
                _randn(bs, device, generator) * sigma + mu
            )

        if mode == "sigmoid":
            return torch.sigmoid(_randn(bs, device, generator))

        if mode == "cosmap":
            u = _rand(bs, device, generator)
            return 1.0 - torch.cos(u * math.pi / 2.0)

        if mode == "mode":
            scale = float(config.get("mode_scale", 1.5))
            u = _rand(bs, device, generator)
            return 1.0 - u / (scale + (1.0 - scale) * u)

        if mode == "flux_shift":
            base_shift = float(config.get("flux_shift_base", 0.5))
            max_shift = float(config.get("flux_shift_max", 1.16))
            seq_len = _patchified_seq_len(
                latents, int(config.get("flux_shift_patchify_factor", 1))
            )
            if seq_len is not None:
                # Flux2 patchifies latents (2× down per spatial dim) before
                # the transformer, so the actual image sequence length the
                # model sees is (h/p)*(w/p).  Without this correction the mu
                # is far too high (3.27 instead of 1.16 for 1024px images),
                # heavily biasing training toward extreme-noise timesteps.
                m = (max_shift - base_shift) / (4096 - 256)
                b = base_shift - m * 256
                mu = seq_len * m + b
                # CLAMP to the declared shift band — the line is calibrated on
                # 256..4096 tokens and EXTRAPOLATES past it. A 2048px still
                # (16384 tokens) reaches mu 3.27 — the exact value the comment
                # above calls "far too high" — and a video latent goes past 4,
                # where sigmoid(N(0,1)+mu) puts essentially every sample at
                # t≈1 and training sees noise only. ``model_shift`` already
                # clamps its interpolation the same way.
                mu = max(min(mu, max(base_shift, max_shift)),
                         min(base_shift, max_shift))
            else:
                mu = (base_shift + max_shift) / 2.0
            return torch.sigmoid(_randn(bs, device, generator) + mu)

        if mode == "radc":
            center = _radc_center(config, progress, latents)
            width = float(config.get("radc_width", 0.5))
            cdf = _radc_cdf(round(center * 1000), round(width * 1000)).to(device)
            u = _rand(bs, device, generator)
            indices = torch.searchsorted(cdf, u)
            # searchsorted can return len(cdf) when u exceeds the final bin by a
            # float epsilon, which would yield t = 1.001; clamp so the raw
            # sample() contract (t in [0, 1]) holds for direct callers too, not
            # only for sample_scaled's own clamp.
            indices = indices.clamp(max=cdf.shape[0] - 1)
            return (indices.float() + 1) / 1000.0

        if mode == "model_shift":
            # Inference-matched flow-match shift, in additive logit space.
            #   LTX  (dynamic): mu interpolates base_shift→max_shift with seq-len.
            #   WAN  (fixed):   mu = ln(flow_shift)  (seq-len independent).
            #   else (fallback): flux-style mid shift.
            std = float(config.get("model_shift_std", 1.0))
            base_shift = config.get("model_shift_base_shift", None)
            max_shift = config.get("model_shift_max_shift", None)
            fixed = config.get("model_shift_fixed", None)
            seq_len = _patchified_seq_len(
                latents, int(config.get("flux_shift_patchify_factor", 1))
            )
            if base_shift is not None and max_shift is not None and seq_len is not None:
                base_seq = float(config.get("model_shift_base_seq", 1024))
                max_seq = float(config.get("model_shift_max_seq", 4096))
                bs_v, ms_v = float(base_shift), float(max_shift)
                m = (ms_v - bs_v) / max(max_seq - base_seq, 1.0)
                mu = bs_v + m * (float(seq_len) - base_seq)
                mu = max(min(mu, max(bs_v, ms_v)), min(bs_v, ms_v))  # clamp to range
            elif fixed is not None and float(fixed) > 0.0:
                mu = math.log(float(fixed))
            else:
                mu = (0.5 + 1.16) / 2.0
            t = torch.sigmoid(_randn(bs, device, generator) * std + mu)
            uniform_prob = float(config.get("timestep_uniform_prob", 0.1))
            if uniform_prob > 0.0:
                u_mask = _rand(bs, device, generator) < uniform_prob
                t = torch.where(u_mask, _rand(bs, device, generator), t)
            return t

        # Unknown → fallback with warning
        logger.warning("unknown_timestep_mode", mode=mode, fallback="uniform")
        return _rand(bs, device, generator)

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
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample timesteps and scale to ``[0, scale]``.

        Convenience wrapper around :meth:`sample` for the common
        ``t * 1000`` pattern used by most model families.

        Returns:
            Timestep tensor ``[bs]`` clamped to ``[0, scale]``.
        """
        # The generator is passed only when given, so a call without one is the same call as before.
        extra = {} if generator is None else {"generator": generator}
        t = TimestepSampler.sample(
            mode, bs, device, config, latents=latents, progress=progress, **extra,
        )
        return (t * scale).clamp(0, scale)

    # ── A distribution restricted to a band ───────────────────────────

    @staticmethod
    def draw_in_band(
        draw: Callable[[int], torch.Tensor],
        n: int,
        device: torch.device,
        in_band: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``n`` draws of a distribution conditioned on a band, by redrawing.

        ``draw(k)`` gives ``k`` candidates on ``device``; ``in_band`` masks them. The first
        round draws ``n``, so a band that takes all of them consumes the stream as one plain
        draw. Each later round doubles, up to ``BAND_MAX_ROUND_CANDIDATES``, and the rounds
        stop at ``BAND_MAX_CANDIDATES`` in all: a thin band costs a few rounds, and one with
        no mass ends rather than spins.

        Slots fill in order with the first accepted candidates, and a round's surplus is
        dropped. Which candidates are kept depends only on their order, never on their
        value, so the slots hold the distribution restricted to the band, whatever the
        number of rounds.

        Returns ``(out, filled)``. ``out`` is undefined where ``filled`` is False: the caller
        decides what an unfilled slot becomes.
        """
        out = torch.empty(n, device=device)
        got = 0
        drawn = 0
        size = n
        while got < n and drawn < TimestepSampler.BAND_MAX_CANDIDATES:
            candidates = draw(size)
            drawn += size
            accepted = candidates[in_band(candidates)][: n - got]
            out[got : got + accepted.shape[0]] = accepted
            got += accepted.shape[0]
            size = min(
                size * 2,
                TimestepSampler.BAND_MAX_ROUND_CANDIDATES,
                TimestepSampler.BAND_MAX_CANDIDATES - drawn,
            )
        filled = torch.zeros(n, dtype=torch.bool, device=device)
        filled[:got] = True
        return out, filled
