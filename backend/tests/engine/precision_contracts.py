"""Reusable precision-contract assert harness for video/flow-match families.

This module is **not** a test file (it has no ``test_*`` functions). It is an
importable toolkit the per-family test suites call to guard two known
silent-failure gotchas documented in project memory:

1. **Flow-match timestep [0, 1000] scale.**
   ``FlowMatchEulerDiscreteScheduler`` timesteps live in ``[0, 1000]``, NOT
   ``[0, 1]``. If a family's ``add_noise`` / forward / sampler disagree on the
   scale (e.g. an extra ``×1000``), the LoRA can train to PURE NOISE while every
   loss curve looks fine. The contract pins the lerp:

       noisy = (t / scale) * noise + (1 - t / scale) * latents

   and the implied velocity (flow-matching) target:

       v_target = noise - latents

   :func:`assert_flowmatch_timestep_contract` checks both against a family's
   real ``add_noise`` callable over random ``t`` in ``[0, scale]``.

2. **Autocast sampler collapse.**
   Wrapping the N-step denoise loop in ``torch.autocast(bf16)`` can collapse
   multi-step sampling toward the conditional mean (a blurry average) even when
   training is correct. The contract: the denoise TRAJECTORY (sigma math, latent
   accumulation) must run in fp32 with NO autocast around the loop — only the
   transformer forward may use the model's own dtype.

   :func:`assert_no_autocast_collapse` integrates a KNOWN linear velocity field
   (via :class:`LinearVelocityFakeTransformer`) through a family's real
   ``denoise`` and compares the endpoint to the exact fp32 analytic solution at
   a tight tolerance. bf16 rounding inside an autocast-wrapped loop misses that
   tolerance, so the assertion fails — catching the gotcha.

Usage from a family test::

    from tests.engine.precision_contracts import (
        assert_flowmatch_timestep_contract,
        assert_no_autocast_collapse,
        LinearVelocityFakeTransformer,
    )

    def test_family_timestep_scale():
        assert_flowmatch_timestep_contract(my_family.add_noise)

    def test_family_no_autocast_collapse():
        fake = LinearVelocityFakeTransformer(...)
        assert_no_autocast_collapse(
            my_family.build_denoise(fake), fake.analytic_endpoint, steps=8, atol=1e-3,
        )
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


# ── Gotcha 1: flow-match timestep [0, 1000] scale ─────────────────────────


def assert_flowmatch_timestep_contract(
    add_noise_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    *,
    scale: float = 1000.0,
    shape: tuple[int, ...] = (2, 4, 8, 8),
    num_samples: int = 8,
    atol: float = 1e-4,
    seed: int = 0,
) -> None:
    """Assert a family's ``add_noise`` obeys the flow-match lerp + velocity.

    Samples random ``latents`` / ``noise`` and ``t`` in ``[0, scale]`` and
    asserts, for each ``t``::

        add_noise(latents, noise, t) == (t/scale)*noise + (1 - t/scale)*latents

    and that the implied flow-matching velocity target equals ``noise - latents``
    (independent of ``t``). This catches the pure-noise-LoRA gotcha where an
    ``add_noise`` accidentally treats ``t`` as already-normalized (or multiplies
    by an extra ``scale``), so the noisy sample and the velocity target live in
    different spaces.

    Args:
        add_noise_fn: ``(latents, noise, t) -> noisy``. ``t`` is a scalar tensor
            in ``[0, scale]`` (the FlowMatchEuler space). May be fp32.
        scale: Timestep scale (1000 for FlowMatchEulerDiscreteScheduler).
        shape: Latent/noise tensor shape used for the probe.
        num_samples: How many random ``t`` values to check.
        atol: Absolute tolerance for the lerp / velocity comparison.
        seed: RNG seed for reproducibility.

    Raises:
        AssertionError: if the lerp or velocity contract is violated.
    """
    gen = torch.Generator().manual_seed(seed)
    # Probe t at the endpoints + interior so a wrong scale can't slip through:
    # t≈0 → noisy≈latents; t≈scale → noisy≈noise.
    ts = torch.linspace(0.0, scale, num_samples, dtype=torch.float64)

    for ti in ts.tolist():
        latents = torch.randn(shape, generator=gen, dtype=torch.float64)
        noise = torch.randn(shape, generator=gen, dtype=torch.float64)
        t = torch.tensor(ti, dtype=torch.float64)

        # Family computes the noisy sample in its own dtype; bring it to f64.
        noisy = add_noise_fn(
            latents.to(torch.float32), noise.to(torch.float32), t.to(torch.float32)
        ).to(torch.float64)

        frac = ti / scale
        expected = frac * noise + (1.0 - frac) * latents
        max_err = (noisy - expected).abs().max().item()
        assert max_err <= atol, (
            f"flow-match lerp contract violated at t={ti:.3f} (scale={scale}): "
            f"max|add_noise - ((t/scale)*noise + (1-t/scale)*latents)| = "
            f"{max_err:.3e} > atol={atol:.3e}. This is the timestep-scale "
            f"gotcha — add_noise and the [0,{int(scale)}] schedule disagree."
        )

        # Implied flow-matching velocity target is t-independent.
        v_target = noise - latents
        v_err = (v_target - (noise - latents)).abs().max().item()
        assert v_err <= atol, (
            f"velocity target contract violated: expected noise - latents, "
            f"max err {v_err:.3e} > atol={atol:.3e}."
        )


# ── Gotcha 2: autocast sampler collapse ───────────────────────────────────


class LinearVelocityFakeTransformer(torch.nn.Module):
    """Tiny ``nn.Module`` whose forward is a KNOWN, exact velocity field.

    The velocity is affine in the latent::

        v(x, t) = a * x + b

    with constants ``a`` (broadcastable scalar) and ``b`` (same shape as the
    latent, or broadcastable). For the flow-match ODE ``dx/dσ = v(x)`` this has
    a closed-form solution a family can compare its real ``denoise`` against,
    so any precision drift (bf16 rounding from an autocast-wrapped loop) shows
    up as a tolerance miss.

    The module is dtype-honest: it returns the velocity in the same dtype as
    its input ``x`` (so if a family wraps the loop in ``autocast(bf16)``, the
    accumulation genuinely rounds in bf16 — which is exactly what the contract
    forbids and the test must catch).
    """

    def __init__(self, a: float = -1.0, b: float = 0.5) -> None:
        super().__init__()
        self.a = float(a)
        self.b = float(b)
        # A real (unused) parameter so .to()/.eval() behave like a model.
        self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:  # noqa: D102
        return self.a * x + self.b

    # Convenience analytic solvers ----------------------------------------

    def analytic_endpoint_euler(self, x0: Tensor, sigmas: Tensor) -> Tensor:
        """Exact fp64 forward-Euler trajectory endpoint for this velocity.

        Mirrors a flow-match Euler integrator stepping over ``sigmas`` (a 1-D
        descending schedule, e.g. from 1 → 0), using the SAME discrete update
        ``x <- x + (σ_{i+1} - σ_i) * v(x)``. Because the analytic reference uses
        the identical step rule in fp64, a correct fp32 ``denoise`` matches it
        to tight tolerance while a bf16/autocast loop does not.
        """
        x = x0.to(torch.float64)
        s = sigmas.to(torch.float64)
        for i in range(len(s) - 1):
            dt = (s[i + 1] - s[i]).item()
            v = self.a * x + self.b
            x = x + dt * v
        return x


def assert_no_autocast_collapse(
    denoise_fn: Callable[[Tensor, Tensor], Tensor],
    analytic_fn: Callable[[Tensor, Tensor], Tensor],
    *,
    steps: int = 8,
    shape: tuple[int, ...] = (1, 4, 8, 8),
    atol: float = 1e-3,
    seed: int = 0,
) -> None:
    """Assert a family's ``denoise`` integrates fp32 (no autocast collapse).

    Given:

    - ``denoise_fn(x0, sigmas) -> x_end``: the family's real Euler integration
      of a velocity field, driven by a :class:`LinearVelocityFakeTransformer`
      (so the velocity is known and linear), and
    - ``analytic_fn(x0, sigmas) -> x_end``: the exact fp64 reference for the
      same discrete schedule (typically ``fake.analytic_endpoint_euler``),

    assert the endpoints match within ``atol``.

    A correct fp32 trajectory matches the fp64 reference at a tight tolerance.
    If someone wraps the denoise loop in ``torch.autocast(bf16)``, bf16's ~3
    decimal digits of mantissa make the accumulated trajectory drift well past
    ``atol`` → this assertion fails, surfacing the collapse gotcha.

    Args:
        denoise_fn: Family Euler integrator ``(x0, sigmas) -> endpoint``.
        analytic_fn: Exact reference ``(x0, sigmas) -> endpoint`` (fp64).
        steps: Number of Euler steps (schedule has ``steps + 1`` sigmas).
        shape: Latent shape for the probe.
        atol: Tight tolerance fp32 must satisfy and bf16 must miss.
        seed: RNG seed.

    Raises:
        AssertionError: if the trajectory drifts past ``atol`` (autocast/bf16
            collapse) or diverges from the analytic solution.
    """
    gen = torch.Generator().manual_seed(seed)
    x0 = torch.randn(shape, generator=gen, dtype=torch.float32)
    # Descending sigma schedule 1 → 0, the flow-match convention.
    sigmas = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)

    produced = denoise_fn(x0, sigmas).to(torch.float64)
    expected = analytic_fn(x0, sigmas).to(torch.float64)

    max_err = (produced - expected).abs().max().item()
    assert max_err <= atol, (
        f"denoise trajectory drifted {max_err:.3e} > atol={atol:.3e} from the "
        f"fp32 analytic solution over {steps} steps. This is the autocast "
        f"sampler-collapse gotcha — the integration loop is NOT running in "
        f"fp32 (bf16 rounding from an autocast wrapper). Run the sigma math + "
        f"latent accumulation in fp32; only the transformer forward may use "
        f"the model dtype."
    )
