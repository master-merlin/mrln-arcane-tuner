"""Self-test for the precision-contract harness.

Proves the harness in ``precision_contracts.py`` actually catches the two
known silent-failure gotchas:

- :func:`assert_flowmatch_timestep_contract` PASSES a correct ``add_noise``
  and FAILS the deliberately wrong "extra ×1000" variant (the pure-noise-LoRA
  timestep-scale gotcha).
- :func:`assert_no_autocast_collapse` PASSES a correct fp32 Euler ``denoise``
  and FAILS one whose loop is wrapped in ``torch.autocast(bf16)`` (the
  sampler-collapse gotcha).

If these self-tests are green, family suites can trust the helpers.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

try:  # pytest rootdir puts ``tests/`` on sys.path → package is ``engine``.
    from engine.precision_contracts import (
        LinearVelocityFakeTransformer,
        assert_flowmatch_timestep_contract,
        assert_no_autocast_collapse,
    )
except ImportError:  # pragma: no cover - alt layout
    from tests.engine.precision_contracts import (
        LinearVelocityFakeTransformer,
        assert_flowmatch_timestep_contract,
        assert_no_autocast_collapse,
    )


SCALE = 1000.0


# ── add_noise variants ────────────────────────────────────────────────────


def _correct_add_noise(latents, noise, t):
    """Contract-correct flow-match lerp: t lives in [0, 1000]."""
    frac = t / SCALE
    return frac * noise + (1.0 - frac) * latents


def _wrong_add_noise_extra_x1000(latents, noise, t):
    """The gotcha: treats ``t`` as already in [0,1] and re-divides → wrong
    space (equivalently an extra ×1000 mismatch). Produces noise≈latents for
    all real t in [0,1000], so the LoRA trains toward pure noise."""
    frac = t / (SCALE * SCALE)  # extra /1000 — scale mismatch
    return frac * noise + (1.0 - frac) * latents


# ── denoise variants (Euler integration of the fake velocity) ─────────────


def _make_fp32_denoise(fake: LinearVelocityFakeTransformer):
    """Correct: integrate the velocity in fp32, no autocast around the loop."""

    def denoise(x0, sigmas):
        x = x0.to(torch.float32)
        s = sigmas.to(torch.float32)
        for i in range(len(s) - 1):
            dt = s[i + 1] - s[i]
            v = fake(x)  # forward in fp32
            x = x + dt * v
        return x

    return denoise


def _make_autocast_denoise(fake: LinearVelocityFakeTransformer):
    """Gotcha: the WHOLE integration loop runs under autocast(bf16), so the
    sigma math + latent accumulation round in bf16 and drift toward the mean."""

    def denoise(x0, sigmas):
        # CPU autocast supports bf16; this is the wrong place to put it.
        ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        x = x0.to(torch.bfloat16)
        s = sigmas.to(torch.bfloat16)
        with ctx:
            for i in range(len(s) - 1):
                dt = s[i + 1] - s[i]
                v = fake(x)
                x = x + dt * v
        return x.to(torch.float32)

    return denoise


# ── flow-match timestep contract ─────────────────────────────────────────


def test_correct_add_noise_passes_timestep_contract():
    assert_flowmatch_timestep_contract(_correct_add_noise, scale=SCALE)


def test_wrong_add_noise_fails_timestep_contract():
    with pytest.raises(AssertionError, match="timestep|lerp|scale"):
        assert_flowmatch_timestep_contract(_wrong_add_noise_extra_x1000, scale=SCALE)


# ── autocast-collapse contract ───────────────────────────────────────────


def test_correct_fp32_denoise_passes_autocast_contract():
    fake = LinearVelocityFakeTransformer(a=-1.0, b=0.5).eval()
    assert_no_autocast_collapse(
        _make_fp32_denoise(fake),
        fake.analytic_endpoint_euler,
        steps=8,
        atol=1e-4,
    )


def test_autocast_wrapped_denoise_fails_autocast_contract():
    fake = LinearVelocityFakeTransformer(a=-1.0, b=0.5).eval()
    # bf16 rounding over the loop must miss the tight fp32 tolerance.
    with pytest.raises(AssertionError, match="autocast|fp32|drift"):
        assert_no_autocast_collapse(
            _make_autocast_denoise(fake),
            fake.analytic_endpoint_euler,
            steps=8,
            atol=1e-4,
        )


def test_fake_transformer_is_linear_velocity():
    """Sanity: the fake returns a*x + b in the input dtype."""
    fake = LinearVelocityFakeTransformer(a=2.0, b=-1.0)
    x = torch.ones(2, 3)
    out = fake(x)
    assert torch.allclose(out, 2.0 * x - 1.0)
    # dtype honesty: bf16 in → bf16 out (so autocast genuinely rounds).
    with contextlib.suppress(RuntimeError):
        xb = x.to(torch.bfloat16)
        assert fake(xb).dtype == torch.bfloat16
