"""WAN 2.1 precision-contract tests (THE point of phase B3).

Two silent-failure gotchas are guarded against the REAL WAN code paths:

1. Flow-match timestep [0, 1000] scale — the REAL ``Wan21Driver.add_noise`` is
   fed to ``assert_flowmatch_timestep_contract``. A deliberately-wrong ``×1000``
   variant is also checked to FAIL the contract (the test bites).

2. Autocast sampler collapse — a ``LinearVelocityFakeTransformer`` is plugged
   into the REAL ``Wan21Sampler`` fp32 Euler integrator (via ``build_denoise``)
   and the endpoint is compared to the fp64 analytic solution. The integrator
   accumulates in fp32 with no autocast, so it matches at a tight tolerance.
"""

import torch

from app.engine.models.families.wan21.driver import Wan21Driver
from app.engine.models.families.wan21.sampler import Wan21Sampler
from tests.engine.precision_contracts import (
    LinearVelocityFakeTransformer,
    assert_flowmatch_timestep_contract,
    assert_no_autocast_collapse,
)


class _Defn:
    """Minimal definition stand-in (no weights, no YAML)."""

    architecture_params = {"mode": "t2v", "te.max_length": 512}
    lora_targetable_modules: list[str] = []


class _Pipeline:
    """Minimal trainer stub for sampler construction."""

    def __init__(self):
        self.config = {"resolutions": [480]}
        self.device = torch.device("cpu")


def _make_driver() -> Wan21Driver:
    return Wan21Driver(_Defn(), torch.device("cpu"))


def _make_sampler() -> Wan21Sampler:
    return Wan21Sampler(_Pipeline())


# ── Gotcha 1: flow-match [0, 1000] lerp on the REAL add_noise ──────────────


def test_real_add_noise_obeys_flowmatch_contract():
    driver = _make_driver()
    # Exercise the REAL driver.add_noise — t in [0, 1000].
    assert_flowmatch_timestep_contract(driver.add_noise, scale=1000.0)


def test_wrong_x1000_add_noise_fails_contract():
    """Sanity: a deliberately-wrong add_noise must FAIL the contract."""
    driver = _make_driver()

    def wrong_add_noise(latents, noise, t):
        # BUG: treat t as already in [0, 1] (extra ×1000 mismatch) — the
        # classic pure-noise-LoRA gotcha.
        while t.ndim < latents.ndim:
            t = t.unsqueeze(-1)
        return t * noise + (1.0 - t) * latents

    raised = False
    try:
        assert_flowmatch_timestep_contract(wrong_add_noise, scale=1000.0)
    except AssertionError:
        raised = True
    assert raised, "wrong ×1000 add_noise should have failed the contract"
    # And the REAL one still passes — proves the difference is meaningful.
    assert_flowmatch_timestep_contract(driver.add_noise, scale=1000.0)


# ── Gotcha 2: autocast sampler collapse on the REAL denoise integrator ─────


def test_real_sampler_denoise_no_autocast_collapse():
    sampler = _make_sampler()
    fake = LinearVelocityFakeTransformer(a=-1.0, b=0.5)
    # build_denoise wraps the REAL euler_integrate fp32 loop around the fake.
    denoise_fn = sampler.build_denoise(fake)
    assert_no_autocast_collapse(
        denoise_fn,
        fake.analytic_endpoint_euler,
        steps=8,
        atol=1e-3,
    )


def test_autocast_wrapped_loop_would_collapse():
    """Sanity: a bf16/autocast-wrapped integrator must MISS the fp32 tolerance.

    This proves the contract bites — it is NOT a vacuous pass.
    """
    fake = LinearVelocityFakeTransformer(a=-1.0, b=0.5)

    def bf16_denoise(x0, sigmas):
        # WRONG: accumulate the trajectory in bf16 (what an autocast wrapper
        # around the loop would do).
        x = x0.to(torch.bfloat16)
        s = sigmas.to(torch.float32)
        for i in range(len(s) - 1):
            dt = (s[i + 1] - s[i]).item()
            v = (fake.a * x + fake.b).to(torch.bfloat16)
            x = (x + dt * v).to(torch.bfloat16)
        return x.to(torch.float32)

    raised = False
    try:
        assert_no_autocast_collapse(
            bf16_denoise, fake.analytic_endpoint_euler, steps=8, atol=1e-3
        )
    except AssertionError:
        raised = True
    assert raised, "bf16-accumulated loop should have failed the fp32 contract"
