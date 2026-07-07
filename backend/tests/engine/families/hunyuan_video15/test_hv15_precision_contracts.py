"""hv15 precision-contract tests against the REAL driver/sampler code paths.

1. Flow-match timestep [0, 1000] scale — the REAL ``Hv15Driver.add_noise``
   divides by 1000 exactly ONCE (the LERP); a deliberately-wrong variant must
   FAIL the contract (proves the test bites).
2. Autocast sampler collapse — a linear-velocity fake transformer runs through
   the REAL ``Hv15Sampler.euler_integrate`` fp32 loop (via ``build_denoise``)
   and matches the fp64 analytic endpoint; a bf16-accumulated loop must MISS.
"""

import torch

from app.engine.models.families.hunyuan_video15.driver import Hv15Driver
from app.engine.models.families.hunyuan_video15.sampler import Hv15Sampler
from tests.engine.precision_contracts import (
    LinearVelocityFakeTransformer,
    assert_flowmatch_timestep_contract,
    assert_no_autocast_collapse,
)


class _Defn:
    architecture_params = {"mode": "t2v", "transformer.num_layers": 1}
    lora_targetable_modules: list[str] = []


class _Pipeline:
    def __init__(self):
        self.config = {"resolutions": [480]}
        self.device = torch.device("cpu")
        self.definition = _Defn()


def _make_driver() -> Hv15Driver:
    return Hv15Driver(_Defn(), torch.device("cpu"))


def _make_sampler() -> Hv15Sampler:
    return Hv15Sampler(_Pipeline())


# ── Gotcha 1: flow-match [0, 1000] lerp on the REAL add_noise ──────────────


def test_real_add_noise_obeys_flowmatch_contract():
    driver = _make_driver()
    assert_flowmatch_timestep_contract(driver.add_noise, scale=1000.0)


def test_wrong_x1000_add_noise_fails_contract():
    """Sanity: an add_noise that treats t as already-[0,1] must FAIL."""
    driver = _make_driver()

    def wrong_add_noise(latents, noise, t):
        while t.ndim < latents.ndim:
            t = t.unsqueeze(-1)
        return t * noise + (1.0 - t) * latents

    raised = False
    try:
        assert_flowmatch_timestep_contract(wrong_add_noise, scale=1000.0)
    except AssertionError:
        raised = True
    assert raised, "wrong x1000 add_noise should have failed the contract"
    assert_flowmatch_timestep_contract(driver.add_noise, scale=1000.0)


# ── Gotcha 2: autocast sampler collapse on the REAL euler integrator ───────


def test_real_sampler_denoise_no_autocast_collapse():
    sampler = _make_sampler()
    fake = LinearVelocityFakeTransformer(a=-1.0, b=0.5)
    denoise_fn = sampler.build_denoise(fake)
    assert_no_autocast_collapse(
        denoise_fn,
        fake.analytic_endpoint_euler,
        steps=8,
        atol=1e-3,
    )


def test_autocast_wrapped_loop_would_collapse():
    """Sanity: a bf16-accumulated integrator must MISS the fp32 tolerance."""
    fake = LinearVelocityFakeTransformer(a=-1.0, b=0.5)

    def bf16_denoise(x0, sigmas):
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
