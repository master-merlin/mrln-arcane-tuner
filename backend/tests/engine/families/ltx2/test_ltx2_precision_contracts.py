"""LTX 2.3 precision-contract tests — flow-match scale + autocast collapse.

These exercise the REAL driver ``add_noise`` and the REAL sampler Euler
integration against the shared precision harness, guarding the two documented
silent-failure gotchas:

1. Flow-match [0, 1000] timestep scale (pure-noise-LoRA gotcha).
2. fp32 denoise trajectory (autocast sampler collapse).
"""

import torch

from app.engine.models.families.ltx2.driver import Ltx2Driver
from app.engine.models.families.ltx2.sampler import Ltx2Sampler
from tests.engine.precision_contracts import (
    LinearVelocityFakeTransformer,
    assert_flowmatch_timestep_contract,
    assert_no_autocast_collapse,
)


def _make_driver() -> Ltx2Driver:
    """A driver with no weights — add_noise needs only device + math."""
    return Ltx2Driver(definition=None, device=torch.device("cpu"))


# ── Gotcha 1: flow-match [0, 1000] timestep scale ─────────────────────────


def test_ltx2_add_noise_obeys_flowmatch_contract():
    """REAL driver add_noise must lerp on the [0, 1000] scale (no extra ×1000)."""
    driver = _make_driver()
    assert_flowmatch_timestep_contract(driver.add_noise)


def test_ltx2_compute_target_is_velocity():
    """Target is the t-independent flow-match velocity ``noise - latents``."""
    driver = _make_driver()
    latents = torch.randn(2, 4, 8, 8)
    noise = torch.randn(2, 4, 8, 8)
    t = torch.tensor([123.0, 777.0])
    target = driver.compute_target(latents, noise, t)
    assert torch.allclose(target, noise - latents)


def test_ltx2_wrong_add_noise_fails_contract():
    """Sanity: a deliberately wrong add_noise (extra ×1000) FAILS the contract.

    This proves the harness actually catches the timestep-scale gotcha rather
    than passing vacuously.
    """

    def wrong_add_noise(latents, noise, timesteps):
        # BUG: treats t as already-normalized then re-scales — the classic
        # extra-×1000 mistake that trains a pure-noise LoRA.
        t = timesteps
        while t.ndim < latents.ndim:
            t = t.unsqueeze(-1)
        return t * noise + (1 - t) * latents  # t is in [0,1000], not [0,1]

    raised = False
    try:
        assert_flowmatch_timestep_contract(wrong_add_noise)
    except AssertionError:
        raised = True
    assert raised, "wrong add_noise should have violated the flow-match contract"


# ── Gotcha 2: autocast sampler collapse ───────────────────────────────────


def test_ltx2_denoise_no_autocast_collapse():
    """REAL sampler Euler integration matches the fp32 analytic endpoint.

    Drives the sampler's ``euler_denoise`` (via ``build_denoise``) with a known
    linear velocity field.  A correct fp32 loop matches the analytic reference
    to tight tolerance; a bf16/autocast-wrapped loop would drift past it.
    """
    sampler = object.__new__(Ltx2Sampler)  # no pipeline needed for the closure
    fake = LinearVelocityFakeTransformer(a=-1.0, b=0.5)
    denoise = sampler.build_denoise(fake)
    assert_no_autocast_collapse(
        denoise, fake.analytic_endpoint_euler, steps=8, atol=1e-3,
    )
