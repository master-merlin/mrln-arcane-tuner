"""WAN 2.2 ``radc`` follows the curriculum, through the call the training loop makes.

``trainer.sample_timesteps`` reaches the driver and the router with the training progress, and the
router's ``p_high`` estimate is refreshed from it at the shape the draws see. Every other mode keeps
its exact draws and ``p_high``: the literal values below were read from the code before ``progress``
was forwarded.

CPU only, no weights: a bare ``Wan22Trainer`` around a real driver and the router ``_build_router``
builds.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from app.engine.models.families.bernini_r.trainer import BerniniRTrainer
from app.engine.models.families.wan22.driver import Wan22Driver
from app.engine.models.families.wan22.expert_router import HIGH, LOW
from app.engine.models.families.wan22.trainer import Wan22Trainer

CPU = torch.device("cpu")
BOUNDARY = 875.0
VIDEO_LATENT = (1, 16, 21, 128, 128)


class _Defn:
    architecture_params = {
        "mode": "t2v",
        "te.max_length": 512,
        "moe.boundary_ratio": 0.875,
    }
    lora_targetable_modules: list[str] = []
    defaults: dict = {"num_frames": 81}


def _trainer(config=None, *, expert_mode="both", step=0, max_steps=100) -> Wan22Trainer:
    t = object.__new__(Wan22Trainer)
    t.device = CPU
    t.config = {
        "expert_switch_interval": 1,
        "seed": 7,
        "resolutions": [1024],
        **(config or {}),
    }
    t.definition = _Defn()
    t.driver = Wan22Driver(_Defn(), CPU)
    t.expert_mode = expert_mode
    t.global_step = step
    t.max_train_steps = max_steps
    t._build_router()
    return t


def _radc(**extra) -> dict:
    return {"timestep_sampling": "radc", **extra}


def _draws(t: Wan22Trainer, step: int, n: int = 4096, seed: int = 0) -> torch.Tensor:
    t.global_step = step
    torch.manual_seed(seed)
    return t.sample_timesteps(n, torch.zeros(*VIDEO_LATENT))


def _p_high_after_step(t: Wan22Trainer, step: int) -> float:
    """The estimate the real loop gets: the driver's hook after the optimizer step at ``step``."""
    t.global_step = step
    t.driver.on_optimizer_step(step)
    return t.expert_router.p_high


# ── Acceptance 1: the draws follow the curriculum ──


@pytest.mark.parametrize(
    ("expert_mode", "active"),
    [("both", HIGH), ("both", LOW), ("high", HIGH), ("low", LOW)],
    ids=["dual-high", "dual-low", "pinned-high", "pinned-low"],
)
def test_radc_mean_noise_falls_with_progress_and_stays_in_the_expert_range(expert_mode, active):
    t = _trainer(_radc(), expert_mode=expert_mode)
    t.driver._set_active(active)

    early, late = _draws(t, 10), _draws(t, 90)

    assert late.mean() < early.mean(), (early.mean().item(), late.mean().item())
    for ts in (early, late):
        if active == HIGH:
            assert bool((ts >= BOUNDARY).all()) and bool((ts <= 1000.0).all())
        else:
            assert bool((ts < BOUNDARY).all()) and bool((ts >= 0.0).all())


def test_radc_progress_is_the_step_fraction_all_the_way_to_the_sampler(monkeypatch):
    from app.engine.strategies.timestep_sampling import TimestepSampler

    seen: list[float] = []
    real = TimestepSampler.sample_scaled

    def sample_scaled(mode, bs, device, config, *, scale=1000.0, latents=None, progress=0.0):
        seen.append(progress)
        return real(mode, bs, device, config, scale=scale, latents=latents, progress=progress)

    t = _trainer(_radc(), expert_mode="low")
    monkeypatch.setattr(TimestepSampler, "sample_scaled", staticmethod(sample_scaled))
    _draws(t, 25, n=8)

    assert seen and set(seen) == {0.25}


# ── Acceptance 1: p_high follows it, and its draws leave the training streams alone ──


def test_radc_p_high_falls_with_progress_through_the_driver_hook():
    t = _trainer(_radc())
    early = _p_high_after_step(t, 10)
    late = _p_high_after_step(t, 90)
    assert late < early, (early, late)


def test_radc_p_high_is_the_estimate_at_the_step_and_the_reference_shape():
    t = _trainer(_radc())
    got = _p_high_after_step(t, 40)
    want = t.expert_router.estimate_p_high(progress=0.4, latent_shape=t._routing_reference_shape())
    assert got == want


def test_radc_defaults_take_p_high_to_what_the_curriculum_asks_for():
    t = _trainer(_radc())
    start = t.expert_router.p_high
    end = t.expert_router.estimate_p_high(progress=1.0)
    assert 0.2 < start < 0.4, start
    assert end < 0.01, end


def _streams():
    cuda = [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    return torch.get_rng_state().clone(), cuda, random.getstate(), np.random.get_state()


def test_radc_estimate_does_not_move_the_training_streams():
    t = _trainer(_radc())
    t.global_step = 40
    torch.rand(8)  # a training step drew, so a reseed could not coincide with the state
    cpu_before, cuda_before, py_before, np_before = _streams()

    t.driver.on_optimizer_step(40)  # a fresh block: a re-estimate

    cpu_after, cuda_after, py_after, np_after = _streams()
    assert torch.equal(cpu_before, cpu_after)
    assert len(cuda_before) == len(cuda_after)
    assert all(torch.equal(a, b) for a, b in zip(cuda_before, cuda_after, strict=True))
    assert py_before == py_after
    assert np_before[0] == np_after[0] and np.array_equal(np_before[1], np_after[1])
    assert np_before[2:] == np_after[2:]


def test_radc_router_built_before_the_loop_starts_estimates_at_progress_zero():
    """``_setup_family`` builds the router before ``global_step`` and ``max_train_steps`` exist."""
    t = object.__new__(Wan22Trainer)
    t.device = CPU
    t.config = {**_radc(), "resolutions": [1024], "seed": 7}
    t.definition = _Defn()
    t.driver = Wan22Driver(_Defn(), CPU)
    t.expert_mode = "both"
    router = t._build_router()
    assert router.p_high == router.estimate_p_high(progress=0.0)


# ── Acceptance 1: every other mode keeps its draws and p_high ──

# Read from the code before ``progress`` was forwarded: seed 7, manual_seed(123), step 50 of 100.
# The HIGH rows need redraws, so they follow the band's rounds of 4, 8, 16...: each is the first
# four in-band values of those plain draws at progress 0, replayed outside the band code.
BEFORE = {
    ("logit_normal", HIGH): [980.07, 927.581, 939.549, 929.147],
    ("logit_normal", LOW): [472.162, 530.054, 408.629, 440.183],
    ("flux_shift", HIGH): [955.039, 926.144, 876.026, 896.464],
    ("flux_shift", LOW): [740.493, 782.512, 687.91, 714.957],
}
BEFORE_P_HIGH = {"logit_normal": 0.025439999997615814, "flux_shift": 0.13311000168323517}


@pytest.mark.parametrize("mode", ["logit_normal", "flux_shift"])
@pytest.mark.parametrize(
    ("expert_mode", "active"),
    [("both", HIGH), ("both", LOW), ("high", HIGH), ("low", LOW)],
    ids=["dual-high", "dual-low", "pinned-high", "pinned-low"],
)
def test_other_modes_keep_their_draws_and_p_high(mode, expert_mode, active):
    t = _trainer({"timestep_sampling": mode}, expert_mode=expert_mode, step=50)
    t.driver._set_active(active)

    got = _draws(t, 50, n=4, seed=123)

    assert got.tolist() == pytest.approx(BEFORE[(mode, active)], abs=0.01)
    if expert_mode == "both":
        assert t.expert_router.p_high == pytest.approx(BEFORE_P_HIGH[mode], abs=1e-4)
    else:
        assert t.expert_router.p_high == (1.0 if active == HIGH else 0.0)


@pytest.mark.parametrize("mode", ["logit_normal", "flux_shift"])
def test_other_modes_get_no_progress_provider_and_no_reference_shape(mode):
    router = _trainer({"timestep_sampling": mode}).expert_router
    assert router._progress_provider is None
    assert router._reference_latent_shape is None


@pytest.mark.parametrize("mode", ["logit_normal", "flux_shift"])
def test_other_modes_p_high_stays_as_built_through_a_run(mode):
    t = _trainer({"timestep_sampling": mode})
    built = t.expert_router.p_high
    assert _p_high_after_step(t, 10) == built
    assert _p_high_after_step(t, 90) == built


# ── Acceptance 2: one reference shape, the same on both families ──

SHAPES = [
    ({"resolutions": [512], "num_frames": 81}, (1, 16, 21, 64, 64)),
    ({"resolutions": [1024]}, (1, 16, 21, 128, 128)),  # frames from the definition
    ({}, (1, 16, 21, 128, 128)),  # both from the defaults
    ({"resolutions": [512], "num_frames": 1}, (1, 16, 1, 64, 64)),
    ({"resolutions": [768, 512], "num_frames": 17}, (1, 16, 5, 96, 96)),
    ({"resolutions": [], "num_frames": 0}, (1, 16, 1, 128, 128)),
    ({"resolutions": [4], "num_frames": 5}, (1, 16, 2, 1, 1)),
]


def _bare(cls, config, *, definition=_Defn()):
    t = object.__new__(cls)
    t.config = config
    if definition is not None:
        t.definition = definition
    return t


@pytest.mark.parametrize(("config", "want"), SHAPES)
def test_bernini_r_reference_shape_is_unchanged_by_the_move(config, want):
    assert BerniniRTrainer._routing_reference_shape(_bare(BerniniRTrainer, config)) == want


@pytest.mark.parametrize(("config", "want"), SHAPES)
def test_wan22_estimates_at_the_same_shape_for_the_same_config(config, want):
    from app.engine.models.families.wan_shared.vae_utils import routing_reference_shape

    bernini = _bare(BerniniRTrainer, config)._routing_reference_shape()
    wan = _bare(Wan22Trainer, config)._routing_reference_shape()
    assert wan == bernini == want
    assert routing_reference_shape(config, _Defn()) == want


@pytest.mark.parametrize("cls", [BerniniRTrainer, Wan22Trainer])
def test_a_trainer_without_a_definition_uses_the_default_frames(cls):
    t = _bare(cls, {"resolutions": [512]}, definition=None)
    assert t._routing_reference_shape() == (1, 16, 21, 64, 64)


def test_wan22_radc_router_holds_the_reference_shape():
    t = _trainer(_radc(resolutions=[512], num_frames=81))
    assert t.expert_router._reference_latent_shape == (1, 16, 21, 64, 64)


def test_wan22_radc_estimate_sees_the_resolution_term():
    def p_high(resolution):
        t = _trainer(_radc(resolutions=[resolution], num_frames=1, radc_res_influence=1.0))
        return t.expert_router.estimate_p_high(progress=0.5)

    assert abs(p_high(64) - p_high(1024)) > 0.02
