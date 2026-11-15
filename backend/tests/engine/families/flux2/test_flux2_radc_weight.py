"""FLUX.2 counts the radc curve once: the draw follows it, the loss is not weighted by it again.

The draw used to be weighted by the curve it was drawn from, which squared the emphasis (the
effective curve was about 0.7 of the set width). The centre each batch draws from still carries
its resolution term; it is taken from the arguments the real draw passes to ``_radc_cdf``, not
recomputed here, so a test can't agree with a wrong formula.
"""

from __future__ import annotations

import pytest
import torch

from app.engine.core.pipeline.pipeline_base import PipelineBaseMixin
from app.engine.models.families.flux2.driver import Flux2Driver
from app.engine.models.families.flux2.trainer import Flux2Trainer
from app.engine.strategies import timestep_sampling

ALL_TIMESTEPS = torch.arange(1, 1001, dtype=torch.float32)
LATENTS_256_TOKENS = torch.zeros(2, 32, 32, 32)
LATENTS_4096_TOKENS = torch.zeros(2, 32, 128, 128)


def _trainer(*, step: int = 100, max_steps: int = 100, **config) -> Flux2Trainer:
    t = object.__new__(Flux2Trainer)
    t.device = torch.device("cpu")
    t.driver = object.__new__(Flux2Driver)
    t.global_step = step
    t.max_train_steps = max_steps
    t.config = {
        "timestep_sampling": "radc",
        "radc_start": 0.8,
        "radc_end": 0.2,
        "radc_width": 0.5,
        "radc_res_influence": 0.15,
        "flux_shift_patchify_factor": 2,  # what Flux2Trainer._setup_family sets
        **config,
    }
    return t


@pytest.fixture
def drawn_curves(monkeypatch):
    """Every ``(center_q, width_q)`` the real draw asks ``_radc_cdf`` for."""
    calls: list[tuple[int, int]] = []
    real = timestep_sampling._radc_cdf

    def spy(center_q, width_q):
        calls.append((center_q, width_q))
        return real(center_q, width_q)

    monkeypatch.setattr(timestep_sampling, "_radc_cdf", spy)
    return calls


@pytest.mark.parametrize(
    ("step", "latents", "expected_q"),
    [
        (100, LATENTS_4096_TOKENS, 50),  # 0.2 - 0.15
        (100, LATENTS_256_TOKENS, 350),  # 0.2 + 0.15
        (50, LATENTS_4096_TOKENS, 425),  # 0.5 - 0.075
        (50, LATENTS_256_TOKENS, 575),  # 0.5 + 0.075
    ],
)
def test_the_draw_uses_the_expected_centre(drawn_curves, step, latents, expected_q):
    _trainer(step=step).sample_timesteps(2, latents)

    assert drawn_curves == [(expected_q, 500)]


def test_each_batch_gets_its_own_centre(drawn_curves):
    t = _trainer()
    for latents in (LATENTS_4096_TOKENS, LATENTS_256_TOKENS, LATENTS_4096_TOKENS):
        t.sample_timesteps(2, latents)

    assert [q for q, _ in drawn_curves] == [50, 350, 50]


@pytest.mark.parametrize("mode", ["radc", "logit_normal", "uniform", "sigmoid", "flux_shift"])
def test_flux2_radc_has_no_loss_weight(mode):
    t = _trainer(timestep_sampling=mode)
    t.sample_timesteps(2, LATENTS_4096_TOKENS)

    assert t.compute_loss_weight(ALL_TIMESTEPS) is None


@pytest.mark.parametrize("mode", ["radc", "logit_normal", "uniform", "sigmoid", "cosmap", "mode", "flux_shift", "model_shift"])
def test_flux2_draw_is_the_base_draw(mode):
    t = _trainer(step=50, timestep_sampling=mode)

    torch.manual_seed(7)
    expected = PipelineBaseMixin.sample_timesteps(t, 4, LATENTS_4096_TOKENS)
    after_expected = torch.rand(1)

    torch.manual_seed(7)
    drawn = t.sample_timesteps(4, LATENTS_4096_TOKENS)

    assert torch.equal(drawn, expected)
    assert torch.equal(torch.rand(1), after_expected)


def test_the_shared_module_keeps_no_weight_curve():
    assert not hasattr(timestep_sampling, "_radc_pdf")


# Pinned on main 54769ca (seed 0, step 50 of 100, batch 4, the real trainer and its profile latents).
MAIN_DRAWS_DEV = {
    "uniform": [496.256592, 768.221802, 88.477432, 132.030487],
    "logit_normal": [823.609497, 427.164642, 101.671448, 638.401123],
    "sigmoid": [823.609497, 427.164642, 101.671448, 638.401123],
    "cosmap": [288.74762, 643.913696, 9.642183, 21.428944],
    "mode": [603.588318, 311.560822, 939.222595, 907.927551],
    "flux_shift": [885.034668, 551.4599, 157.255844, 744.298462],
    "model_shift": [914.588745, 631.014404, 206.068359, 801.934814],
}


@pytest.mark.parametrize("mode", sorted(MAIN_DRAWS_DEV))
def test_flux2_other_modes_draw_as_before(mode):
    from app.engine.models.registry import registry
    from tests.test_engine_fields_reach_the_trainer import _PROFILES, _Profile, _trainer as _real_trainer

    registry.initialize()
    profile = _PROFILES.get("flux2", _Profile())
    trainer = _real_trainer(registry.get_definition("flux2-dev"), **{**profile.config, "timestep_sampling": mode})
    trainer.global_step, trainer.max_train_steps = 50, 100
    torch.manual_seed(0)
    drawn = trainer.sample_timesteps(4, torch.zeros(4, profile.latents[1], *profile.latents[2:]))
    assert torch.allclose(drawn, torch.tensor(MAIN_DRAWS_DEV[mode]), atol=1e-4), drawn.tolist()
    assert trainer.compute_loss_weight(drawn) is None
