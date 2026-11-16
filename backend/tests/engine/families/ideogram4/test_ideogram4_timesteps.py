"""Ideogram 4's clock: ``t / 1000`` is the clean fraction (t = 0 noise, t = 1000 data).

The shared sampler draws a noise fraction, so the family turns it into its clock (``1000 - t``),
and every mode leans to the side its help names.
"""

from __future__ import annotations

import pytest
import torch

from app.engine.models.families.ideogram4.trainer import IdeogramV4Trainer
from app.engine.models.registry import registry
from app.engine.strategies.timestep_sampling import TimestepSampler


def _trainer(**config) -> IdeogramV4Trainer:
    registry.initialize()
    trainer = object.__new__(IdeogramV4Trainer)
    trainer.device = torch.device("cpu")
    trainer.definition = registry.get_definition("ideogram4-fp8")
    trainer.config = dict(config)
    trainer.driver = None
    trainer.global_step, trainer.max_train_steps = 0, 100
    return trainer


def _noise_fraction(trainer, t: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros(t.shape[0], 1)
    return trainer.add_noise(zeros, torch.ones_like(zeros), t).reshape(-1)


def test_ideogram4_flux_shift_leans_to_noise():
    trainer = _trainer(timestep_sampling="flux_shift")
    torch.manual_seed(0)
    t = trainer.sample_timesteps(8192)
    assert float(_noise_fraction(trainer, t).median()) > 0.5


def test_ideogram4_draw_is_the_shared_noise_fraction_on_its_clock():
    trainer = _trainer(timestep_sampling="radc", radc_start=0.9, radc_res_influence=0.0)
    torch.manual_seed(0)
    t = trainer.sample_timesteps(64)
    torch.manual_seed(0)
    sigma = TimestepSampler.sample_scaled("radc", 64, torch.device("cpu"), trainer.config)
    assert torch.allclose(t, 1000.0 - sigma)
    assert torch.allclose(_noise_fraction(trainer, t), sigma / 1000.0, atol=1e-6)


@pytest.mark.parametrize("mode", ["uniform", "logit_normal", "sigmoid"])
def test_symmetric_modes_keep_their_distribution(mode):
    trainer = _trainer(timestep_sampling=mode)
    torch.manual_seed(0)
    drawn = trainer.sample_timesteps(8192)
    torch.manual_seed(0)
    before = TimestepSampler.sample_scaled(mode, 8192, torch.device("cpu"), {"timestep_sampling": mode})
    q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    assert torch.allclose(torch.quantile(drawn, q), torch.quantile(before, q), atol=20.0)
