"""The sigma statistics record each timestep's noise fraction on the family's own clock.

A family whose ``t`` counts the clean fraction records ``1 - t / scale``, the scale being the upper
end of its declared ``validation_timestep_domain``: ideogram4's clock runs to 1000, boogu_image's,
omnigen2's and the two H3 families' to 1. Every other family records ``t / 1000`` as before.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from app.engine.strategies.sigma_tracker import SigmaTracker


def _bare(module: str, class_name: str):
    cls = getattr(importlib.import_module(f"app.engine.models.families.{module}.trainer"), class_name)
    return object.__new__(cls)


def _recorded(trainer, timesteps: torch.Tensor) -> list[float]:
    return [round(float(v), 6) for v in trainer._sigma_fraction(timesteps)]


def test_ideogram4_timestep_700_is_recorded_as_noise_fraction_0_3():
    trainer = _bare("ideogram4", "IdeogramV4Trainer")
    assert _recorded(trainer, torch.tensor([700.0, 0.0, 1000.0])) == [0.3, 1.0, 0.0]


def test_boogu_timestep_0_7_is_recorded_as_noise_fraction_0_3():
    trainer = _bare("boogu_image", "BooguImageTrainer")
    assert _recorded(trainer, torch.tensor([0.7])) == [0.3]


@pytest.mark.parametrize(
    ("module", "class_name"),
    [
        ("boogu_image", "BooguImageTrainer"),
        ("omnigen2", "OmniGen2Trainer"),
        ("minimax_h3", "MiniMaxH3Trainer"),
        ("minimax_h3_image", "MiniMaxH3ImageTrainer"),
    ],
)
def test_clean_fraction_families_record_as_before(module, class_name):
    trainer = _bare(module, class_name)
    t = torch.tensor([0.0, 0.25, 0.9, 1.0])
    assert torch.allclose(trainer._sigma_fraction(t), 1.0 - t)


def test_a_family_whose_add_noise_never_runs_records_its_own_noise_fraction():
    trainer = _bare("hidream_o1", "HiDreamO1Trainer")
    sigma = torch.tensor([0.05, 0.5, 0.9])
    assert torch.allclose(trainer._sigma_fraction(sigma), sigma)
    # a family without a domain and without a noise_fraction keeps the shared clock
    assert torch.allclose(_bare("bernini_r", "BerniniRTrainer")._sigma_fraction(torch.tensor([500.0])),
                          torch.tensor([0.5]))


def test_a_thousand_clock_family_records_as_before():
    trainer = _bare("flux2", "Flux2Trainer")
    t = torch.tensor([0.0, 250.0, 900.0, 1000.0])
    assert torch.allclose(trainer._sigma_fraction(t), t / 1000.0)


def test_the_training_step_records_through_the_family_clock():
    trainer = _bare("ideogram4", "IdeogramV4Trainer")
    trainer._sigma_tracker = SigmaTracker()
    trainer.config = {}
    trainer.distillation = None
    latents = torch.zeros(4, 2)
    trainer.make_training_noise = lambda lat: torch.ones_like(lat)
    trainer.prepare_latents_for_training = lambda lat: lat
    trainer.prepare_noise_for_training = lambda noise: noise
    trainer.sample_timesteps = lambda bs, lat=None: torch.full((bs,), 700.0)
    trainer.forward_pass = lambda noisy, t, text, batch: noisy
    trainer._compute_step_loss = lambda pred, target, t, batch, accum: torch.zeros(())
    trainer.forward_loss_step(latents, None, {}, 1)
    summary = trainer._sigma_tracker.summary()
    assert summary["n"] == 4 and summary["mean"] == pytest.approx(0.3)
