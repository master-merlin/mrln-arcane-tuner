"""FLUX.3's timestep draw: radc sets the noise level itself, every other mode keeps its fixed shift.

The values below were drawn on main before the change (seed 0, step 50 of 100, batch 4): a mode
other than radc must still return them, so the earlier measurements stay comparable.
"""

from __future__ import annotations

import pytest
import torch

from app.engine.models.registry import registry
from app.engine.strategies import timestep_sampling
from tests.test_engine_fields_reach_the_trainer import _trainer

# Pinned on main 54769ca (pin_main.py in the change's evidence).
MAIN_DRAWS: dict[str, list[float]] = {
    "uniform": [872.237183, 958.279846, 402.151367, 513.180969],
    "logit_normal": [970.022034, 837.865417, 439.564514, 924.442078],
    "sigmoid": [970.022034, 837.865417, 439.564514, 924.442078],
    "cosmap": [737.765259, 926.098755, 63.206314, 131.759415],
    "mode": [913.433533, 758.234985, 990.748718, 985.577637],
    "flux_shift": [872.237183, 958.279846, 402.151367, 513.180969],
    "model_shift": [872.237183, 958.279846, 402.151367, 513.180969],
}


def _flux3(**config):
    registry.initialize()
    trainer = _trainer(registry.get_definition("flux3-image-klein-base"), **config)
    trainer.global_step, trainer.max_train_steps = 50, 100
    return trainer


@pytest.mark.parametrize("mode", sorted(MAIN_DRAWS))
def test_flux3_other_modes_draw_as_before(mode):
    trainer = _flux3(timestep_sampling=mode)
    torch.manual_seed(0)
    drawn = trainer.sample_timesteps(4, torch.zeros(4, 32, 8, 8))
    assert torch.allclose(drawn, torch.tensor(MAIN_DRAWS[mode]), atol=1e-4), (mode, drawn.tolist())


def test_flux3_pins_every_mode_but_radc():
    assert len(MAIN_DRAWS) == 7 and "radc" not in MAIN_DRAWS


def test_flux3_radc_is_the_shared_draw_on_the_noise_scale(monkeypatch):
    trainer = _flux3(timestep_sampling="radc", radc_start=0.9, radc_end=0.3, radc_res_influence=0.0)
    torch.manual_seed(0)
    drawn = trainer.sample_timesteps(4, torch.zeros(4, 32, 8, 8))
    torch.manual_seed(0)
    shared = timestep_sampling.TimestepSampler.sample_scaled("radc", 4, torch.device("cpu"), trainer.config, progress=0.5)
    assert torch.equal(drawn, shared)


@pytest.mark.parametrize(("cells", "expected_q"), [(32, 700), (64, 660)])
def test_flux3_radc_reads_the_latents_and_its_patchify_factor(monkeypatch, cells, expected_q):
    # At progress 0.5 the centre is 0.6. 32x32 cells patchify 2x2 into 256 tokens (term +0.1);
    # 64x64 into 1024 tokens: res_norm (1024 - 256) / 3840 = 0.2, term 0.1 * (1 - 0.4) = +0.06.
    # Without the factor 64x64 would count 4096 cells and land on 0.5.
    asked = []
    real = timestep_sampling._radc_cdf
    monkeypatch.setattr(timestep_sampling, "_radc_cdf", lambda c, w: asked.append(c) or real(c, w))
    trainer = _flux3(timestep_sampling="radc", radc_start=0.9, radc_end=0.3, radc_res_influence=0.2)
    trainer.sample_timesteps(2, torch.zeros(2, 32, cells, cells))
    assert asked == [expected_q]
    assert "flux_shift_patchify_factor" not in trainer.config, "the job's config must not be rewritten"
