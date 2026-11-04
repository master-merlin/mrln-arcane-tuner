"""FLUX.2's radc loss weight and its draw use one centre per batch.

``compute_loss_weight`` keeps its one-argument signature, so it cannot see the
batch's latents. It used to compute the centre with no latents at all, which
dropped the ``radc_res_influence`` term the draw applies, and rebuilt its 1000-bin
curve on every call. The draw now records its centre for the weight to read.

The centre "the draw used" is taken from the arguments the real draw passes to
``_radc_cdf``, not recomputed here, so a test can't agree with a wrong formula.
"""

from __future__ import annotations

import pytest
import torch

from app.engine.core.pipeline.pipeline_base import PipelineBaseMixin
from app.engine.models.families.flux2.driver import Flux2Driver
from app.engine.models.families.flux2.trainer import Flux2Trainer
from app.engine.strategies import timestep_sampling
from app.engine.strategies.timestep_sampling import _make_radc_pdf, _radc_center

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


def _peak_q(trainer: Flux2Trainer) -> int:
    """The timestep (1..1000) at which the weight curve is highest."""
    return int(trainer.compute_loss_weight(ALL_TIMESTEPS).argmax()) + 1


@pytest.mark.parametrize(
    ("step", "latents", "expected_q"),
    [
        (100, LATENTS_4096_TOKENS, 50),  # 0.2 - 0.15
        (100, LATENTS_256_TOKENS, 350),  # 0.2 + 0.15
        (50, LATENTS_4096_TOKENS, 425),  # 0.5 - 0.075
        (50, LATENTS_256_TOKENS, 575),  # 0.5 + 0.075
    ],
)
def test_weight_peaks_at_the_centre_the_draw_used(drawn_curves, step, latents, expected_q):
    t = _trainer(step=step)

    t.sample_timesteps(2, latents)

    assert drawn_curves == [(expected_q, 500)]
    assert _peak_q(t) == expected_q


def test_each_batch_gets_its_own_centre(drawn_curves):
    t = _trainer()
    peaks = []
    for latents in (LATENTS_4096_TOKENS, LATENTS_256_TOKENS, LATENTS_4096_TOKENS):
        t.sample_timesteps(2, latents)
        peaks.append(_peak_q(t))

    assert peaks == [q for q, _ in drawn_curves]
    assert peaks == [50, 350, 50]


def test_the_drawn_timesteps_are_weighted_from_that_centre():
    t = _trainer()

    timesteps = t.sample_timesteps(8, LATENTS_4096_TOKENS)
    weights = t.compute_loss_weight(timesteps)

    curve = _make_radc_pdf(0.05, 0.5)
    expected = curve[timesteps.long().clamp(1, 1000) - 1]
    assert torch.equal(weights, expected)


@pytest.mark.parametrize("latents", [None, LATENTS_256_TOKENS, LATENTS_4096_TOKENS])
def test_with_influence_zero_weights_equal_the_draws_quantized_curve(drawn_curves, latents):
    # Step 1 of 3000 puts the raw centre at 0.7998, off the 0.001 grid.
    t = _trainer(step=1, max_steps=3000, radc_res_influence=0.0)

    t.sample_timesteps(2, latents)
    weights = t.compute_loss_weight(ALL_TIMESTEPS)

    assert drawn_curves == [(800, 500)]
    assert torch.equal(weights, _make_radc_pdf(0.8, 0.5))
    cdf = timestep_sampling._radc_cdf(800, 500)
    from_cdf = torch.diff(cdf, prepend=torch.zeros(1))
    assert torch.allclose(weights / weights.sum(), from_cdf, atol=1e-6)


@pytest.mark.parametrize("mode", ["logit_normal", "uniform", "sigmoid", "flux_shift"])
def test_other_modes_have_no_weight_and_record_nothing(mode):
    t = _trainer(timestep_sampling=mode)

    t.sample_timesteps(2, LATENTS_4096_TOKENS)

    assert t.compute_loss_weight(ALL_TIMESTEPS) is None
    assert t._radc_draw_center is None


@pytest.mark.parametrize("mode", ["radc", "logit_normal"])
def test_recording_does_not_change_the_draw_or_move_the_stream(mode):
    t = _trainer(step=50, timestep_sampling=mode)

    torch.manual_seed(7)
    expected = PipelineBaseMixin.sample_timesteps(t, 4, LATENTS_4096_TOKENS)
    after_expected = torch.rand(1)

    torch.manual_seed(7)
    drawn = t.sample_timesteps(4, LATENTS_4096_TOKENS)

    assert torch.equal(drawn, expected)
    assert torch.equal(torch.rand(1), after_expected)


def test_the_curve_is_built_once_not_on_every_call(monkeypatch):
    from app.engine.strategies.timestep_sampling import _radc_pdf

    _radc_pdf.cache_clear()
    built: list[tuple[float, float]] = []
    real = timestep_sampling._make_radc_pdf

    def spy(center, width):
        built.append((center, width))
        return real(center, width)

    monkeypatch.setattr(timestep_sampling, "_make_radc_pdf", spy)
    t = _trainer()
    t.compute_loss_weight(ALL_TIMESTEPS)
    t.compute_loss_weight(ALL_TIMESTEPS)

    assert built == [(0.2, 0.5)]
    assert _radc_pdf(200, 500) is _radc_pdf(200, 500)


def test_a_returned_weight_does_not_alias_the_shared_curve():
    t = _trainer()
    first = t.compute_loss_weight(ALL_TIMESTEPS)
    before = first.clone()

    first.mul_(0.0)

    assert torch.equal(t.compute_loss_weight(ALL_TIMESTEPS), before)


def test_a_direct_call_before_any_draw_uses_the_centre_without_latents():
    t = _trainer(step=50)

    assert t._radc_draw_center is None
    expected = round(_radc_center(t.config, 0.5, None) * 1000)
    assert expected == 500
    assert _peak_q(t) == expected
