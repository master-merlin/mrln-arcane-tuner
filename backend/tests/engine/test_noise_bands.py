"""Which timestep means which noise level, measured through the family's own add_noise."""
import pytest
import torch

from app.engine.core.pipeline.noise_bands import band_centres, timesteps_for_bands


class _Family:
    """Only the hook the probe may use, plus a sampler it must not depend on."""

    family_name = "stub"
    validation_timestep_domain = (0.0, 1000.0)

    def __init__(self, fraction, domain=None):
        self._fraction = fraction
        if domain is not None or domain == ():
            self.validation_timestep_domain = domain

    def add_noise(self, latents, noise, timesteps):
        sigma = self._fraction(timesteps).reshape(-1, *[1] * (latents.dim() - 1))
        return (1 - sigma) * latents + sigma * noise

    def sample_timesteps(self, batch_size, latents=None):
        raise AssertionError("the probe must not depend on the configured sampler")


def _measure(family, timesteps):
    latents = torch.zeros(1, 1)
    noise = torch.ones(1, 1)
    return family.add_noise(latents, noise, timesteps.reshape(1)).item()


def _assert_on_centres(family, count=8):
    timesteps = timesteps_for_bands(family, count)
    assert timesteps.shape == (count,)
    for centre, timestep in zip(band_centres(count), timesteps):
        assert abs(_measure(family, timestep) - centre.item()) <= 0.01


def test_band_centres_have_equal_widths_in_noise():
    assert band_centres(4).tolist() == [0.125, 0.375, 0.625, 0.875]


def test_a_forward_clock_on_a_thousand():
    _assert_on_centres(_Family(lambda t: t / 1000.0))


def test_a_reversed_clock():
    _assert_on_centres(_Family(lambda t: 1.0 - t / 1000.0))


def test_a_unit_clock():
    _assert_on_centres(_Family(lambda t: t, domain=(0.0, 1.0)))


def test_a_nonlinear_shifted_schedule():
    def shifted(t):
        u = t / 1000.0
        return 3.0 * u / (1.0 + 2.0 * u)

    _assert_on_centres(_Family(shifted))


def test_bands_are_returned_in_noise_order_on_a_reversed_clock():
    family = _Family(lambda t: 1.0 - t / 1000.0)
    timesteps = timesteps_for_bands(family, 4)
    assert (timesteps[1:] < timesteps[:-1]).all()


def test_a_clock_that_is_not_monotonic_fails_naming_the_family():
    # Reaches both ends and every centre, so only the order of its values is wrong.
    family = _Family(lambda t: t / 1000.0 + 0.2 * torch.sin(t / 1000.0 * 4.0 * 3.14159265))
    with pytest.raises(ValueError, match="stub"):
        timesteps_for_bands(family, 4)


def test_a_clock_missing_the_clean_end_fails_naming_the_family():
    # Every centre of four bands is still reached.
    family = _Family(lambda t: 0.1 + 0.9 * t / 1000.0)
    with pytest.raises(ValueError, match="stub"):
        timesteps_for_bands(family, 4)


def test_a_clock_missing_the_noisy_end_fails_naming_the_family():
    # Every centre of four bands is still reached.
    family = _Family(lambda t: 0.9 * t / 1000.0)
    with pytest.raises(ValueError, match="stub"):
        timesteps_for_bands(family, 4)


def test_a_clock_that_jumps_over_a_band_fails_naming_the_family():
    def jumping(t):
        return torch.where(t < 500.0, t / 5000.0, 0.9 + (t - 500.0) / 5000.0)

    with pytest.raises(ValueError, match="stub"):
        timesteps_for_bands(_Family(jumping), 4)


def test_an_end_within_tolerance_is_accepted():
    _assert_on_centres(_Family(lambda t: 0.015 + 0.97 * t / 1000.0))


def test_a_family_without_a_domain_fails_naming_it():
    family = _Family(lambda t: t / 1000.0)
    family.validation_timestep_domain = None
    with pytest.raises(ValueError, match="unavailable.*stub"):
        timesteps_for_bands(family, 4)


def test_the_random_streams_are_untouched():
    family = _Family(lambda t: t / 1000.0)
    torch.manual_seed(1)
    expected = torch.rand(1)
    torch.manual_seed(1)
    timesteps_for_bands(family, 8)
    assert torch.equal(torch.rand(1), expected)


@pytest.mark.parametrize(
    "module, class_name, domain",
    [
        ("boogu_image", "BooguImageTrainer", (0.0, 1.0)),
        ("minimax_h3", "MiniMaxH3Trainer", (0.0, 1.0)),
        ("minimax_h3_image", "MiniMaxH3ImageTrainer", (0.0, 1.0)),
        ("omnigen2", "OmniGen2Trainer", (0.0, 1.0)),
        ("sdxl", "SDXLTrainer", None),
        ("hidream_o1", "HiDreamO1Trainer", None),
        ("wan22", "Wan22Trainer", None),
        ("bernini_r", "BerniniRTrainer", None),
    ],
)
def test_each_named_family_declares_its_domain(module, class_name, domain):
    import importlib

    trainer = importlib.import_module(f"app.engine.models.families.{module}.trainer")
    assert getattr(trainer, class_name).validation_timestep_domain == domain


def test_the_base_trainer_declares_the_thousand_clock():
    from app.engine.core.pipeline.pipeline_base import PipelineBaseMixin

    assert PipelineBaseMixin.validation_timestep_domain == (0.0, 1000.0)
