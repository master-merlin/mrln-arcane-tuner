"""The noise-coordinate map: which noise levels a family's timestep sampler trains, on its own scale.

CPU only, no weights, no network: every row is drawn by the family's real trainer class, built
without weights, through its own ``sample_timesteps``.
"""
import pytest

from app.engine.core.noise_map import (
    SamplerSpec,
    latent_downscale,
    latents_for,
    parse_sampler,
    sampling_shell,
)
from app.engine.models.registry import registry


@pytest.fixture(scope="module", autouse=True)
def _registry():
    registry.initialize()
    return registry


def _definition(definition_id):
    definition = registry.get_definition(definition_id)
    assert definition is not None, f"definition {definition_id} is not in the registry"
    return definition


def _driver_class(definition):
    return type(sampling_shell(definition, {}, 10).driver)


# ── Task 2: the weightless shell, the latent grid, the sampler spec ──


def test_the_shell_is_the_familys_own_trainer_class():
    definition = _definition("moss-image-base")
    family = object.__new__(registry.get_family_class(definition.family))
    family.definition = definition
    family.config = {}
    expected = family.get_trainer_class()
    shell = sampling_shell(definition, {}, 10)
    assert type(shell) is expected
    assert type(shell).__name__ == "MossImageTrainer"


def test_a_scheduler_that_returns_none_stays_none():
    shell = sampling_shell(_definition("moss-image-base"), {}, 10)
    assert shell.scheduler is None
    assert shell.scheduler_error is None


def test_a_scheduler_that_raises_is_kept_as_its_error(monkeypatch):
    definition = _definition(registry.list_models()[0])
    driver_class = _driver_class(definition)

    def raising(self):
        raise RuntimeError("no scheduler")

    monkeypatch.setattr(driver_class, "init_scheduler", raising)
    shell = sampling_shell(definition, {}, 10)
    assert shell.scheduler_error == "RuntimeError: no scheduler"
    with pytest.raises(RuntimeError, match="no scheduler"):
        shell.scheduler.alphas_cumprod  # noqa: B018 -- the read itself must raise


def test_the_shell_carries_the_family_default():
    moss = sampling_shell(_definition("moss-image-base"), {}, 10)
    assert moss.config["timestep_sampling"] == "logit_normal"
    assert moss.config["logit_normal_mu"] == 1.09
    assert moss.config["logit_normal_sigma"] == 0.95
    flux3 = sampling_shell(_definition("flux3-image-klein-base"), {}, 10)
    assert flux3.config["timestep_sampling"] == "uniform"


def test_a_named_sampler_is_never_replaced_by_the_default():
    shell = sampling_shell(_definition("moss-image-base"), {"timestep_sampling": "uniform"}, 10)
    assert shell.config["timestep_sampling"] == "uniform"


@pytest.mark.parametrize(
    "definition_id, expected",
    [
        ("moss-image-base", (8, "vae.block_out_channels")),
        ("ltx2-5-base", (32, "vae.spatial_compression_ratio")),
        ("qwen-image-2.1", (16, "vae.scale_factor_spatial")),
        ("wan2.1-t2v-14b", None),
    ],
)
def test_latent_downscale_reads_the_declared_key(definition_id, expected):
    assert latent_downscale(_definition(definition_id)) == expected


@pytest.mark.parametrize("definition_id", ["flux1-schnell", "flux2-klein-base-4b"])
def test_missing_vae_keys_are_an_unknown_downscale_not_pixel_space(definition_id):
    assert latent_downscale(_definition(definition_id)) is None


@pytest.mark.parametrize("definition_id", ["prx-pixel-t2i", "hidream_o1_image"])
def test_only_a_model_without_a_vae_is_pixel_space(definition_id):
    assert latent_downscale(_definition(definition_id)) == (1, "has_vae false: pixel space")


def test_latents_have_the_familys_rank():
    audio = latents_for(_definition("ace-step-1.5"), cells=16, frames=5, batch=2)
    assert audio.ndim == 3 and audio.shape[0] == 2 and audio.shape[2] == 5
    video = latents_for(_definition("wan2.1-t2v-14b"), cells=16, frames=3, batch=2)
    assert video.ndim == 5 and video.shape[2] == 3 and tuple(video.shape[3:]) == (16, 16)
    image = latents_for(_definition("moss-image-base"), cells=16, frames=1, batch=2)
    assert tuple(image.shape) == (2, 32, 16, 16)


def test_parse_sampler_reads_mode_and_settings():
    spec = parse_sampler("radc:radc_start=0.9,radc_end=0.3,radc_width=0.5,radc_res_influence=0.15")
    assert isinstance(spec, SamplerSpec)
    assert spec.config == {
        "timestep_sampling": "radc",
        "radc_start": 0.9,
        "radc_end": 0.3,
        "radc_width": 0.5,
        "radc_res_influence": 0.15,
    }
    assert parse_sampler("default").config == {}
    assert parse_sampler("default").label == "default"


@pytest.mark.parametrize("text, bad", [("gaussian", "gaussian"), ("radc:centre=1", "centre"),
                                       ("uniform:radc_start=x", "x")])
def test_parse_sampler_refuses_an_unknown_mode_or_key(text, bad):
    with pytest.raises(ValueError, match=bad):
        parse_sampler(text)
