"""The noise-coordinate map: which noise levels a family's timestep sampler trains, on its own scale.

CPU only, no weights, no network: every row is drawn by the family's real trainer class, built
without weights, through its own ``sample_timesteps``.
"""
import pytest
import torch

from app.engine.core.noise_map import (
    SamplerSpec,
    capabilities,
    latent_downscale,
    latents_for,
    map_definition,
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


# ── Task 3: one definition, sampler and resolution ──

RADC_FIXED = "radc:radc_start=0.5,radc_end=0.5,radc_width=0.2,radc_res_influence=0"
RADC_CURRICULUM = "radc:radc_start=0.9,radc_end=0.3,radc_width=0.5,radc_res_influence=0.15"
NO_DOWNSCALE = "this sampler depends on resolution and the definition declares no latent downscale: pass --latent-downscale"
NO_FLOW = "declares no flow-matching timestep sampling: only its own draw (default) applies"


def _row(definition_id, sampler="default", resolution=1024, **options):
    settings = {"steps": 2000, "draws_per_step": 8, "bands": 4, "seed": 0, **options}
    return map_definition(_definition(definition_id), parse_sampler(sampler), resolution, **settings)


def _assert_shares(row, expected, tolerance=0.015):
    assert row["status"] == "mapped", row["reason"]
    shares = row["all"]["shares"]
    assert len(shares) == len(expected)
    assert all(abs(a - b) <= tolerance for a, b in zip(shares, expected)), f"{shares} != {expected}"


def _trainer_class(definition_id):
    return type(sampling_shell(_definition(definition_id), {}, 10))


class _CallSpy:
    """Counts the calls of a class method and still runs it."""

    def __init__(self, monkeypatch, cls, name):
        self.calls = []
        real = getattr(cls, name)

        def spy(this, *args, **kwargs):
            self.calls.append(args)
            return real(this, *args, **kwargs)

        monkeypatch.setattr(cls, name, spy)


def test_flux3_uniform_carries_its_fixed_shift():
    _assert_shares(_row("flux3-image-klein-base", "uniform"), [0.0459, 0.0802, 0.1760, 0.6979])


def test_moss_uniform_is_flat():
    _assert_shares(_row("moss-image-base", "uniform"), [0.25, 0.25, 0.25, 0.25])


def test_moss_default_is_its_own_logit_normal():
    row = _row("moss-image-base")
    _assert_shares(row, [0.0106, 0.1151, 0.3779, 0.4964])
    config = row["sampler_config"]
    assert config["timestep_sampling"] == "logit_normal"
    assert config["logit_normal_mu"] == 1.09
    assert config["logit_normal_sigma"] == 0.95


@pytest.mark.parametrize("definition_id", ["flux3-image-klein-base", "moss-image-base"])
def test_radc_lands_on_the_same_bands_in_both_families(definition_id):
    _assert_shares(_row(definition_id, RADC_FIXED), [0.0062, 0.4938, 0.4938, 0.0062])


def test_radc_moves_from_high_to_low_noise_over_the_phases():
    row = _row("moss-image-base", RADC_CURRICULUM)
    assert row["status"] == "mapped", row["reason"]
    assert [phase["steps"] for phase in row["phases"]] == [[0, 500], [500, 1000], [1000, 1500], [1500, 2000]]
    medians = [phase["median"] for phase in row["phases"]]
    assert all(b < a for a, b in zip(medians, medians[1:])), medians


def test_moss_radc_late_focus_moves_with_resolution():
    rows = {res: _row("moss-image-base", RADC_CURRICULUM, res) for res in (512, 768, 1024)}
    assert [rows[res]["latent_grid"] for res in (512, 768, 1024)] == [[64, 64], [96, 96], [128, 128]]
    late = {res: row["phases"][-1]["median"] for res, row in rows.items()}
    assert late[1024] < late[768] < late[512], late
    without = "radc:radc_start=0.9,radc_end=0.3,radc_width=0.5,radc_res_influence=0"
    assert _row("moss-image-base", without, 512)["all"] == _row("moss-image-base", without, 1024)["all"]


def test_every_phase_sums_to_one():
    row = _row("moss-image-base", RADC_CURRICULUM)
    for part in [*row["phases"], row["all"]]:
        assert abs(sum(part["shares"]) - 1.0) <= 1e-9


def test_phase_bounds_with_a_remainder():
    row = _row("moss-image-base", "uniform", steps=10, phases=4)
    assert [phase["steps"] for phase in row["phases"]] == [[0, 2], [2, 5], [5, 7], [7, 10]]


def test_more_phases_than_steps_is_refused():
    with pytest.raises(ValueError):
        _row("moss-image-base", "uniform", steps=3, phases=4)


def test_the_same_seed_gives_the_same_row_and_another_seed_another():
    first = _row("moss-image-base", "uniform", steps=50)
    assert _row("moss-image-base", "uniform", steps=50) == first
    assert _row("moss-image-base", "uniform", steps=50, seed=1)["all"] != first["all"]


def _three_rows(monkeypatch):
    """A mapped row, a two-grid comparison, and a row whose shell construction raises."""
    _row("moss-image-base", "uniform", steps=50)
    _row("wan2.1-t2v-14b", "uniform", steps=50)
    cls = _trainer_class("moss-image-base")

    def broken(self):
        torch.rand(3)  # consumes the stream before it raises
        raise RuntimeError("broken setup")

    with monkeypatch.context() as patched:
        patched.setattr(cls, "_setup_family", broken)
        row = _row("moss-image-base", "uniform", steps=50)
    assert row["status"] == "not_mapped" and row["reason"] == "RuntimeError: broken setup"


def test_the_callers_random_stream_is_untouched(monkeypatch):
    before = torch.default_generator.get_state()
    _three_rows(monkeypatch)
    assert torch.equal(torch.default_generator.get_state(), before)


def test_no_cuda_generator_is_reseeded(monkeypatch):
    called = []
    for owner, name in ((torch, "manual_seed"), (torch.cuda, "manual_seed"), (torch.cuda, "manual_seed_all")):
        monkeypatch.setattr(owner, name, lambda *a, _n=f"{owner.__name__}.{name}", **k: called.append(_n))
    _three_rows(monkeypatch)
    assert called == []


def test_a_row_records_what_sets_its_scale():
    keys = {"definition", "family", "sampler", "sampler_config", "resolution", "latent_grid",
            "latent_grid_source", "latent_frames", "latent_shape", "band_edges", "scheduler_error",
            "steps", "draws_per_step", "seed", "status", "reason", "phases", "all"}
    moss = _row("moss-image-base", "uniform", steps=20)
    assert keys <= set(moss)
    assert moss["definition"] == "moss-image-base" and moss["family"] == "moss_image"
    assert moss["latent_grid_source"] == "vae.block_out_channels"
    assert moss["band_edges"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert moss["latent_frames"] == 1
    assert moss["latent_shape"] == [8, 32, 128, 128]
    wan = _row("wan2.1-t2v-14b", RADC_CURRICULUM, steps=20, latent_downscale=8, latent_frames=3)
    assert wan["status"] == "mapped", wan["reason"]
    assert wan["latent_frames"] == 3
    assert wan["latent_shape"][2:] == [3, 128, 128]
    assert wan["latent_grid_source"] == "--latent-downscale"


def test_a_resolution_term_without_a_declared_downscale_is_not_mapped():
    wan = _row("wan2.1-t2v-14b", RADC_CURRICULUM, steps=20)
    assert wan["status"] == "not_mapped" and "latent downscale" in wan["reason"]
    given = _row("wan2.1-t2v-14b", RADC_CURRICULUM, steps=20, latent_downscale=8)
    assert given["status"] == "mapped", given["reason"]
    assert given["latent_grid"] == [128, 128]
    for definition_id in ("flux1-schnell", "flux2-klein-base-4b"):
        row = _row(definition_id, RADC_CURRICULUM, steps=20)
        assert row["status"] == "not_mapped", definition_id
        assert row["latent_grid"] != [1024, 1024]


def test_a_late_resolution_term_is_refused_whatever_the_first_steps_draw(monkeypatch):
    spy = _CallSpy(monkeypatch, _trainer_class("wan2.1-t2v-14b"), "sample_timesteps")
    row = _row("wan2.1-t2v-14b", RADC_CURRICULUM, steps=200000)
    assert row["status"] == "not_mapped" and row["reason"] == NO_DOWNSCALE
    assert spy.calls == []


def test_a_resolution_dependent_mode_without_a_downscale_is_not_mapped():
    row = _row("wan2.1-t2v-14b", "flux_shift", steps=50)
    assert row["status"] == "not_mapped" and row["reason"] == NO_DOWNSCALE


def test_a_resolution_free_sampler_needs_no_downscale(monkeypatch):
    spy = _CallSpy(monkeypatch, _trainer_class("wan2.1-t2v-14b"), "sample_timesteps")
    row = _row("wan2.1-t2v-14b", "uniform", steps=50)
    assert row["status"] == "mapped", row["reason"]
    assert row["latent_grid"] is None and row["latent_shape"] is None
    assert len(spy.calls) == 2 * 50


def test_radc_without_a_resolution_term_needs_no_downscale():
    row = _row("ace-step-1.5", RADC_CURRICULUM, steps=50)
    assert row["status"] == "mapped", row["reason"]


def test_a_scheduler_that_raises_reaches_the_reason_when_it_is_read(monkeypatch):
    definition = _definition("sdxl_base_1.0")

    def raising(self):
        raise RuntimeError("no scheduler")

    monkeypatch.setattr(_driver_class(definition), "init_scheduler", raising)
    row = _row("sdxl_base_1.0", steps=20)
    assert row["status"] == "not_mapped"
    assert row["reason"] == "RuntimeError: no scheduler"


def test_a_scheduler_that_raises_unread_is_shown_on_a_mapped_row():
    row = _row("boogu-image-base", steps=20)
    assert row["status"] == "mapped", row["reason"]
    assert row["scheduler_error"].startswith("RuntimeError:")


def test_a_family_without_flow_sampling_maps_only_its_own_draw():
    own = _row("sdxl_base_1.0", steps=50)
    assert own["status"] == "mapped", own["reason"]
    assert abs(sum(own["all"]["shares"]) - 1.0) <= 1e-9
    other = _row("sdxl_base_1.0", "uniform", steps=50)
    assert other["status"] == "not_mapped" and other["reason"] == NO_FLOW


def test_a_draw_that_raises_is_reported_by_name(monkeypatch):
    def boom(self, batch_size, latents=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(_trainer_class("moss-image-base"), "sample_timesteps", boom)
    row = _row("moss-image-base", "uniform", steps=20)
    assert row["status"] == "not_mapped" and row["reason"] == "RuntimeError: boom"


def test_the_optimizer_step_hook_runs_before_each_steps_draw(monkeypatch):
    definition_id = next(i for i in registry.list_models()
                         if capabilities(_definition(i)).get("dual_expert"))
    driver_class = _driver_class(_definition(definition_id))
    spy = _CallSpy(monkeypatch, driver_class, "on_optimizer_step")
    row = _row(definition_id, steps=8, phases=1, latent_downscale=8)  # one run, not the two-grid pair
    assert row["status"] == "mapped", row["reason"]
    assert [args[0] for args in spy.calls] == [-1, 0, 1, 2, 3, 4, 5, 6]
