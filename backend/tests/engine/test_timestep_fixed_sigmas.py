"""The shared ``fixed_sigmas`` timestep mode: a uniform draw from a configured list of sigmas.

A model distilled to a fixed sampling schedule trains only on that schedule's levels (MiniMax-H3-Image
trains on its 6 sampling sigmas, as ostris/ai-toolkit does). The mode lives in the shared
``TimestepSampler``, the run-config schema and the archetype gating; a definition names it as its
default through ``timestep.default_mode`` / ``timestep.fixed_sigmas``, which ``resolve_capabilities``
turns into the form's defaults.
"""

from __future__ import annotations

import math
from typing import get_args

import pytest
import torch
from pydantic import ValidationError

from app.engine.core.archetypes import FLOW_TIMESTEP_FIELDS, resolve_capabilities
from app.engine.models.base import BaseTrainingConfig
from app.engine.models.registry import registry
from app.engine.strategies.timestep_sampling import TimestepSampler, validate_fixed_sigmas

CPU = torch.device("cpu")
# MiniMax-H3-Image's sampling sigmas without the terminal 0.0 (modular_model_index.json, sha fbde899f).
H3_SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132]


@pytest.fixture(scope="module", autouse=True)
def _registry():
    registry.initialize()
    return registry


# ── The draw ────────────────────────────────────────────────────────────────


def test_fixed_sigmas_is_a_supported_mode_of_the_sampler_and_the_schema():
    assert "fixed_sigmas" in TimestepSampler.SUPPORTED_MODES
    schema_modes = list(get_args(BaseTrainingConfig.model_fields["timestep_sampling"].annotation))
    assert "fixed_sigmas" in schema_modes
    assert set(schema_modes) == set(TimestepSampler.SUPPORTED_MODES)


def test_every_draw_is_a_value_of_the_list():
    t = TimestepSampler.sample("fixed_sigmas", 4096, CPU, {"fixed_sigmas": H3_SIGMAS})
    assert t.shape == (4096,)
    allowed = torch.tensor(H3_SIGMAS, dtype=t.dtype)
    assert bool(torch.isclose(t[:, None], allowed[None, :], rtol=0, atol=0).any(dim=1).all())


def test_the_draw_is_uniform_over_the_list_from_a_local_generator():
    n, p = 60000, 1.0 / 6.0
    bound = 3.0 * math.sqrt(n * p * (1.0 - p))
    global_state = torch.get_rng_state()
    gen = torch.Generator(device="cpu").manual_seed(0)
    t = TimestepSampler.sample("fixed_sigmas", n, CPU, {"fixed_sigmas": H3_SIGMAS}, generator=gen)
    assert torch.equal(torch.get_rng_state(), global_state), "the draw touched the global RNG"
    for value in H3_SIGMAS:
        count = int((t == torch.tensor(value, dtype=t.dtype)).sum())
        assert abs(count - n * p) <= bound, f"{value}: {count} draws, expected {n * p:.0f} +- {bound:.0f}"


def test_the_same_seed_gives_the_same_draws():
    a = TimestepSampler.sample("fixed_sigmas", 64, CPU, {"fixed_sigmas": H3_SIGMAS},
                               generator=torch.Generator().manual_seed(7))
    b = TimestepSampler.sample("fixed_sigmas", 64, CPU, {"fixed_sigmas": H3_SIGMAS},
                               generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_sample_scaled_puts_the_list_on_the_1000_grid():
    t = TimestepSampler.sample_scaled("fixed_sigmas", 512, CPU, {"fixed_sigmas": [0.75]})
    assert torch.allclose(t, torch.full((512,), 750.0))


@pytest.mark.parametrize(
    "bad",
    [None, [], [0.0], [0.5, 0.0], [1.0000001], [-0.25], [float("nan")], [0.5, 2.0], "0.5"],
)
def test_the_sampler_refuses_a_missing_or_empty_list_or_a_value_outside_0_1(bad):
    config = {} if bad is None else {"fixed_sigmas": bad}
    with pytest.raises(ValueError, match="fixed_sigmas"):
        TimestepSampler.sample("fixed_sigmas", 4, CPU, config)


def test_validate_fixed_sigmas_keeps_a_good_list_as_floats():
    assert validate_fixed_sigmas([1, 0.5, "0.25"]) == [1.0, 0.5, 0.25]
    assert validate_fixed_sigmas(H3_SIGMAS) == H3_SIGMAS


# ── Every other mode is unchanged ───────────────────────────────────────────

# Drawn on the code before ``fixed_sigmas`` existed: torch.manual_seed(1234), bs 4, a [1,4,64,64]
# latent, progress 0.3, the model_shift dynamic band 0.5..1.15.
_PINNED = {
    "uniform": [0.028979, 0.401899, 0.259844, 0.366641],
    "logit_normal": [0.511531, 0.599265, 0.266681, 0.553973],
    "sigmoid": [0.511531, 0.599265, 0.266681, 0.553973],
    "cosmap": [0.001036, 0.19274, 0.082148, 0.161308],
    "mode": [0.980492, 0.690621, 0.810344, 0.721541],
    "flux_shift": [0.769614, 0.826698, 0.53705, 0.798467],
    "radc": [0.132, 0.506, 0.411, 0.484],
    "model_shift": [0.767836, 0.825261, 0.534562, 0.796853],
}
_PIN_CFG = {"model_shift_base_shift": 0.5, "model_shift_max_shift": 1.15}


@pytest.mark.parametrize("mode", sorted(_PINNED))
def test_the_other_modes_draw_exactly_what_they_drew_before(mode):
    torch.manual_seed(1234)
    t = TimestepSampler.sample(mode, 4, CPU, dict(_PIN_CFG), latents=torch.zeros(1, 4, 64, 64), progress=0.3)
    torch.testing.assert_close(t, torch.tensor(_PINNED[mode]), rtol=0, atol=1e-6)


@pytest.mark.parametrize("mode", sorted(_PINNED))
def test_the_other_modes_ignore_a_fixed_sigmas_list(mode):
    torch.manual_seed(1234)
    t = TimestepSampler.sample(mode, 4, CPU, {**_PIN_CFG, "fixed_sigmas": [0.5]},
                               latents=torch.zeros(1, 4, 64, 64), progress=0.3)
    torch.testing.assert_close(t, torch.tensor(_PINNED[mode]), rtol=0, atol=1e-6)


@pytest.mark.parametrize("mode", sorted(_PINNED))
def test_every_mode_draws_from_a_given_generator_and_leaves_the_global_rng_alone(mode):
    lat = torch.zeros(1, 4, 64, 64)
    global_state = torch.get_rng_state()
    a = TimestepSampler.sample(mode, 8, CPU, dict(_PIN_CFG), latents=lat, progress=0.3,
                               generator=torch.Generator().manual_seed(3))
    assert torch.equal(torch.get_rng_state(), global_state), mode
    b = TimestepSampler.sample(mode, 8, CPU, dict(_PIN_CFG), latents=lat, progress=0.3,
                               generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b), mode


# ── The run-config schema and the archetype gate ────────────────────────────


def test_the_schema_field_holds_the_list_and_shows_only_for_its_mode():
    field = BaseTrainingConfig.model_fields["fixed_sigmas"]
    assert field.default is None
    extra = field.json_schema_extra
    assert extra["group"] == "STRATEGY"
    assert extra["depends_on"] == "timestep_sampling:fixed_sigmas"
    cfg = BaseTrainingConfig.model_validate(
        {"datasets": [{"dataset_name": "d"}], "timestep_sampling": "fixed_sigmas", "fixed_sigmas": H3_SIGMAS})
    assert cfg.timestep_sampling == "fixed_sigmas"
    assert cfg.fixed_sigmas == H3_SIGMAS
    assert BaseTrainingConfig.model_validate({"datasets": [{"dataset_name": "d"}]}).fixed_sigmas is None


@pytest.mark.parametrize("bad", [[0.0], [1.5], [-0.1], [0.5, float("nan")]])
def test_the_schema_refuses_a_value_outside_0_1(bad):
    with pytest.raises(ValidationError, match="fixed_sigmas"):
        BaseTrainingConfig.model_validate({"datasets": [{"dataset_name": "d"}], "fixed_sigmas": bad})


def test_the_list_is_a_flow_timestep_field_gated_with_the_mode():
    assert "fixed_sigmas" in FLOW_TIMESTEP_FIELDS
    sdxl = registry.get_definition("sdxl_base_1.0")
    vis = resolve_capabilities(sdxl)["field_visibility"]
    assert vis["fixed_sigmas"]["supported"] is False
    flux2 = next(d for d in registry._definitions.values() if d.family == "flux2")
    assert resolve_capabilities(flux2)["field_visibility"]["fixed_sigmas"] == {"supported": True}


# ── The form defaults derive from the definition's timestep.* keys ──────────


class _Stub:
    family = "flux2"
    control_inputs = 0

    def __init__(self, architecture_params: dict, defaults: dict | None = None) -> None:
        self.architecture_params = architecture_params
        self.defaults = defaults or {}


def test_the_defaults_carry_the_family_default_mode_and_its_list():
    stub = _Stub({"timestep.default_mode": "fixed_sigmas", "timestep.fixed_sigmas": H3_SIGMAS},
                 {"learning_rate": "1e-4"})
    defaults = resolve_capabilities(stub)["defaults"]
    assert defaults["timestep_sampling"] == "fixed_sigmas"
    assert defaults["fixed_sigmas"] == H3_SIGMAS
    assert defaults["learning_rate"] == 1e-4


def test_a_definition_without_timestep_keys_gets_no_derived_defaults():
    defaults = resolve_capabilities(_Stub({}, {"learning_rate": 1e-4}))["defaults"]
    assert "timestep_sampling" not in defaults
    assert "fixed_sigmas" not in defaults


def test_a_defaults_block_that_disagrees_with_the_timestep_keys_is_refused():
    stub = _Stub({"timestep.default_mode": "fixed_sigmas", "timestep.fixed_sigmas": H3_SIGMAS},
                 {"timestep_sampling": "logit_normal"})
    with pytest.raises(ValueError, match="timestep.default_mode"):
        resolve_capabilities(stub)
    stub = _Stub({"timestep.default_mode": "fixed_sigmas", "timestep.fixed_sigmas": H3_SIGMAS},
                 {"fixed_sigmas": [0.5]})
    with pytest.raises(ValueError, match="timestep.fixed_sigmas"):
        resolve_capabilities(stub)


def test_a_fixed_sigmas_default_without_a_valid_list_is_refused():
    with pytest.raises(ValueError, match="fixed_sigmas"):
        resolve_capabilities(_Stub({"timestep.default_mode": "fixed_sigmas"}))
    with pytest.raises(ValueError, match="fixed_sigmas"):
        resolve_capabilities(_Stub({"timestep.default_mode": "fixed_sigmas", "timestep.fixed_sigmas": [0.0]}))


def test_an_unknown_default_mode_is_refused():
    with pytest.raises(ValueError, match="timestep.default_mode"):
        resolve_capabilities(_Stub({"timestep.default_mode": "warp_drive"}))
