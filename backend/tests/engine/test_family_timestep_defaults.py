"""A definition's timestep defaults come from one place, for the form and for the job.

A definition states its family's default draw either in its ``defaults:`` block or as
``architecture_params`` ``timestep.*`` keys (``timestep.default_mode`` and one key per mode
parameter, e.g. ``timestep.logit_normal_mu``). ``family_timestep_defaults`` merges both into the
run-config keys; the descriptor the Training form reads (``resolve_capabilities``) carries all of
them, so an untouched form submits the family's values, not the schema's.
"""

from __future__ import annotations

import pytest

from app.engine.core.archetypes import resolve_capabilities
from app.engine.models.registry import registry
from app.engine.strategies.timestep_sampling import (
    TIMESTEP_PARAMETERS,
    apply_family_timestep_defaults,
    family_timestep_defaults,
)


class _Stub:
    id = "stub"
    family = "flux2"
    control_inputs = 0

    def __init__(self, architecture_params: dict, defaults: dict | None = None) -> None:
        self.architecture_params = architecture_params
        self.defaults = defaults or {}


@pytest.fixture(scope="module", autouse=True)
def _registry():
    registry.initialize()


@pytest.fixture(scope="module")
def moss_definitions():
    return [registry.get_definition("moss-image-base"), registry.get_definition("moss-image-sft")]


def test_the_moss_descriptor_carries_the_family_logit_normal_parameters(moss_definitions):
    for definition in moss_definitions:
        defaults = resolve_capabilities(definition)["defaults"]
        assert defaults["timestep_sampling"] == "logit_normal"
        assert defaults["logit_normal_mu"] == 1.09
        assert defaults["logit_normal_sigma"] == 0.95


def test_every_schema_mode_parameter_is_in_the_shared_set():
    from app.engine.models.base import BaseTrainingConfig

    fields = set(BaseTrainingConfig.model_fields)
    prefixes = ("logit_normal_", "flux_shift_", "radc_", "model_shift_")
    schema_params = {
        name for name in fields
        if name.startswith(prefixes) or name in ("mode_scale", "timestep_uniform_prob", "fixed_sigmas")
    }
    assert schema_params == set(TIMESTEP_PARAMETERS)


def test_the_schema_default_table_is_the_schemas():
    from app.engine.models.base import BaseTrainingConfig
    from app.engine.strategies.timestep_sampling import TIMESTEP_SCHEMA_DEFAULTS

    fields = BaseTrainingConfig.model_fields
    assert set(TIMESTEP_SCHEMA_DEFAULTS) == {"timestep_sampling", *TIMESTEP_PARAMETERS} - {"fixed_sigmas"}
    for key, value in TIMESTEP_SCHEMA_DEFAULTS.items():
        assert fields[key].default == value, key


def test_a_setting_is_the_configs_then_the_familys_then_the_schemas():
    from app.engine.strategies.timestep_sampling import timestep_setting

    stub = _Stub({"timestep.default_mode": "flux_shift", "timestep.flux_shift_max": 1.15})
    assert timestep_setting({"flux_shift_max": 2.0}, "flux_shift_max", stub) == 2.0
    assert timestep_setting({"flux_shift_max": None}, "flux_shift_max", stub) == 1.15
    assert timestep_setting({}, "timestep_sampling", stub) == "flux_shift"
    assert timestep_setting({}, "flux_shift_base", stub) == 0.5
    assert timestep_setting({}, "timestep_sampling") == "logit_normal"
    with pytest.raises(KeyError):
        timestep_setting({}, "learning_rate")


def test_a_radc_declaration_is_derived():
    stub = _Stub({"timestep.default_mode": "radc", "timestep.radc_start": 0.6, "timestep.radc_width": "0.2"})
    assert family_timestep_defaults(stub) == {"timestep_sampling": "radc", "radc_start": 0.6, "radc_width": 0.2}
    defaults = resolve_capabilities(stub)["defaults"]
    assert defaults["radc_start"] == 0.6
    assert defaults["radc_width"] == 0.2


def test_the_defaults_block_timestep_keys_are_part_of_the_family_default():
    stub = _Stub({}, {"timestep_sampling": "mode", "mode_scale": 1.29, "learning_rate": 1e-4})
    assert family_timestep_defaults(stub) == {"timestep_sampling": "mode", "mode_scale": 1.29}


def test_a_parameter_that_disagrees_with_the_defaults_block_is_refused():
    stub = _Stub({"timestep.default_mode": "logit_normal", "timestep.logit_normal_mu": 1.09},
                 {"logit_normal_mu": 0.0})
    with pytest.raises(ValueError, match="timestep.logit_normal_mu"):
        resolve_capabilities(stub)


def test_an_unknown_timestep_key_is_refused():
    with pytest.raises(ValueError, match="timestep.logit_normal_mean"):
        family_timestep_defaults(_Stub({"timestep.default_mode": "logit_normal", "timestep.logit_normal_mean": 1.0}))


def test_a_parameter_that_is_no_number_is_refused():
    with pytest.raises(ValueError, match="timestep.flux_shift_max"):
        family_timestep_defaults(_Stub({"timestep.default_mode": "flux_shift", "timestep.flux_shift_max": "high"}))


def test_a_definition_without_timestep_keys_has_no_family_default():
    assert family_timestep_defaults(_Stub({}, {"learning_rate": 1e-4})) == {}


# ── Applying it to a job config ──────────────────────────────────────────────


def _moss_stub():
    return _Stub({"timestep.default_mode": "logit_normal", "timestep.logit_normal_mu": 1.09,
                  "timestep.logit_normal_sigma": 0.95})


def test_a_config_without_a_mode_gets_the_whole_family_default():
    config: dict = {}
    apply_family_timestep_defaults(_moss_stub(), config)
    assert config == {"timestep_sampling": "logit_normal", "logit_normal_mu": 1.09, "logit_normal_sigma": 0.95}


def test_the_users_values_are_never_replaced():
    config = {"timestep_sampling": "uniform", "logit_normal_mu": 0.5}
    apply_family_timestep_defaults(_moss_stub(), config)
    assert config == {"timestep_sampling": "uniform", "logit_normal_mu": 0.5, "logit_normal_sigma": 0.95}


def test_a_legacy_alias_counts_as_the_users_value():
    config = {"timestep_sampling": "logit_normal", "logit_normal_mean": 0.3, "logit_normal_std": 0.7}
    apply_family_timestep_defaults(_moss_stub(), config)
    assert "logit_normal_mu" not in config
    assert "logit_normal_sigma" not in config


def _h3_stub():
    return _Stub({"timestep.default_mode": "fixed_sigmas", "timestep.fixed_sigmas": [1.0, 0.75, 0.25]})


def test_the_family_list_is_filled_only_for_the_fixed_sigmas_mode():
    config = {"timestep_sampling": "logit_normal"}
    apply_family_timestep_defaults(_h3_stub(), config)
    assert config == {"timestep_sampling": "logit_normal"}
    config = {"timestep_sampling": "fixed_sigmas", "fixed_sigmas": None}
    apply_family_timestep_defaults(_h3_stub(), config)
    assert config["fixed_sigmas"] == [1.0, 0.75, 0.25]
    config = {}
    apply_family_timestep_defaults(_h3_stub(), config)
    assert config == {"timestep_sampling": "fixed_sigmas", "fixed_sigmas": [1.0, 0.75, 0.25]}


def test_a_definition_without_a_default_leaves_the_config_alone():
    config = {"learning_rate": 1e-4}
    apply_family_timestep_defaults(_Stub({}), config)
    assert config == {"learning_rate": 1e-4}
