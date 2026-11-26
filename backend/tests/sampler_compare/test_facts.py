"""Each arm's effective sampler and the trainer's loopback port, resolved the product's own way."""
import json
import os
import subprocess
import sys
import time

import psutil
import pytest
import torch

from app.core.sampler_compare import facts
from app.engine.models.registry import registry
from app.engine.strategies import timestep_keys
from app.engine.strategies.timestep_sampling import (
    _PARAMETER_ALIASES,
    TIMESTEP_PARAMETERS,
    TIMESTEP_SCHEMA_DEFAULTS,
    TimestepSampler,
    apply_family_timestep_defaults,
    timestep_setting,
    validate_fixed_sigmas,
)


@pytest.fixture(scope="module", autouse=True)
def _registry():
    registry.initialize()


def _arm(definition_id, arm):
    return facts.facts_here(definition_id, {"a": arm})["arms"]["a"]


@pytest.mark.parametrize("definition_id", ["sdxl_base_1.0", "moss-image-base"])
def test_the_hub_paths_are_every_component_the_loader_fetches(definition_id):
    definition = registry.get_definition(definition_id)
    wanted = sorted({c.path for c in definition.components.values()
                     if isinstance(getattr(c, "path", None), str) and c.path.startswith("huggingface:")})
    assert wanted  # both definitions load from the hub
    assert facts.facts_here(definition_id, {"a": {}})["hub_paths"] == wanted


def test_the_download_child_names_a_definition_it_cannot_fetch(tmp_path):
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "MRLN_SETTINGS_PATH": str(tmp_path / "settings.json"),
           "MRLN_DB_PATH": str(tmp_path / "db" / "x.db"), "MRLN_SERVER_LOG_PATH": str(tmp_path / "server.log"),
           "PYTHONDONTWRITEBYTECODE": "1"}
    with pytest.raises(RuntimeError, match="(?s)the weight download exited with .*no-such-model"):
        facts.prefetch_in_child("no-such-model", env, sys.executable)


def test_the_timestep_keys_are_the_products_own():
    assert timestep_keys.TIMESTEP_KEYS == {"timestep_sampling", *TIMESTEP_PARAMETERS, "logit_normal_mean",
                                           "logit_normal_std"}
    assert timestep_keys.PARAMETER_ALIASES is _PARAMETER_ALIASES
    assert timestep_keys.TIMESTEP_SCHEMA_DEFAULTS is TIMESTEP_SCHEMA_DEFAULTS


@pytest.mark.parametrize("key", sorted(TIMESTEP_SCHEMA_DEFAULTS))
def test_the_canonical_reading_of_each_key_is_the_samplers(key):
    value = "uniform" if key == "timestep_sampling" else "0.25"
    read = timestep_setting({key: value}, key)
    assert facts.canonical_sampler({key: value}) == {key: read if key == "timestep_sampling" else float(read)}


def test_the_canonical_reading_of_fixed_sigmas_is_the_samplers():
    given = [0.5, "0.25"]
    assert facts.canonical_sampler({"fixed_sigmas": given}) == {"fixed_sigmas": validate_fixed_sigmas(given)}


@pytest.mark.parametrize(("key", "alias"), [(k, a) for k, names in _PARAMETER_ALIASES.items() for a in names])
def test_an_older_name_counts_as_set_the_way_the_sampler_counts_it(key, alias):
    config = {"timestep_sampling": "logit_normal", alias: 0.25}
    assert facts.canonical_sampler(config)[key] == 0.25
    assert key not in apply_family_timestep_defaults(registry.get_definition("moss-image-base"), dict(config))


def test_the_family_default_arm_is_the_definitions_own_draw():
    arm = _arm("moss-image-base", {})
    effective, source = arm["effective"], arm["source"]
    assert effective["timestep_sampling"] == "logit_normal"
    assert effective["logit_normal_mu"] == 1.09
    assert effective["logit_normal_sigma"] == 0.95
    family = apply_family_timestep_defaults(registry.get_definition("moss-image-base"), {})
    assert {k: effective[k] for k in family} == family
    assert all(source[k] == "family" for k in family)


def test_a_definition_that_declares_nothing_gets_the_schema_defaults():
    arm = _arm("sdxl_base_1.0", {})
    assert arm["effective"] == TIMESTEP_SCHEMA_DEFAULTS
    assert arm["effective"]["logit_normal_mu"] == 0.0 and arm["effective"]["logit_normal_sigma"] == 1.0
    assert set(arm["source"].values()) == {"schema"}


def test_an_arm_overrides_only_the_keys_it_sets():
    arm = _arm("moss-image-base", {"timestep_sampling": "logit_normal"})
    assert arm["effective"]["logit_normal_mu"] == 1.09
    assert arm["source"]["logit_normal_mu"] == "family"
    assert arm["source"]["timestep_sampling"] == "arm"
    arm = _arm("moss-image-base", {"timestep_sampling": "logit_normal", "logit_normal_mu": 0.0})
    assert arm["effective"]["logit_normal_mu"] == 0.0
    assert arm["source"]["logit_normal_mu"] == "arm"


def test_fixed_sigmas_are_filled_only_in_their_own_mode():
    definition = registry.get_definition("minimax-h3-image-t2i")
    own = apply_family_timestep_defaults(definition, {})["fixed_sigmas"]
    assert _arm("minimax-h3-image-t2i", {})["effective"]["fixed_sigmas"] == own
    assert "fixed_sigmas" not in _arm("minimax-h3-image-t2i", {"timestep_sampling": "uniform"})["effective"]


def test_a_logged_key_that_differs_is_named():
    effective = {**TIMESTEP_SCHEMA_DEFAULTS, "timestep_sampling": "logit_normal"}
    logged = {"timestep_sampling": "logit_normal", "logit_normal_mu": 1.09}
    assert facts.sampler_mismatch(logged, effective) == "the run's logit_normal_mu is 1.09, not 0.0"
    assert facts.sampler_mismatch({"timestep_sampling": "logit_normal"}, effective) is None


def test_a_logged_mode_that_differs_is_named():
    effective = {**TIMESTEP_SCHEMA_DEFAULTS, "timestep_sampling": "uniform"}
    assert facts.sampler_mismatch({"timestep_sampling": "radc"}, effective) == "the run's sampler is 'radc', not 'uniform'"


def test_an_older_mean_name_sets_the_effective_mu():
    arm = _arm("moss-image-base", {"timestep_sampling": "logit_normal", "logit_normal_mean": 2})
    assert arm["effective"]["logit_normal_mu"] == 2.0
    assert arm["source"]["logit_normal_mu"] == "arm"


def test_an_older_std_name_sets_the_effective_sigma():
    arm = _arm("moss-image-base", {"timestep_sampling": "logit_normal", "logit_normal_std": 0.7})
    assert arm["effective"]["logit_normal_sigma"] == 0.7
    assert arm["source"]["logit_normal_sigma"] == "arm"


def _sampler_reads(config):
    """(mu, sigma) as TimestepSampler draws them: the same draw from a known normal."""
    drawn = TimestepSampler.sample("logit_normal", 64, torch.device("cpu"), config,
                                   generator=torch.Generator().manual_seed(3))
    normal = torch.randn((64,), generator=torch.Generator().manual_seed(3))
    logits = torch.logit(drawn.double())
    sigma = float(((logits - logits.mean()).std() / normal.double().std()))
    mu = float(logits.mean() - sigma * normal.double().mean())
    return round(mu, 3), round(sigma, 3)


def test_the_canonical_name_wins_over_the_older_one():
    mu_config = {"timestep_sampling": "logit_normal", "logit_normal_mu": 0.5, "logit_normal_mean": 2}
    sigma_config = {"timestep_sampling": "logit_normal", "logit_normal_sigma": 0.9, "logit_normal_std": 0.7}
    mu = _arm("moss-image-base", mu_config)["effective"]["logit_normal_mu"]
    sigma = _arm("moss-image-base", sigma_config)["effective"]["logit_normal_sigma"]
    assert (mu, sigma) == (0.5, 0.9)
    assert _sampler_reads(mu_config)[0] == mu
    assert _sampler_reads(sigma_config)[1] == sigma


def test_a_log_that_carries_only_the_older_names_matches():
    logged = {"timestep_sampling": "logit_normal", "logit_normal_mean": 2, "logit_normal_std": 0.7}
    effective = {**TIMESTEP_SCHEMA_DEFAULTS, "logit_normal_mu": 2.0, "logit_normal_sigma": 0.7}
    assert facts.sampler_mismatch(logged, effective) is None
    assert facts.sampler_mismatch(logged, {**effective, "logit_normal_mu": 0.0}) == \
        "the run's logit_normal_mu is 2.0, not 0.0"


def test_the_api_port_follows_the_port_variable(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"application": {"backend_port": 8000}}), encoding="utf-8")
    env = {**os.environ, "PORT": "8001", "MRLN_SETTINGS_PATH": str(settings),
           "MRLN_SERVER_LOG_PATH": str(tmp_path / "server.log"), "PYTHONDONTWRITEBYTECODE": "1"}
    got = facts.facts_in_child("moss-image-base", {"a": {}}, env, sys.executable)
    assert got["api_port"] == 8001
    assert got["arms"]["a"]["effective"]["logit_normal_mu"] == 1.09


def _gone(pid):
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def test_a_child_past_its_time_is_ended_with_everything_it_started(tmp_path):
    pids = tmp_path / "pids.txt"
    code = ("import os, pathlib, subprocess, sys, time\n"
            "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            f"pathlib.Path({str(pids)!r}).write_text(f'{{os.getpid()}} {{g.pid}}')\n"
            "time.sleep(120)\n")
    with pytest.raises(subprocess.TimeoutExpired):
        facts.run_child([sys.executable, "-c", code], dict(os.environ), timeout_s=8)
    started = [int(pid) for pid in pids.read_text().split()]
    deadline = time.monotonic() + 10
    while not all(_gone(pid) for pid in started):
        assert time.monotonic() < deadline, f"still alive: {started}"
        time.sleep(0.1)


def test_an_unknown_definition_is_an_error_naming_it():
    with pytest.raises(ValueError, match="no-such-model"):
        facts.facts_here("no-such-model", {"a": {}})
