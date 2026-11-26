"""The comparison's config, its queue, each run's config and the identity a finished run records."""
import dataclasses
import json

import pytest

from app.core.sampler_compare import plan
from app.engine.core.pipeline.validation_pass import is_due
from app.engine.models.base import BaseTrainingConfig
from app.engine.strategies.timestep_keys import TIMESTEP_KEYS

CONFIG = {
    "version": 1,
    "name": "radc-moss",
    "reference_job": "job-1",
    "dataset": "Porsche 911 Targa",
    "arms": {
        "uniform": {"timestep_sampling": "uniform"},
        "logit_normal": {"timestep_sampling": "logit_normal", "logit_normal_mu": 0.0, "logit_normal_sigma": 1.0},
        "radc": {"timestep_sampling": "radc", "radc_start": 0.9, "radc_end": 0.3, "radc_width": 0.5,
                 "radc_res_influence": 0.15, "radc_uniform_prob": 0.0},
    },
    "seeds": [42, 43],
    "steps": 2000,
    "validation": {"every": 500, "noise_draws": 4, "holdout_fraction": 0.3, "split_seed": 0},
    "port": 8001,
    "limits": {"temperature_c": 94, "vram_mib": 88000, "free_ram_gib": 10, "stall_minutes": 15,
               "run_minutes": 120, "idle_vram_mib": 20000, "idle_wait_minutes": 30, "power_limit_w": 500},
}
REQUIRED = ["version", "name", "reference_job", "arms", "seeds", "steps", "validation", "port", "limits"]

BASE = {
    "learning_rate": 0.0001,
    "datasets": [{"dataset_name": "Original", "repeats": 2}],
    "timestep_sampling": "radc",
    "radc_start": 0.8,
    "logit_normal_mean": 0.5,
    "validation_enabled": False,
    "seed": 1,
}


def _write(tmp_path, config):
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _comparison(tmp_path, **changes):
    config = json.loads(json.dumps(CONFIG))
    config.update(changes)
    return plan.load_comparison(_write(tmp_path, config))


def test_a_config_loads_with_every_field(tmp_path):
    c = _comparison(tmp_path)
    assert c.name == "radc-moss"
    assert c.reference_job == "job-1"
    assert c.dataset == "Porsche 911 Targa"
    assert list(c.arms) == ["uniform", "logit_normal", "radc"]
    assert c.seeds == (42, 43)
    assert c.steps == 2000
    assert (c.validation.every, c.validation.noise_draws, c.validation.holdout_fraction,
            c.validation.split_seed) == (500, 4, 0.3, 0)
    assert c.port == 8001
    assert c.limits.power_limit_w == 500
    assert c.allow_downloads is False
    assert c.cpu_affinity is None
    assert c.work_dir == plan.BACKEND / "outputs" / "sampler_compare" / "radc-moss"


@pytest.mark.parametrize("key", [*REQUIRED, "unknown_key"])
def test_a_missing_or_unknown_key_is_refused_by_name(tmp_path, key):
    config = json.loads(json.dumps(CONFIG))
    if key == "unknown_key":
        config[key] = 1
    else:
        del config[key]
    with pytest.raises(plan.ConfigError, match=key):
        plan.load_comparison(_write(tmp_path, config))


def test_a_version_other_than_one_is_refused(tmp_path):
    with pytest.raises(plan.ConfigError, match="version"):
        _comparison(tmp_path, version=2)


def test_one_seed_is_refused(tmp_path):
    with pytest.raises(plan.ConfigError, match="needs two or more seeds"):
        _comparison(tmp_path, seeds=[42])


def test_an_arm_key_that_is_no_timestep_key_is_refused(tmp_path):
    with pytest.raises(plan.ConfigError, match="learning_rate"):
        _comparison(tmp_path, arms={"a": {"learning_rate": 1}, "b": {}})


def test_the_queue_runs_every_arm_per_seed_in_config_order(tmp_path):
    c = _comparison(tmp_path)
    assert [plan.label(a, s) for a, s in plan.queue(c)] == [
        "uniform_s42", "logit_normal_s42", "radc_s42", "uniform_s43", "logit_normal_s43", "radc_s43",
    ]


def test_the_passes_are_every_interval_and_the_last_step():
    assert plan.passes(2000, 500) == [0, 500, 1000, 1500, 1999]
    assert plan.passes(2000, 500) == [s for s in range(2000) if is_due(s, 500, 2000)]


def test_the_default_band_count_is_the_schemas():
    assert plan.DEFAULT_BANDS == BaseTrainingConfig.model_fields["validation_timestep_bands"].default


def test_an_arm_never_inherits_the_reference_jobs_sampler(tmp_path):
    c = _comparison(tmp_path)
    config = plan.run_config(BASE, c, "uniform", 42)
    assert config["timestep_sampling"] == "uniform"
    assert "radc_start" not in config
    assert "logit_normal_mean" not in config


def _differing(a, b):
    """Every key outside the sampler, the seed and the names in which two run configs differ."""
    keys = (set(a) | set(b)) - TIMESTEP_KEYS - plan.NAME_AND_PATH_KEYS - {"seed"}
    return sorted(k for k in keys if (k in a) != (k in b) or a.get(k) != b.get(k))


def test_arms_differ_only_in_timestep_keys_seed_and_names(tmp_path):
    c = _comparison(tmp_path)
    configs = [plan.run_config(BASE, c, arm, seed) for arm, seed in plan.queue(c)]
    for a in configs:
        for b in configs:
            assert _differing(a, b) == []
    changed = plan.run_config({**BASE, "learning_rate": 0.0002}, c, "radc", 42)
    assert _differing(configs[0], changed) == ["learning_rate"]


def test_a_separate_validation_dataset_of_the_reference_job_is_dropped(tmp_path):
    c = _comparison(tmp_path)
    config = plan.run_config({**BASE, "validation_dataset": "Other"}, c, "uniform", 42)
    assert "validation_dataset" not in config
    assert len(config["datasets"]) == 1
    assert config["datasets"][0] == {"dataset_name": "Porsche 911 Targa", "repeats": 2}


@pytest.mark.parametrize("base", [{**BASE, "validation_enabled": False},
                                  {k: v for k, v in BASE.items() if k != "validation_enabled"}])
def test_validation_is_switched_on_whatever_the_reference_job_says(tmp_path, base):
    config = plan.run_config(base, _comparison(tmp_path), "uniform", 42)
    assert config["validation_enabled"] is True


def test_the_band_count_is_set_explicitly(tmp_path):
    c = _comparison(tmp_path)
    assert plan.run_config(BASE, c, "uniform", 42)["validation_timestep_bands"] == 8
    with_four = {**BASE, "validation_timestep_bands": 4}
    assert plan.run_config(with_four, c, "uniform", 42)["validation_timestep_bands"] == 4


@pytest.mark.parametrize("base", [{**BASE, "store_quantized_version": True}, BASE])
def test_a_run_never_stores_a_quantized_copy(tmp_path, base):
    config = plan.run_config(base, _comparison(tmp_path), "uniform", 42)
    assert config["store_quantized_version"] is False


def _identity(c, base=BASE, arm="radc", seed=42, definition="moss-image-base", digest="a", effective=None):
    config = plan.run_config(base, c, arm, seed)
    if effective is None:
        effective = {k: v for k, v in config.items() if k in TIMESTEP_KEYS}
    return plan.identity(c, definition, digest, config, effective)


def _with_validation(c, **changes):
    return dataclasses.replace(c, validation=dataclasses.replace(c.validation, **changes))


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("learning_rate", ["config.learning_rate"]),
        ("seed", ["config.seed"]),
        ("steps", ["config.max_train_steps", "passes"]),
        ("every", ["config.validation_every_n_steps", "passes"]),
        ("draws", ["config.validation_noise_draws"]),
        ("holdout", ["config.validation_holdout_fraction"]),
        ("radc_end", ["config.radc_end", "effective_sampler.radc_end"]),
        ("definition", ["definition_id"]),
        ("digest", ["dataset_digest"]),
        ("version", ["version"]),
    ],
)
def test_an_identity_names_everything_that_sets_the_result(tmp_path, monkeypatch, change, expected):
    c = _comparison(tmp_path)
    recorded = json.loads(json.dumps(_identity(c)))
    if change == "learning_rate":
        requested = _identity(c, base={**BASE, "learning_rate": 0.0002})
    elif change == "seed":
        requested = _identity(c, seed=44)
    elif change == "steps":
        requested = _identity(dataclasses.replace(c, steps=1000))
    elif change == "every":
        requested = _identity(_with_validation(c, every=250))
    elif change == "draws":
        requested = _identity(_with_validation(c, noise_draws=1))
    elif change == "holdout":
        requested = _identity(_with_validation(c, holdout_fraction=0.2))
    elif change == "radc_end":
        arms = {**c.arms, "radc": {**c.arms["radc"], "radc_end": 0.2}}
        requested = _identity(dataclasses.replace(c, arms=arms))
    elif change == "definition":
        requested = _identity(c, definition="moss-image-sft")
    elif change == "digest":
        requested = _identity(c, digest="b")
    else:
        monkeypatch.setattr(plan, "IDENTITY_VERSION", 2)
        requested = _identity(c)
    assert plan.identity_difference(recorded, requested) == expected


def test_names_and_paths_are_no_part_of_an_identity(tmp_path):
    c = _comparison(tmp_path)
    elsewhere = dataclasses.replace(c, work_dir=tmp_path / "elsewhere")
    assert plan.identity_difference(_identity(c), _identity(elsewhere)) == []


def test_a_run_without_an_identity_differs(tmp_path):
    assert plan.identity_difference(None, _identity(_comparison(tmp_path))) == ["no recorded identity"]
