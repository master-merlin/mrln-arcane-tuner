"""The comparison's config, its queue, each run's config, and the identity a finished run records."""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from app.engine.strategies.timestep_keys import TIMESTEP_KEYS

BACKEND = Path(__file__).resolve().parents[3]

CONFIG_VERSION = 1
IDENTITY_VERSION = 1
# BaseTrainingConfig's default, which validation_pass / validation_split read for a config without
# one (that module imports torch; a test pins the two equal).
DEFAULT_BANDS = 8

# Run-config keys that name a run's own folders or files: they differ between runs by design.
NAME_AND_PATH_KEYS = frozenset({
    "lora_name", "output_dir", "run_dir", "logging_dir", "log_dir", "job_name",
    "validation_output_dir", "validation_split_path", "validation_metrics_path",
})

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ARM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$")
_REQUIRED = ("version", "name", "reference_job", "arms", "seeds", "steps", "validation", "port", "limits")
_OPTIONAL = ("dataset", "global_triggerword", "cpu_affinity", "allow_downloads", "work_dir")
_VALIDATION = ("every", "noise_draws", "holdout_fraction", "split_seed")
_LIMITS = ("temperature_c", "vram_mib", "free_ram_gib", "stall_minutes", "run_minutes",
           "idle_vram_mib", "idle_wait_minutes", "power_limit_w")


class ConfigError(ValueError):
    """The comparison's config, or the reference job it names, can't make a comparison."""


@dataclass(frozen=True)
class Limits:
    temperature_c: float
    vram_mib: int
    free_ram_gib: float
    stall_minutes: float
    run_minutes: float
    idle_vram_mib: int
    idle_wait_minutes: float
    power_limit_w: float | None


@dataclass(frozen=True)
class Validation:
    every: int
    noise_draws: int
    holdout_fraction: float
    split_seed: int


@dataclass(frozen=True)
class Comparison:
    name: str
    reference_job: str
    dataset: str | None
    global_triggerword: str | None
    arms: dict[str, dict[str, Any]]
    seeds: tuple[int, ...]
    steps: int
    validation: Validation
    port: int
    limits: Limits
    cpu_affinity: tuple[int, ...] | None
    allow_downloads: bool
    work_dir: Path


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _keys(where: str, given: dict, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> None:
    if not isinstance(given, dict):
        raise ConfigError(f"{where} must be an object")
    for key in required:
        if key not in given:
            raise ConfigError(f"{where} lacks the key {key!r}")
    for key in given:
        if key not in required and key not in optional:
            raise ConfigError(f"{where} has an unknown key {key!r}")


def _positive_int(where: str, value: Any) -> int:
    if not _integer(value) or value < 1:
        raise ConfigError(f"{where} must be a positive integer, not {value!r}")
    return value


def _limits(given: Any) -> Limits:
    _keys("limits", given, _LIMITS)
    for key in _LIMITS:
        value = given[key]
        if key == "power_limit_w" and value is None:
            continue
        if not _number(value) or value <= 0:
            raise ConfigError(f"limits.{key} must be a positive number, not {value!r}")
    return Limits(**{key: given[key] for key in _LIMITS})


def _validation(given: Any) -> Validation:
    _keys("validation", given, _VALIDATION)
    fraction = given["holdout_fraction"]
    if not _number(fraction) or not 0 < fraction < 1:
        raise ConfigError(f"validation.holdout_fraction must lie between 0 and 1, not {fraction!r}")
    if not _integer(given["split_seed"]):
        raise ConfigError(f"validation.split_seed must be an integer, not {given['split_seed']!r}")
    return Validation(
        every=_positive_int("validation.every", given["every"]),
        noise_draws=_positive_int("validation.noise_draws", given["noise_draws"]),
        holdout_fraction=float(fraction),
        split_seed=given["split_seed"],
    )


def _arms(given: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(given, dict) or len(given) < 2:
        raise ConfigError("arms must name two or more arms")
    for arm, settings in given.items():
        if not _ARM.match(arm):
            raise ConfigError(f"the arm name {arm!r} must be letters, digits, '_' or '-'")
        if not isinstance(settings, dict):
            raise ConfigError(f"arm {arm} must be an object of timestep keys")
        for key in settings:
            if key not in TIMESTEP_KEYS:
                raise ConfigError(f"arm {arm}: {key!r} is no timestep key")
    return {arm: dict(settings) for arm, settings in given.items()}


def load_comparison(path: Path) -> Comparison:
    """Read and check a comparison's config file."""
    path = Path(path)
    try:
        given = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{path}: {exc}") from None
    _keys(str(path), given, _REQUIRED, _OPTIONAL)
    if given["version"] != CONFIG_VERSION:
        raise ConfigError(f"version {given['version']!r} is not {CONFIG_VERSION}")
    name = given["name"]
    if not isinstance(name, str) or not _NAME.match(name):
        raise ConfigError(f"name {name!r} must be letters, digits, '.', '_' or '-'")
    if not isinstance(given["reference_job"], str) or not given["reference_job"]:
        raise ConfigError("reference_job must be a job id")
    for key in ("dataset", "global_triggerword"):
        if given.get(key) is not None and not isinstance(given[key], str):
            raise ConfigError(f"{key} must be text")
    seeds = given["seeds"]
    if not isinstance(seeds, list) or not all(_integer(s) for s in seeds):
        raise ConfigError("seeds must be a list of integers")
    repeated = sorted({s for s in seeds if seeds.count(s) > 1})
    if repeated:
        raise ConfigError(f"seed {repeated[0]} repeats: each seed runs once per arm")
    if len(seeds) < 2:
        raise ConfigError("the comparison needs two or more seeds (distinct), else it has no noise floor")
    port = given["port"]
    if not _integer(port) or not 1 <= port <= 65535:
        raise ConfigError(f"port {port!r} is no port")
    affinity = given.get("cpu_affinity")
    if affinity is not None and (not isinstance(affinity, list) or not affinity
                                 or not all(_integer(cpu) and cpu >= 0 for cpu in affinity)):
        raise ConfigError("cpu_affinity must be a non-empty list of CPU numbers")
    host = psutil.cpu_count() or 1
    for cpu in affinity or ():
        if cpu >= host:
            raise ConfigError(f"cpu_affinity names CPU {cpu}; this host has CPUs 0 to {host - 1}")
    downloads = given.get("allow_downloads", False)
    if not isinstance(downloads, bool):
        raise ConfigError("allow_downloads must be true or false")
    work = given.get("work_dir")
    if work is None:
        work_dir = BACKEND / "outputs" / "sampler_compare" / name
    elif isinstance(work, str) and work:
        work_dir = Path(work) if Path(work).is_absolute() else path.resolve().parent / work
    else:
        raise ConfigError("work_dir must be a folder path")
    return Comparison(
        name=name,
        reference_job=given["reference_job"],
        dataset=given.get("dataset"),
        global_triggerword=given.get("global_triggerword"),
        arms=_arms(given["arms"]),
        seeds=tuple(seeds),
        steps=_positive_int("steps", given["steps"]),
        validation=_validation(given["validation"]),
        port=port,
        limits=_limits(given["limits"]),
        cpu_affinity=tuple(affinity) if affinity is not None else None,
        allow_downloads=downloads,
        work_dir=work_dir,
    )


def queue(c: Comparison) -> list[tuple[str, int]]:
    """Seed by seed, the arms in config order."""
    return [(arm, seed) for seed in c.seeds for arm in c.arms]


def label(arm: str, seed: int) -> str:
    return f"{arm}_s{seed}"


def passes(steps: int, every: int) -> list[int]:
    """The steps a validation pass runs at (``validation_pass.is_due``)."""
    return [step for step in range(steps) if step % every == 0 or step == steps - 1]


def run_names(c: Comparison, arm: str, seed: int) -> dict[str, str]:
    """The run-config keys that name a run's own output."""
    return {"output_dir": str(c.work_dir / "runs"), "lora_name": f"{c.name}_{label(arm, seed)}"}


def dataset_entry(base: dict[str, Any]) -> Any:
    """The reference job's one dataset entry; a job with another count is refused."""
    datasets = base.get("datasets") or []
    if len(datasets) != 1:
        raise ConfigError(f"the reference job has {len(datasets)} dataset entries, not one")
    return datasets[0]


def dataset_name(config: dict[str, Any]) -> str:
    entry = dataset_entry(config)
    return entry if isinstance(entry, str) else str(entry.get("dataset_name", ""))


def run_config(base: dict, c: Comparison, arm: str, seed: int) -> dict:
    """One run's config: the reference job without its sampler, plus the arm and the comparison."""
    config = copy.deepcopy(base)
    for key in TIMESTEP_KEYS:
        config.pop(key, None)
    entry = dataset_entry(config)
    if c.dataset is not None:
        entry = {**entry, "dataset_name": c.dataset} if isinstance(entry, dict) else c.dataset
    config["datasets"] = [entry]
    if c.global_triggerword is not None:
        config["global_triggerword"] = c.global_triggerword
    # A separate validation dataset would switch the trainer to it, and the copy holds only one;
    # the reference job's own id names that job, not this run.
    config.pop("validation_dataset", None)
    config.pop("job_id", None)
    config.update(copy.deepcopy(c.arms[arm]))
    config.update(
        seed=seed,
        max_train_steps=c.steps,
        validation_enabled=True,
        validation_every_n_steps=c.validation.every,
        validation_noise_draws=c.validation.noise_draws,
        validation_holdout_fraction=c.validation.holdout_fraction,
        validation_split_seed=c.validation.split_seed,
        validation_timestep_bands=base.get("validation_timestep_bands", DEFAULT_BANDS),
        # Else a cache miss saves a quantized copy under backend/models, outside the work folder.
        store_quantized_version=False,
        sample_every_n_steps=0,
        sample_before_training=False,
        sample_prompts=[],
        save_every_n_steps=0,
        resume_from_checkpoint="",
        **run_names(c, arm, seed),
    )
    return config


def identity(c: Comparison, definition_id: str, dataset_digest: str, config: dict, effective: dict,
             model_source: dict) -> dict:
    """What sets a run's result: a finished run is reused only for an equal identity.
    ``model_source`` is where the weights load from (``isolation.model_source``)."""
    content = {
        "version": IDENTITY_VERSION,
        "definition_id": definition_id,
        "model_source": model_source,
        "dataset": dataset_name(config),
        "dataset_digest": dataset_digest,
        "config": {k: v for k, v in config.items() if k not in NAME_AND_PATH_KEYS},
        "effective_sampler": effective,
        "passes": passes(c.steps, c.validation.every),
    }
    return json.loads(json.dumps(content))  # the form it has once read back from its file


def identity_difference(recorded: dict | None, requested: dict) -> list[str]:
    """The keys (dotted inside ``config`` and ``effective_sampler``) in which two identities differ."""
    if recorded is None:
        return ["no recorded identity"]
    recorded, requested = json.loads(json.dumps(recorded)), json.loads(json.dumps(requested))
    out = []
    for key in sorted(set(recorded) | set(requested)):
        a, b = recorded.get(key), requested.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            out += [f"{key}.{k}" for k in sorted(set(a) | set(b)) if (k in a) != (k in b) or a.get(k) != b.get(k)]
        elif (key in recorded) != (key in requested) or a != b:
            out.append(key)
    return sorted(out)
