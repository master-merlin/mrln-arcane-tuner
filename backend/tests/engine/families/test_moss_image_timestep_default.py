"""The MOSS-Image timestep default reaches a job started from an untouched Training form.

The family default draw is the technical report's logit-normal, mu 1.09 / sigma 0.95
(``timestep.*`` in the definition). The run-config schema's own defaults are logit_normal
0.0 / 1.0, and the form submits every field, so the family values must already be in what the
form submits. These tests build that payload the way the form does: the schema route's defaults,
the capabilities route's ``defaults`` over them, fields whose ``depends_on`` is not met left out.
Then the app's path: ``POST /api/jobs``, ``JobManager.start_job`` handing the stored config to the
training plugin, the JSON round trip of ``run_trainer.py``'s ``--config``, the family's trainer
class and its ``setup()``. The draw is recorded where the trainer hands it to ``TimestepSampler``.
Nothing loads weights or spawns a process.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest
import torch

from app.core.job_manager import JobManager
from app.engine.models.registry import registry
from app.engine.strategies.timestep_sampling import TimestepSampler

MOSS_IDS = ("moss-image-base", "moss-image-sft")
FORM_PLUGIN = "standard"  # the Training screen's plugin id, for the schema and the job POST


class _RecordingPlugin:
    """Stands in for the training plugin: records the config it launches, spawns nothing."""

    def __init__(self) -> None:
        self.launched: list[dict] = []

    def start_training(self, config):
        self.launched.append(dict(config))
        return object()


def _resolve(schema: dict, node: dict) -> dict:
    ref = node.get("$ref")
    if ref:
        return _resolve(schema, schema["$defs"][ref.rsplit("/", 1)[-1]])
    return node


def _depends_met(depends_on: str | None, payload: dict) -> bool:
    if not depends_on:
        return True
    field, _, values = depends_on.partition(":")
    return str(payload.get(field)) in values.split(",")


def untouched_form_payload(client, definition_id: str, **extra) -> dict:
    """What the Training form submits for ``definition_id`` when nothing is touched."""
    response = client.get(f"/api/plugins/{FORM_PLUGIN}/schema")
    assert response.status_code == 200, response.text
    schema = response.json()
    assert "timestep_sampling" in schema["properties"]
    caps = client.get(f"/api/models/capabilities/{definition_id}")
    assert caps.status_code == 200, caps.text
    payload: dict = {}
    for key, node in schema.get("properties", {}).items():
        node = _resolve(schema, node)
        if "default" not in node or node["default"] is None:
            continue
        value = node["default"]
        if isinstance(value, (dict,)) or (isinstance(value, list) and any(isinstance(v, dict) for v in value)):
            continue
        payload[key] = value
    payload.update(caps.json()["defaults"])
    payload = {
        key: value for key, value in payload.items()
        if _depends_met(_resolve(schema, schema["properties"].get(key, {})).get("depends_on"), payload)
    }
    payload.update({
        "definition_id": definition_id,
        "lora_name": "fox",
        "output_dir": "outputs",
        "cache_text_embeddings": True,
        "cache_latents": True,
        **extra,
    })
    return payload


def job_config_through_the_app(client, definition_id: str, config: dict) -> dict:
    """POST the job, start it, and return the config ``run_trainer.py`` hands the trainer."""
    manager = JobManager()
    with patch("app.api.training.job_routes.job_manager", manager), \
            patch.object(manager, "schedule_advance_queue", lambda: None), \
            patch("app.core.db.repositories.job_repo.JobHistoryRepository"):
        response = client.post("/api/jobs", json={"plugin_id": FORM_PLUGIN, "config": config})
        assert response.status_code == 200, response.text
        job_id = response.json()["id"]
        plugin = _RecordingPlugin()
        with patch("app.core.job_manager.plugin_manager.get_plugin", return_value=plugin):
            manager.start_job(job_id, clear_stale_signal=False, preflight=False)
    assert len(plugin.launched) == 1
    return json.loads(json.dumps(plugin.launched[0]))  # run_trainer's --config


def set_up_trainer(definition, run_config: dict):
    family_cls = registry.get_family_class(definition.family)
    trainer_cls = family_cls(definition, run_config).get_trainer_class()
    trainer = trainer_cls(definition=definition, run_config=run_config)
    asyncio.run(trainer.setup())
    return trainer


class DrawRecorder:
    """Records each ``TimestepSampler.sample`` call's mode and config; returns mid sigmas."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, mode, bs, device, config, **_kw):
        self.calls.append((mode, dict(config)))
        return torch.full((bs,), 0.5)


@pytest.fixture
def draws(monkeypatch):
    recorder = DrawRecorder()
    monkeypatch.setattr(TimestepSampler, "sample", staticmethod(recorder))
    return recorder


@pytest.fixture(scope="module")
def definitions():
    """Plugin discovery runs in the app lifespan, which the ``client`` fixture does not execute."""
    from app.core.plugin_manager import plugin_manager

    registry.initialize()
    if not plugin_manager.get_plugin(FORM_PLUGIN):
        plugin_manager.discover_plugins()
    return {did: registry.get_definition(did) for did in MOSS_IDS}


def _drawn(client, definitions, draws, did, config) -> tuple[str, dict]:
    run_config = job_config_through_the_app(client, did, config)
    trainer = set_up_trainer(definitions[did], run_config)
    trainer.sample_timesteps(2)
    assert len(draws.calls) == 1
    return draws.calls[0]


@pytest.mark.parametrize("did", MOSS_IDS)
def test_an_untouched_form_job_draws_the_family_logit_normal(client, definitions, draws, did):
    payload = untouched_form_payload(client, did)
    assert payload["logit_normal_mu"] == 1.09  # the form submits the family's values
    mode, config = _drawn(client, definitions, draws, did, payload)
    assert mode == "logit_normal"
    assert config["logit_normal_mu"] == 1.09
    assert config["logit_normal_sigma"] == 0.95


def test_a_job_naming_no_mode_draws_the_family_logit_normal(client, definitions, draws):
    payload = {"definition_id": "moss-image-base", "lora_name": "fox", "output_dir": "outputs",
               "cache_text_embeddings": True, "cache_latents": True}
    mode, config = _drawn(client, definitions, draws, "moss-image-base", payload)
    assert (mode, config["logit_normal_mu"], config["logit_normal_sigma"]) == ("logit_normal", 1.09, 0.95)


def test_the_users_own_mu_is_kept(client, definitions, draws):
    payload = untouched_form_payload(client, "moss-image-base", logit_normal_mu=0.5)
    mode, config = _drawn(client, definitions, draws, "moss-image-base", payload)
    assert (mode, config["logit_normal_mu"], config["logit_normal_sigma"]) == ("logit_normal", 0.5, 0.95)


def test_a_legacy_mean_alias_is_the_users_value(client, definitions, draws):
    payload = {"definition_id": "moss-image-base", "lora_name": "fox", "output_dir": "outputs",
               "cache_text_embeddings": True, "cache_latents": True,
               "timestep_sampling": "logit_normal", "logit_normal_mean": 0.3}
    mode, config = _drawn(client, definitions, draws, "moss-image-base", payload)
    assert mode == "logit_normal"
    assert "logit_normal_mu" not in config
    assert config["logit_normal_mean"] == 0.3


def test_setup_writes_the_family_default_into_the_jobs_config(client, definitions):
    payload = {"definition_id": "moss-image-base", "lora_name": "fox", "output_dir": "outputs",
               "cache_text_embeddings": True, "cache_latents": True}
    run_config = job_config_through_the_app(client, "moss-image-base", payload)
    trainer = set_up_trainer(definitions["moss-image-base"], run_config)
    assert trainer.config["timestep_sampling"] == "logit_normal"
    assert trainer.config["logit_normal_mu"] == 1.09
    assert trainer.config["logit_normal_sigma"] == 0.95


def test_a_legacy_mean_without_a_mode_is_still_the_users_value(client, definitions, draws):
    payload = {"definition_id": "moss-image-base", "lora_name": "fox", "output_dir": "outputs",
               "cache_text_embeddings": True, "cache_latents": True, "logit_normal_mean": 0.3}
    mode, config = _drawn(client, definitions, draws, "moss-image-base", payload)
    assert mode == "logit_normal"
    assert "logit_normal_mu" not in config  # the sampler reads mu before the mean alias
    assert config["logit_normal_mean"] == 0.3
    assert config["logit_normal_sigma"] == 0.95


def test_every_family_trainer_runs_the_shared_setup(definitions):
    from app.engine.core.pipeline.pipeline_base import PipelineBaseMixin

    seen = set()
    for definition in registry._definitions.values():
        family_cls = registry.get_family_class(definition.family)
        trainer_cls = family_cls(definition, {}).get_trainer_class()
        if trainer_cls in seen:
            continue
        seen.add(trainer_cls)
        assert issubclass(trainer_cls, PipelineBaseMixin), trainer_cls
        assert trainer_cls.setup is PipelineBaseMixin.setup, (
            f"{trainer_cls.__name__} overrides setup(): the family timestep defaults may not reach it")
    assert len(seen) > 10


def test_a_user_picked_mode_is_kept(client, definitions, draws):
    payload = untouched_form_payload(client, "moss-image-base", timestep_sampling="uniform")
    mode, _config = _drawn(client, definitions, draws, "moss-image-base", payload)
    assert mode == "uniform"
