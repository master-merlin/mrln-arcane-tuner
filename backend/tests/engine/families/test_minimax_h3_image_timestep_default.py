"""The MiniMax-H3-Image job default reaches a real job: API -> job config -> trainer setup.

The family's default draw is ``fixed_sigmas`` over the model's 6 sampling sigmas
(``timestep.default_mode`` / ``timestep.fixed_sigmas``). The run-config schema's own default is
``logit_normal`` (``base.py``), so the question is what an unset field arrives as. These tests
run the path the app runs:

1. ``POST /api/jobs`` (the real route and the real ``JobManager.create_job``: availability
   gate, video contract, capability allowlist);
2. ``JobManager.start_job`` handing the stored config to the training plugin;
3. what ``run_trainer.py`` does with it: the JSON round trip of ``--config``, the family's
   ``get_trainer_class()``, ``TrainerClass(definition=..., run_config=config)`` and ``setup()``.

An unset field reaches the trainer as ``fixed_sigmas`` with the 6-value list; an explicit
``logit_normal`` reaches it untouched. Nothing loads weights or spawns a process.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from app.core.job_manager import JobManager
from app.engine.models.registry import registry
from tests.engine.families.test_minimax_h3_image_trainer import (
    DEFINITION_ID,
    H3_IMAGE_SIGMAS,
)


class _RecordingPlugin:
    """Stands in for the training plugin: records the config it launches, spawns nothing."""

    def __init__(self) -> None:
        self.launched: list[dict] = []

    def start_training(self, config):
        self.launched.append(dict(config))
        return object()


@pytest.fixture
def definition():
    """The shipped definition, exactly as the registry loads it from its YAML."""
    registry.initialize()
    defn = registry.get_definition(DEFINITION_ID)
    assert defn is not None, f"{DEFINITION_ID} is not in the registry"
    assert defn.family == "minimax_h3_image"
    return defn


def _job_config(**extra) -> dict:
    return {
        "definition_id": DEFINITION_ID,
        "lora_name": "fox",
        "output_dir": "outputs",
        "mixed_precision": "bf16",
        "cache_text_embeddings": True,
        "cache_latents": True,
        **extra,
    }


def _through_the_app(client, definition, config: dict):
    """POST the job, start it, and build the trainer the way ``run_trainer.py`` does."""
    manager = JobManager()
    with patch("app.api.training.job_routes.job_manager", manager), \
            patch.object(manager, "schedule_advance_queue", lambda: None), \
            patch("app.core.db.repositories.job_repo.JobHistoryRepository"):
        response = client.post("/api/jobs", json={"plugin_id": DEFINITION_ID, "config": config})
        assert response.status_code == 200, response.text
        job_id = response.json()["id"]
        plugin = _RecordingPlugin()
        with patch("app.core.job_manager.plugin_manager.get_plugin", return_value=plugin):
            manager.start_job(job_id, clear_stale_signal=False, preflight=False)
    assert len(plugin.launched) == 1
    run_config = json.loads(json.dumps(plugin.launched[0]))  # run_trainer's --config
    family_cls = registry.get_family_class(definition.family)
    trainer_cls = family_cls(definition, run_config).get_trainer_class()
    trainer = trainer_cls(definition=definition, run_config=run_config)
    asyncio.run(trainer.setup())
    return trainer


def test_an_unset_timestep_setting_reaches_the_trainer_as_the_family_default(client, definition):
    trainer = _through_the_app(client, definition, _job_config())
    assert trainer.config["timestep_sampling"] == "fixed_sigmas"
    assert trainer.config["fixed_sigmas"] == H3_IMAGE_SIGMAS


def test_an_explicit_logit_normal_reaches_the_trainer_untouched(client, definition):
    trainer = _through_the_app(client, definition, _job_config(timestep_sampling="logit_normal"))
    assert trainer.config["timestep_sampling"] == "logit_normal"
    assert not trainer.config.get("fixed_sigmas")


def test_an_explicit_fixed_sigmas_without_a_list_gets_the_models_list(client, definition):
    trainer = _through_the_app(
        client, definition, _job_config(timestep_sampling="fixed_sigmas", fixed_sigmas=None)
    )
    assert trainer.config["fixed_sigmas"] == H3_IMAGE_SIGMAS


def test_an_explicit_list_is_kept(client, definition):
    own = [1.0, 0.75, 0.23200887441635132]
    trainer = _through_the_app(client, definition, _job_config(timestep_sampling="fixed_sigmas", fixed_sigmas=own))
    assert trainer.config["fixed_sigmas"] == own


def test_a_list_outside_the_range_is_refused_at_setup(client, definition):
    with pytest.raises(ValueError, match="fixed_sigmas"):
        _through_the_app(client, definition, _job_config(timestep_sampling="fixed_sigmas", fixed_sigmas=[1.5]))


def test_the_default_drives_the_draw():
    """The default the trainer set is the one its step draws from: every sigma a grid value."""
    import torch

    from tests.engine.families.test_minimax_h3_image_trainer import set_up

    trainer = set_up()
    assert trainer.config["timestep_sampling"] == "fixed_sigmas"
    torch.manual_seed(0)
    sigmas = {round(1.0 - float(trainer.sample_timesteps(2)[0]), 6) for _ in range(300)}
    assert sigmas == {round(s, 6) for s in H3_IMAGE_SIGMAS}
