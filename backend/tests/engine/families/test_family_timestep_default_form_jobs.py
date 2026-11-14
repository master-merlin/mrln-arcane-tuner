"""Each family's documented timestep default reaches a job, from the form and from the API.

flux3 (``uniform``, under its fixed 6.93 shift), minimax_h3 (``uniform``, research section 4) and
omnigen2 (``flux_shift`` base 0.5 / max 1.15, upstream's training shift) carried their default only
as a code fallback, which an untouched form (it always submits the schema's ``logit_normal``)
never reached. Their definitions now state it as ``timestep.*`` keys. The jobs run the app's path
(the helpers of ``test_moss_image_timestep_default``): the form's payload, ``POST /api/jobs``,
``start_job``, the ``--config`` JSON round trip, the trainer's ``setup()``, and the draw recorded
at ``TimestepSampler.sample``.
"""

from __future__ import annotations

import pytest
import torch

from app.engine.core.archetypes import resolve_capabilities
from app.engine.models.registry import registry
from app.engine.strategies.timestep_sampling import TimestepSampler
from tests.engine.families.test_moss_image_timestep_default import (
    FORM_PLUGIN,
    DrawRecorder,
    job_config_through_the_app,
    set_up_trainer,
    untouched_form_payload,
)


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

FAMILY_DEFAULTS = {
    "flux3-image-klein-base": ("uniform", {}),
    "minimax-h3-t2va": ("uniform", {}),
    "omnigen2": ("flux_shift", {"flux_shift_base": 0.5, "flux_shift_max": 1.15}),
}
USER_MODE = "sigmoid"  # a mode none of them replaces


def _drawn(client, did, config, draws) -> tuple[str, dict]:
    run_config = job_config_through_the_app(client, did, config)
    trainer = set_up_trainer(registry.get_definition(did), run_config)
    trainer.sample_timesteps(2)
    assert draws.calls, f"{did}: no TimestepSampler.sample call"
    return draws.calls[-1]


@pytest.mark.parametrize("did", sorted(FAMILY_DEFAULTS))
def test_an_untouched_form_job_draws_the_family_default(client, definitions, draws, did):
    mode, params = FAMILY_DEFAULTS[did]
    drawn_mode, config = _drawn(client, did, untouched_form_payload(client, did), draws)
    assert drawn_mode == mode
    for key, value in params.items():
        assert config[key] == value, key


@pytest.mark.parametrize("did", sorted(FAMILY_DEFAULTS))
def test_a_user_picked_mode_is_kept(client, definitions, draws, did):
    payload = untouched_form_payload(client, did, timestep_sampling=USER_MODE)
    drawn_mode, _config = _drawn(client, did, payload, draws)
    assert drawn_mode == USER_MODE


@pytest.mark.parametrize("did", ["minimax-h3-fl2va", "minimax-h3-ref2va"])
def test_the_gated_minimax_definitions_default_to_uniform(definitions, draws, did):
    definition = registry.get_definition(did)
    assert definition.unavailable_reason  # they stay refused at create and start
    assert resolve_capabilities(definition)["defaults"]["timestep_sampling"] == "uniform"
    from app.engine.models.families.minimax_h3.driver import MiniMaxH3Driver

    driver = MiniMaxH3Driver(definition, torch.device("cpu"))
    driver.sample_timesteps(2, torch.device("cpu"), {})
    assert draws.calls[-1][0] == "uniform"


def test_the_expert_router_refuses_a_config_without_a_mode():
    from app.engine.models.families.wan22.expert_router import ExpertRouter

    with pytest.raises(ValueError, match="timestep_sampling"):
        ExpertRouter(0.9, 1, {})


def test_a_wan22_job_naming_no_mode_routes_on_its_definition_default(definitions):
    from app.engine.models.families.wan22.expert_router import ExpertRouter
    from app.engine.strategies.timestep_sampling import apply_family_timestep_defaults

    config = apply_family_timestep_defaults(registry.get_definition("wan2.2-t2v-a14b"), {})
    assert ExpertRouter(0.9, 1, config).mode == "model_shift"
