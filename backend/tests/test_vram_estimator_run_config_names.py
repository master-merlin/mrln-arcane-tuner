"""The VRAM estimator reads the names a RUN CONFIG actually carries (plan row 4.2, F1/F2).

A saved run config states ``network_rank`` / ``train_batch_size`` /
``optimizer_type`` and a ``resolutions`` list. The estimator read only
``lora_rank``/``rank``, ``batch_size``, ``optimizer`` and a scalar
``resolution``, so the ``/jobs/estimate-vram`` route — which hands the payload
through untouched (`definition_routes.py` ``estimate_vram``) — budgeted every
job at rank 16, batch 1, AdamW and 1024 px whatever the user chose.
``cost_model.py`` already aliased the same names; the alias list lives there
ONCE and the estimator reads through it.

Only model + config rows are asserted (never the live-device rows), so no test
here queries the GPU for its verdict.
"""

from __future__ import annotations

import asyncio

import pytest

from app.api.schemas.definition_schemas import VRAMEstimateRequest
from app.api.training.definition_routes import estimate_vram
from app.engine.models.registry import registry
from app.engine.utils.vram_estimator import VRAMEstimator

_DEF = "sdxl_base_1.0"


@pytest.fixture(scope="module", autouse=True)
def _loaded_registry():
    registry.discover_families()
    registry.load_definitions("app/engine/models/definitions")
    return registry


def _route(config: dict) -> dict:
    """The REAL route handler on the payload shape the SPA posts: a run config."""
    return asyncio.run(estimate_vram(VRAMEstimateRequest(definition_id=_DEF, config=config)))


def _run_config(**over) -> dict:
    cfg = {
        "network_rank": 16,
        "train_batch_size": 1,
        "optimizer_type": "AdamW",
        "resolutions": [512],
        "gradient_checkpointing": True,
        "quantization": "none",
    }
    cfg.update(over)
    return cfg


def test_route_budgets_the_rank_the_run_config_states():
    low = _route(_run_config(network_rank=4))
    high = _route(_run_config(network_rank=64))
    # The route rounds to whole MB, so compare on the larger rows (optimizer = 6x the adapters).
    assert high["optimizer_states_mb"] == pytest.approx(16 * low["optimizer_states_mb"], rel=0.02)
    assert high["lora_adapters_mb"] > 10 * low["lora_adapters_mb"]


def test_route_budgets_the_batch_the_run_config_states():
    one = _route(_run_config(train_batch_size=1))
    four = _route(_run_config(train_batch_size=4))
    assert four["activations_mb"] == pytest.approx(4 * one["activations_mb"], rel=0.02)


def test_route_budgets_the_optimizer_the_run_config_states():
    adamw = _route(_run_config(optimizer_type="AdamW"))
    adamw8 = _route(_run_config(optimizer_type="AdamW8bit"))
    prodigy = _route(_run_config(optimizer_type="Prodigy"))
    assert adamw8["optimizer_states_mb"] == pytest.approx(adamw["optimizer_states_mb"] / 2, rel=0.02)
    assert prodigy["optimizer_states_mb"] == pytest.approx(adamw["optimizer_states_mb"] * 1.5, rel=0.02)


def test_a_resolutions_list_without_a_scalar_is_budgeted_at_its_own_edge():
    """F2: ``resolutions: [512]`` and no scalar was budgeted at the 1024 default."""
    small = _route(_run_config(resolutions=[512]))
    big = _route(_run_config(resolutions=[1024]))
    assert big["activations_mb"] == pytest.approx(4 * small["activations_mb"], rel=0.02)


def test_a_stated_scalar_resolution_still_counts():
    """The scalar is a stated quantity too: the larger of scalar and edges wins."""
    defn = registry._definitions[_DEF]
    scalar = VRAMEstimator.estimate(defn, _run_config(resolutions=[512], resolution=1024))
    edges = VRAMEstimator.estimate(defn, _run_config(resolutions=[1024]))
    assert scalar.activations_mb == edges.activations_mb


def test_formula_version_moved_with_the_name_fix():
    """Calibration is ``measured / analytic`` over a job's RUN CONFIG; the analytic rows for
    that config changed (rank / batch / optimizer / size are now read), so every coefficient
    stamped 2 or lower is stale and ``definition_stats_service`` must drop it."""
    from app.engine.utils.vram_estimator import VRAM_FORMULA_VERSION

    assert VRAM_FORMULA_VERSION >= 3


def test_the_legacy_names_still_read():
    """Pinned aliases: a caller that says ``lora_rank`` / ``batch_size`` / ``optimizer`` is not broken."""
    defn = registry._definitions[_DEF]
    legacy = VRAMEstimator.estimate(
        defn, {"lora_rank": 64, "batch_size": 4, "optimizer": "prodigy", "resolutions": [512], "quantization": "none"}
    )
    current = VRAMEstimator.estimate(
        defn, _run_config(network_rank=64, train_batch_size=4, optimizer_type="Prodigy", resolutions=[512])
    )
    for row in ("lora_adapters_mb", "optimizer_states_mb", "gradients_mb", "activations_mb"):
        assert getattr(legacy, row) == getattr(current, row), row
