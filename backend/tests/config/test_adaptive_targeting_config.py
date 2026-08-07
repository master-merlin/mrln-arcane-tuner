"""Validation contract for the adaptive-targeting sub-config (spec §4)."""

import pytest
from pydantic import ValidationError

from app.engine.models.adaptive import FACTORY_PRESETS, AdaptiveTargetingConfig
from app.engine.models.base import BaseTrainingConfig


def test_defaults_are_balanced_preset_values():
    cfg = AdaptiveTargetingConfig()
    assert cfg.preset == "factory:balanced"
    assert cfg.warmup_pct == 0.25
    assert cfg.interval_steps == 200
    assert cfg.energy_threshold == 0.93
    assert cfg.min_active_pct == 0.25
    assert cfg.heat_ema == 0.5
    assert cfg.reactivation is False
    assert cfg.action == "freeze"
    assert cfg.rebuild_min_shrink_pct == 25.0


@pytest.mark.parametrize(
    "bad",
    [
        {"energy_threshold": 0.0},
        {"energy_threshold": 1.5},
        {"warmup_pct": -0.1},
        {"warmup_pct": 1.0},
        {"interval_steps": 5},
        {"min_active_pct": 0.0},
        {"min_active_pct": 1.5},
        {"probe_steps": 200, "interval_steps": 200},  # must be < interval
        {"rebuild_min_shrink_pct": 0.0},
        {"rebuild_min_shrink_pct": 101.0},
        {"action": "prune"},
        {"heat_ema": -0.1},
        {"heat_ema": 1.0},
    ],
)
def test_out_of_range_values_rejected(bad):
    with pytest.raises(ValidationError):
        AdaptiveTargetingConfig(**bad)


def test_rebuild_excludes_reactivation():
    """Spec §5: post-rebuild the optimizer has no state for frozen params, so a
    probe could not step them — reject the combination at config time."""
    with pytest.raises(ValidationError):
        AdaptiveTargetingConfig(action="rebuild", reactivation=True)
    # each alone is fine
    AdaptiveTargetingConfig(action="rebuild")
    AdaptiveTargetingConfig(reactivation=True)


def test_factory_presets_match_spec_table():
    assert FACTORY_PRESETS["conservative"]["energy_threshold"] == 0.97
    assert FACTORY_PRESETS["balanced"]["interval_steps"] == 200
    assert FACTORY_PRESETS["aggressive"]["min_active_pct"] == 0.15
    for values in FACTORY_PRESETS.values():
        AdaptiveTargetingConfig(**values)  # every preset validates


def test_base_config_validates_subconfig_only_when_enabled():
    # BaseTrainingConfig requires `datasets` (min_length=1); every other field
    # used here has a default, so this is the minimal valid construction —
    # mirrors the `_minimal()` pattern in test_dead_key_compat.py.
    base = {
        "model_family": "flux2",
        "definition_id": "dev",
        "output_dir": "outputs",
        "datasets": [{"dataset_name": "demo"}],
    }
    # Off: garbage sub-config is ignored (feature disabled → inert dict)
    BaseTrainingConfig.model_validate(
        {
            **base,
            "adaptive_targeting": False,
            "adaptive_targeting_config": {"action": "prune"},
        }
    )
    # On: same garbage must be rejected at config time, never mid-run
    with pytest.raises(ValidationError):
        BaseTrainingConfig.model_validate(
            {
                **base,
                "adaptive_targeting": True,
                "adaptive_targeting_config": {"action": "prune"},
            }
        )
