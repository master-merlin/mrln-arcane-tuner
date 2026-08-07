"""Adaptive LoRA layer targeting — sub-config and factory presets (spec §4)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Factory preset knob values (spec §4 table). Seeded as readonly templates in
# the `adaptive` template domain; duplicated here so the trainer validates
# without a DB dependency.
FACTORY_PRESETS: dict[str, dict] = {
    "conservative": dict(
        warmup_pct=0.40,
        interval_steps=300,
        energy_threshold=0.97,
        min_active_pct=0.35,
        heat_ema=0.60,
    ),
    "balanced": dict(
        warmup_pct=0.25,
        interval_steps=200,
        energy_threshold=0.93,
        min_active_pct=0.25,
        heat_ema=0.50,
    ),
    "aggressive": dict(
        warmup_pct=0.15,
        interval_steps=150,
        energy_threshold=0.85,
        min_active_pct=0.15,
        heat_ema=0.35,
    ),
}


class AdaptiveTargetingConfig(BaseModel):
    """Validated shape of the ``adaptive_targeting_config`` dict field."""

    preset: str = "factory:balanced"  # provenance only — values below are authoritative
    warmup_pct: float = Field(0.25, ge=0.0, lt=1.0)
    interval_steps: int = Field(200, ge=10)
    energy_threshold: float = Field(0.93, gt=0.0, le=1.0)
    min_active_pct: float = Field(0.25, gt=0.0, le=1.0)
    heat_ema: float = Field(0.5, ge=0.0, lt=1.0)
    reactivation: bool = False
    probe_every: int = Field(5, ge=2)
    probe_steps: int = Field(30, ge=1)
    action: Literal["freeze", "rebuild"] = "freeze"
    rebuild_min_shrink_pct: float = Field(25.0, gt=0.0, le=100.0)

    @model_validator(mode="after")
    def _cross_field_rules(self) -> "AdaptiveTargetingConfig":
        if self.probe_steps >= self.interval_steps:
            raise ValueError("probe_steps must be < interval_steps")
        if self.action == "rebuild" and self.reactivation:
            # Post-rebuild the optimizer holds no state/groups for frozen
            # params — a probe could not step them and would silently measure
            # zero heat. Rebuild mode is contractually monotonic.
            raise ValueError("action='rebuild' cannot be combined with reactivation")
        return self
