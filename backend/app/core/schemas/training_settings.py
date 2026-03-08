"""
Pydantic models for training settings.

These models validate the 'training' module in settings.json,
covering the template system for training configurations.

The per-template ``config`` reuses ``BaseTrainingConfig`` from the engine,
which already handles all field groups (BASE, STRATEGY, NETWORK, OPTIMIZER,
ENGINE, SAMPLING) including optimizer-dependent params (AdamW vs Prodigy).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Template ────────────────────────────────────────────────────────────


class TrainingTemplate(BaseModel):
    """A single training settings template.

    Each template is linked to a ``definition_id`` (the model family it
    targets) and carries a ``config`` dict that conforms to
    ``BaseTrainingConfig``.  The config is stored as ``dict[str, Any]``
    to support forward-compatibility with new fields added to definitions.
    """
    id: str
    name: str
    definition_id: str
    is_default: bool = False
    config: dict[str, Any] = Field(default_factory=dict)

    def validate_config(self) -> list[str]:
        """
        Validate the config dict against BaseTrainingConfig.
        Returns a list of warning messages (non-throwing).
        """
        warnings: list[str] = []
        try:
            from app.engine.models.base import BaseTrainingConfig
            BaseTrainingConfig.model_validate(self.config)
        except Exception as exc:
            warnings.append(f"Template '{self.name}' ({self.id}): {exc}")
        return warnings


# ── Module Root ─────────────────────────────────────────────────────────


class TrainingSettings(BaseModel):
    """Root schema for the 'training' module in settings.json.

    Unlike captioning/masking (which nest per-model), training stores a
    flat list of templates, each linked to a model family via
    ``definition_id``.
    """
    templates: list[TrainingTemplate] = Field(default_factory=list)

    def validate_all_configs(self) -> list[str]:
        """
        Validate config for all templates.  Returns a flat list of warnings.
        """
        warnings: list[str] = []
        for tpl in self.templates:
            tpl_warnings = tpl.validate_config()
            for w in tpl_warnings:
                warnings.append(f"[{tpl.definition_id}] {w}")
        return warnings
