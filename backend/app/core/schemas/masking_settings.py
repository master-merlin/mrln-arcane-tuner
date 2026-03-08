"""
Pydantic models for masking settings.

These models validate the 'masking' module in settings.json,
covering the template system used by all masking model backends.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Per-Model Param Schemas ─────────────────────────────────────────────


class Sam3Params(BaseModel):
    """Parameters for Meta SAM 3 segmentation model."""
    text_prompt: str = "subject"
    multimask_output: bool = True
    max_hole_area: int = Field(0, ge=0, le=1000)
    max_sprinkle_area: int = Field(0, ge=0, le=1000)


class RembgParams(BaseModel):
    """Parameters for RemBG background removal model."""
    model_name: Literal[
        "u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg",
        "silueta", "isnet-general-use", "isnet-anime",
        "birefnet-general", "birefnet-general-lite", "birefnet-massive",
        "birefnet-portrait", "birefnet-dis", "birefnet-hrsod", "birefnet-cod",
        "bria-rmbg",
    ] = "birefnet-general"
    alpha_matting: bool = False
    alpha_matting_foreground_threshold: int = Field(240, ge=0, le=255)
    alpha_matting_background_threshold: int = Field(10, ge=0, le=255)
    alpha_matting_erode_size: int = Field(10, ge=0, le=50)
    post_process_mask: bool = True


# Registry mapping model IDs to their param schemas
MASKING_PARAM_MODELS: dict[str, type[BaseModel]] = {
    "sam3": Sam3Params,
    "rembg": RembgParams,
}


# ── Template ────────────────────────────────────────────────────────────


class MaskingTemplate(BaseModel):
    """A single masking settings template."""
    id: str
    name: str
    is_default: bool = False
    readonly: bool = False
    params: dict[str, Any] = Field(default_factory=dict)


# ── Per-Model Settings ──────────────────────────────────────────────────


class MaskingModelSettings(BaseModel):
    """Settings for a single masking model (template list + active selection)."""
    active_template_id: str = "default"
    templates: list[MaskingTemplate] = Field(default_factory=list)

    def validate_params(self, model_id: str) -> list[str]:
        """
        Validate all template params against the known param schema for *model_id*.
        Returns a list of warning messages for any issues found (non-throwing).
        """
        warnings: list[str] = []
        param_cls = MASKING_PARAM_MODELS.get(model_id)
        if param_cls is None:
            return warnings

        for tpl in self.templates:
            try:
                param_cls.model_validate(tpl.params)
            except Exception as exc:
                warnings.append(f"Template '{tpl.name}' ({tpl.id}): {exc}")
        return warnings


# ── Module Root ─────────────────────────────────────────────────────────


class MaskingSettings(BaseModel):
    """Root schema for the 'masking' module in settings.json."""
    models: dict[str, MaskingModelSettings] = Field(default_factory=dict)
    selected_model: str = "sam3"
    saved_concepts: list[str] = Field(default_factory=list)

    def migrate_defaults(self) -> None:
        """Refresh readonly default templates from code-defined param defaults.

        Called on load to ensure persisted default templates stay in sync
        with the latest code defaults (e.g. new default model variant,
        new params, removed params).
        """
        for model_id, param_cls in MASKING_PARAM_MODELS.items():
            model_settings = self.models.get(model_id)
            if model_settings is None:
                continue
            code_defaults = param_cls().model_dump()
            for tpl in model_settings.templates:
                if tpl.id == "default" and tpl.readonly:
                    tpl.params = code_defaults

    def validate_all_params(self) -> list[str]:
        """
        Validate params for all models.  Returns a flat list of warning strings.
        """
        warnings: list[str] = []
        for model_id, model_settings in self.models.items():
            model_warnings = model_settings.validate_params(model_id)
            for w in model_warnings:
                warnings.append(f"[{model_id}] {w}")
        return warnings
