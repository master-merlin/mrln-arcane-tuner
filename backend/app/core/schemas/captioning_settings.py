"""
Pydantic models for captioning settings.

These models validate the 'captioning' module in settings.json,
covering the template system used by all caption model backends.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Per-Model Param Schemas ─────────────────────────────────────────────


class Florence2Params(BaseModel):
    """Parameters for Microsoft Florence-2 captioning model."""
    task_type: Literal[
        "Caption", "Detailed Caption", "More Detailed Caption"
    ] = "Detailed Caption"
    max_tokens: int = Field(512, ge=64, le=2048)
    num_beams: int = Field(5, ge=1, le=10)


class YoutuVLParams(BaseModel):
    """Parameters for Tencent Youtu-VL captioning model."""
    max_long_side: int | str = Field(768)
    max_num_patches: int = Field(256, ge=64, le=1024)
    temperature: float = Field(0.1, ge=0, le=2)
    top_p: float = Field(0.001, ge=0, le=1)
    repetition_penalty: float = Field(1.05, ge=1, le=2)
    max_tokens: int = Field(512, ge=64, le=32768)


class Qwen3VLParams(BaseModel):
    """Parameters for Alibaba Qwen3 VL captioning model."""
    max_long_side: int | str = Field(1280)
    temperature: float = Field(0.7, ge=0, le=2)
    top_p: float = Field(0.8, ge=0, le=1)
    num_beams: int = Field(1, ge=1, le=10)
    repetition_penalty: float = Field(1.2, ge=1, le=2)
    max_tokens: int = Field(512, ge=64, le=2048)
    frames: int = Field(16, ge=1, le=64)


class JoyCaptionParams(BaseModel):
    """Parameters for JoyCaption Beta captioning model."""
    caption_type: Literal[
        "Descriptive", "Descriptive (Casual)", "Straightforward",
        "Stable Diffusion Prompt", "MidJourney",
        "Danbooru tag list", "e621 tag list", "Rule34 tag list",
        "Booru-like tag list",
        "Art Critic", "Product Listing", "Social Media Post",
    ] = "Descriptive"
    caption_length: Literal[
        "any", "very short", "short", "medium-length", "long", "very long"
    ] = "long"
    temperature: float = Field(0.6, ge=0, le=2)
    top_p: float = Field(0.9, ge=0, le=1)
    max_tokens: int = Field(512, ge=64, le=2048)
    name_input: str = ""

    # ── Extra options (prompt modifiers) ─────────────────────────────
    include_lighting: bool = False
    include_camera_angle: bool = False
    include_watermark: bool = False
    include_jpeg_artifacts: bool = False
    include_exif: bool = False
    include_aesthetic_quality: bool = False
    include_composition: bool = False
    include_nsfw_rating: bool = False
    include_shot_type: bool = False
    include_vantage_height: bool = False
    include_character_age: bool = False
    identify_orientation: bool = False
    specify_depth_field: bool = False
    specify_lighting_sources: bool = False
    exclude_people_info: bool = False
    exclude_sexual: bool = False
    exclude_resolution: bool = False
    exclude_text: bool = False
    exclude_mood: bool = False
    exclude_artist_name: bool = False
    no_ambiguous_language: bool = False
    no_euphemisms: bool = False
    use_profanity: bool = False
    only_important_elements: bool = False
    mention_watermark: bool = False
    avoid_meta_phrases: bool = False
    refer_character_name: bool = False


class ApiCaptionParams(BaseModel):
    """Parameters for external OpenAI-compatible API captioning providers."""
    model: str = ""
    temperature: float = Field(0.7, ge=0, le=2)
    top_p: float = Field(1.0, ge=0, le=1)
    max_tokens: int = Field(512, ge=64, le=8192)
    max_long_side: int | str = Field(1024)


# Registry mapping model IDs to their param schemas
CAPTION_PARAM_MODELS: dict[str, type[BaseModel]] = {
    "florence-2": Florence2Params,
    "youtu-vl": YoutuVLParams,
    "qwen3-vl": Qwen3VLParams,
    "joycaption": JoyCaptionParams,
    "api-openai": ApiCaptionParams,
    "api-anthropic": ApiCaptionParams,
    "api-gemini": ApiCaptionParams,
    "api-openrouter": ApiCaptionParams,
    "api-custom": ApiCaptionParams,
}


# ── Template ────────────────────────────────────────────────────────────


class CaptionTemplate(BaseModel):
    """A single captioning settings template."""
    id: str
    name: str
    is_default: bool = False
    readonly: bool = False
    system_prompt: str = "Describe this image in detail."
    params: dict[str, Any] = Field(default_factory=dict)


# ── Per-Model Settings ──────────────────────────────────────────────────


class CaptionModelSettings(BaseModel):
    """Settings for a single captioning model (template list + active selection)."""
    active_template_id: str = "default"
    templates: list[CaptionTemplate] = Field(default_factory=list)

    def validate_params(self, model_id: str) -> list[str]:
        """
        Validate all template params against the known param schema for *model_id*.
        Returns a list of warning messages for any issues found (non-throwing).
        """
        warnings: list[str] = []
        param_cls = CAPTION_PARAM_MODELS.get(model_id)
        if param_cls is None:
            return warnings  # unknown model — skip validation

        for tpl in self.templates:
            try:
                param_cls.model_validate(tpl.params)
            except Exception as exc:
                warnings.append(f"Template '{tpl.name}' ({tpl.id}): {exc}")
        return warnings


# ── Module Root ─────────────────────────────────────────────────────────


class CaptioningSettings(BaseModel):
    """Root schema for the 'captioning' module in settings.json."""
    models: dict[str, CaptionModelSettings] = Field(default_factory=dict)
    selected_model: str = "florence-2"
    qwen3_variant: str = "4B-Instruct"

    def migrate_defaults(self) -> None:
        """Refresh readonly default templates from code-defined param defaults."""
        for model_id, param_cls in CAPTION_PARAM_MODELS.items():
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
