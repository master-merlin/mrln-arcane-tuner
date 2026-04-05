"""Model Source Override schemas — per-model path/source configuration.

Stored in ``settings.json`` under the ``models`` module.  These schemas
define the three supported source types and per-model overrides that
control how the engine locates model weights at load time.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, model_validator


class ModelSourceType(str, Enum):
    """Where to load model weights from."""

    HF_HUB = "hf_hub"
    LOCAL_DIFFUSERS = "local_diffusers"
    LOCAL_SAFETENSORS = "local_safetensors"


class ModelOverride(BaseModel):
    """Per-model source override.

    Stored in ``settings.json`` at ``models.overrides.<definition_id>``.

    Attributes:
        source_type:  How to resolve the model path.
        local_path:   Absolute filesystem path (required for ``local_*`` types).
        skip_update:  When ``True`` and source is ``hf_hub``, use
                      ``local_files_only=True`` — never contact HF Hub.
    """

    source_type: ModelSourceType = ModelSourceType.HF_HUB
    local_path: str | None = None
    skip_update: bool = False

    @model_validator(mode="after")
    def validate_local_path_required(self) -> "ModelOverride":
        """Require ``local_path`` when source_type is a local variant."""
        if self.source_type in (
            ModelSourceType.LOCAL_DIFFUSERS,
            ModelSourceType.LOCAL_SAFETENSORS,
        ):
            if not self.local_path:
                raise ValueError(
                    "local_path is required for local source types"
                )
        return self


class ModelSettings(BaseModel):
    """Top-level model settings module.

    Stored in ``settings.json`` under the ``models`` key.

    Attributes:
        global_offline_mode:  When ``True``, ALL HF Hub models use
                              ``local_files_only=True`` (no network).
        default_model_path:   Global base directory for model storage.
                              Used as the initial browse directory and
                              HF cache redirect when set.
        overrides:            Per-definition source overrides keyed by
                              ``ModelDefinition.id``.
    """

    global_offline_mode: bool = False
    default_model_path: str = ""
    overrides: dict[str, ModelOverride] = {}
