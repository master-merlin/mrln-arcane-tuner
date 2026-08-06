"""Pure validation helpers for template import.

No DB, no model registry, no FastAPI — only the Pydantic schemas that describe
template configs and model definitions. The route layer supplies registry /
repo / offline state; this module answers "is this data valid / supported?".
"""

from __future__ import annotations

from typing import Any

from app.core.schemas.captioning_settings import CAPTION_PARAM_MODELS
from app.core.schemas.masking_settings import MASKING_PARAM_MODELS
from app.engine.core.definitions import ModelDefinition
from app.engine.models.adaptive import AdaptiveTargetingConfig

_PARAM_MODELS = {"captioning": CAPTION_PARAM_MODELS, "masking": MASKING_PARAM_MODELS}


def model_available(domain: str, model_id: str) -> bool:
    """True if *model_id* is a built-in model for *domain*.

    Training has no ``model_id`` registry (it uses a definition) and adaptive
    presets are not model-scoped at all, so both are always available here.
    """
    registry = _PARAM_MODELS.get(domain)
    if registry is None:
        return True
    return model_id in registry


def validate_config(domain: str, model_id: str | None, config: Any) -> str | None:
    """Return a warning string if *config* fails its param schema, else None.

    Non-blocking. Training is skipped (templates legitimately omit ``datasets``,
    which the full training config requires). Unknown captioning/masking models
    have no schema and are skipped here (availability is a separate check).
    """
    if domain == "adaptive":
        # An archive is untrusted: a preset whose knobs violate the schema
        # (or its cross-field rules) must be surfaced here, not discovered
        # when it is later materialized into a job config.
        try:
            AdaptiveTargetingConfig.model_validate(config or {})
        except Exception as exc:  # noqa: BLE001 — surface any validation error as text
            return str(exc)
        return None

    registry = _PARAM_MODELS.get(domain)
    if registry is None:
        return None
    param_cls = registry.get(model_id or "")
    if param_cls is None:
        return None
    try:
        param_cls.model_validate(config or {})
    except Exception as exc:  # noqa: BLE001 — surface any validation error as text
        return str(exc)
    return None


def validate_carried_definition(definition: Any) -> str | None:
    """Return an error string if the carried definition fails the
    ``ModelDefinition`` schema, else None."""
    try:
        ModelDefinition.model_validate(definition)
    except Exception as exc:  # noqa: BLE001
        return str(exc)
    return None
