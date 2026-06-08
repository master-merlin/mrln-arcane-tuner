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

_PARAM_MODELS = {"captioning": CAPTION_PARAM_MODELS, "masking": MASKING_PARAM_MODELS}


def model_available(domain: str, model_id: str) -> bool:
    """True if *model_id* is a built-in model for *domain*.

    Training has no ``model_id`` registry (it uses a definition), so it is
    always considered available here.
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
