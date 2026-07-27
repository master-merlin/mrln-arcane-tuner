"""Settings API — read and update application-level settings only.

Training, captioning, and masking templates have been moved to
domain-specific template APIs and the project system as of V4. The
generic ``/{module}`` GET/PUT is a schemaless per-module key-value store
(see ``SettingsManager`` / ``frontend/src/app/services/settings.service.ts``)
— any module name is valid, and a module that has never been written yet
simply reads back as ``{}``. There is deliberately no fixed enum of "known"
module names to 404 against.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.core.logger import get_logger
from app.core.settings_manager import SettingsManager

router = APIRouter(prefix="/api/settings", tags=["settings"])
logger = get_logger(__name__)

# Modules holding secrets — served only through dedicated masked endpoints
_PROTECTED_MODULES = frozenset({"api_captioning"})

# response_model intentionally left as the bare `dict[str, Any]` FastAPI/
# Pydantic type (not a named BaseModel) for BOTH routes below. This module is
# a schemaless per-module key-value store — any module name is valid and its
# value shape is whatever the caller last PUT, so there is no fixed field set
# to declare without risking silently stripping a module's keys the day a new
# settings field is added on the frontend. `dict[str, Any]` documents the
# contract (a JSON object) without filtering it — see task-p3c-brief.md rule 3.


@router.get("/{module}", response_model=dict[str, Any])
async def get_settings(module: str) -> dict[str, Any]:
    """Return settings for a specific module."""
    if module in _PROTECTED_MODULES:
        raise HTTPException(
            403,
            f"'{module}' holds credentials and is only accessible via "
            "/api/captions/api-providers.",
        )

    manager = SettingsManager.get_instance()
    return manager.get_module_settings(module)


@router.put("/{module}", response_model=dict[str, Any])
async def update_settings(module: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Update settings for a specific module."""
    if module in _PROTECTED_MODULES:
        raise HTTPException(
            403,
            f"'{module}' holds credentials and is only accessible via "
            "/api/captions/api-providers.",
        )

    logger.info("updating_settings", module=module)
    manager = SettingsManager.get_instance()

    # Hot-swap log level when application module changes
    if module == "application" and "log_level" in settings:
        from app.core.logger import config_log_level
        config_log_level(settings["log_level"])
        logger.info("log_level_updated", level=settings["log_level"])

    # Use the async variant so SettingsStore subscribers get an
    # entity.changed:updated broadcast (emission lives inside the async
    # method — sync callers in engine subprocesses don't have a loop).
    await manager.update_module_settings_async(module, settings)

    # update_module_settings MERGES the payload into the module's existing
    # dict (settings[module].update(settings)), so the persisted state can
    # differ from the raw request body on a partial update — re-read it so
    # the response reflects what was actually saved, not just what was sent.
    merged = manager.get_module_settings(module)

    # Rewrite runtime config when port settings change
    if module == "application":
        from app.core.runtime_config import write_runtime_config
        write_runtime_config(
            merged.get("backend_port", 8000),
            merged.get("frontend_port", 4200),
        )

    return merged
