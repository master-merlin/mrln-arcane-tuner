"""Settings API — read and update application-level settings only.

Training, captioning, and masking templates have been moved to
domain-specific template APIs and the project system as of V4.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.core.logger import get_logger
from app.core.settings_manager import SettingsManager

router = APIRouter(prefix="/api/settings", tags=["settings"])
logger = get_logger(__name__)

# Modules that have been migrated to the project/template system
_MIGRATED_MODULES = frozenset({"training", "captioning", "masking"})

# Modules holding secrets — served only through dedicated masked endpoints
_PROTECTED_MODULES = frozenset({"api_captioning"})


@router.get("/{module}")
async def get_settings(module: str) -> dict[str, Any]:
    """Return settings for a specific module."""
    if module in _PROTECTED_MODULES:
        raise HTTPException(
            403,
            f"'{module}' holds credentials and is only accessible via "
            "/api/captions/api-providers.",
        )
    if module in _MIGRATED_MODULES:
        raise HTTPException(
            410,
            f"'{module}' settings have been moved to the project/template API. "
            "Use /api/templates/{domain} and /api/projects endpoints instead.",
        )

    manager = SettingsManager.get_instance()
    return manager.get_module_settings(module)


@router.put("/{module}")
async def update_settings(module: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Update settings for a specific module."""
    if module in _PROTECTED_MODULES:
        raise HTTPException(
            403,
            f"'{module}' holds credentials and is only accessible via "
            "/api/captions/api-providers.",
        )
    if module in _MIGRATED_MODULES:
        raise HTTPException(
            410,
            f"'{module}' settings have been moved to the project/template API. "
            "Use /api/templates/{domain} and /api/projects endpoints instead.",
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

    # Rewrite runtime config when port settings change
    if module == "application":
        from app.core.runtime_config import write_runtime_config
        merged = manager.get_module_settings("application")
        write_runtime_config(
            merged.get("backend_port", 8000),
            merged.get("frontend_port", 4200),
        )

    return settings
