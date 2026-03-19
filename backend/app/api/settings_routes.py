"""Settings API — read and update per-module application settings."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.core.logger import get_logger
from app.core.settings_manager import SettingsManager

router = APIRouter(prefix="/api/settings", tags=["settings"])
logger = get_logger(__name__)


@router.get("/{module}")
async def get_settings(module: str) -> dict[str, Any]:
    """Return settings for a specific module."""
    manager = SettingsManager.get_instance()

    if module == "captioning":
        return manager.get_captioning_settings().model_dump(mode="json")

    if module == "masking":
        return manager.get_masking_settings().model_dump(mode="json")

    if module == "training":
        return manager.get_training_settings().model_dump(mode="json")

    return manager.get_module_settings(module)


@router.put("/{module}")
async def update_settings(module: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Update settings for a specific module."""
    logger.info("updating_settings", module=module)
    manager = SettingsManager.get_instance()

    # Hot-swap log level when application module changes
    if module == "application" and "log_level" in settings:
        from app.core.logger import config_log_level
        config_log_level(settings["log_level"])
        logger.info("log_level_updated", level=settings["log_level"])

    if module == "captioning":
        from app.core.schemas.captioning_settings import CaptioningSettings
        validated = CaptioningSettings.model_validate(settings)
        manager.update_captioning_settings(validated)
        return validated.model_dump(mode="json")

    if module == "masking":
        from app.core.schemas.masking_settings import MaskingSettings
        validated = MaskingSettings.model_validate(settings)
        manager.update_masking_settings(validated)
        return validated.model_dump(mode="json")

    if module == "training":
        from app.core.schemas.training_settings import TrainingSettings
        validated = TrainingSettings.model_validate(settings)
        manager.update_training_settings(validated)
        return validated.model_dump(mode="json")

    manager.update_module_settings(module, settings)

    # Rewrite runtime config when port settings change
    if module == "application":
        from app.core.runtime_config import write_runtime_config
        merged = manager.get_module_settings("application")
        write_runtime_config(
            merged.get("backend_port", 8000),
            merged.get("frontend_port", 4200),
        )

    return settings
