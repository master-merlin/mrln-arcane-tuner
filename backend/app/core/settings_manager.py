
import os
import json
import asyncio
from typing import Any
import structlog

logger = structlog.get_logger(__name__)

class SettingsManager:
    """
    Singleton service for managing global application settings.

    As of V4, this only handles application-level config (ports, log level).
    Training, captioning, and masking templates are in SQLite via the
    project/template system.
    """
    _instance = None
    
    def __init__(self, storage_file: str = "settings.json"):
        # Resolve absolute paths relative to this file
        # This file is in backend/app/core/
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.storage_file = os.path.join(self.root_dir, storage_file)
        
        self.settings: dict[str, Any] = {}
        self.load()

        # Ensure default application settings
        defaults = {
            "backend_port": 8000,
            "frontend_port": 4200,
            "log_level": "INFO"
        }
        
        if "application" not in self.settings:
            self.settings["application"] = defaults
            self.save()
        else:
            # Merge checks: If any key is missing in existing settings, add it
            changed = False
            for k, v in defaults.items():
                if k not in self.settings["application"]:
                    self.settings["application"][k] = v
                    changed = True
            if changed:
                self.save()

    @classmethod
    def get_instance(cls) -> "SettingsManager":
        if cls._instance is None:
            cls._instance = SettingsManager()
        return cls._instance

    def load(self):
        """Load settings from disk."""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    self.settings = json.load(f)
            except Exception as e:
                logger.error("settings_load_failed", path=self.storage_file, error=str(e))
                self.settings = {}
        else:
            self.settings = {}

    def save(self):
        """Save settings to disk."""
        try:
            # Defensive: Ensure application settings exist before saving
            if "application" not in self.settings:
                logger.warning("missing_app_settings_restoring_default")
                self.settings["application"] = {
                    "backend_port": 8000,
                    "frontend_port": 4200,
                    "log_level": "INFO"
                }

            # Enforce 'application' key is first for readability
            ordered_settings = {}
            if "application" in self.settings:
                ordered_settings["application"] = self.settings["application"]
            
            for k, v in self.settings.items():
                if k != "application":
                    ordered_settings[k] = v
            
            with open(self.storage_file, 'w') as f:
                json.dump(ordered_settings, f, indent=2)
        except Exception as e:
            logger.error("settings_save_failed", path=self.storage_file, error=str(e))

    def get_module_settings(self, module: str) -> dict[str, Any]:
        """Get settings for a specific module."""
        self.load()
        return self.settings.get(module, {})

    def update_module_settings(self, module: str, settings: dict[str, Any]):
        """
        Update settings for a specific module.
        Reloads from disk first to prevent overwriting other modules/keys with stale state.
        """
        self.load() # Merge strategy: Load latest
        if module not in self.settings:
            self.settings[module] = {}
        
        # Merge if dict, otherwise replace
        if isinstance(self.settings[module], dict) and isinstance(settings, dict):
             self.settings[module].update(settings)
        else:
             self.settings[module] = settings

        self.save()

    # ── Async variants (R-API-07) ───────────────────────────────────────
    # Offload disk I/O to a worker thread so FastAPI route handlers
    # don't block the event loop. Sync versions stay for engine callers
    # (engine/utils/model_override_manager.py, engine/core/pipeline/pipeline_data.py)
    # which run in trainer subprocess contexts without an event loop.

    async def load_async(self) -> None:
        """Async wrapper around load()."""
        await asyncio.to_thread(self.load)

    async def save_async(self) -> None:
        """Async wrapper around save()."""
        await asyncio.to_thread(self.save)

    async def get_module_settings_async(self, module: str) -> dict[str, Any]:
        """Async wrapper around get_module_settings()."""
        return await asyncio.to_thread(self.get_module_settings, module)

    async def update_module_settings_async(
        self, module: str, settings: dict[str, Any],
    ) -> None:
        """Async wrapper around update_module_settings()."""
        await asyncio.to_thread(self.update_module_settings, module, settings)


def get_settings_manager() -> SettingsManager:
    return SettingsManager.get_instance()


