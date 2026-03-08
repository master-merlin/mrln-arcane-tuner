
import os
import json
from typing import TYPE_CHECKING, Any
import structlog

if TYPE_CHECKING:
    from app.core.schemas.captioning_settings import CaptioningSettings
    from app.core.schemas.masking_settings import MaskingSettings
    from app.core.schemas.training_settings import TrainingSettings

logger = structlog.get_logger(__name__)

class SettingsManager:
    """
    Singleton service for managing global application settings.
    Settings are stored in a JSON file in the application root.
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

    # ── Typed accessors ──────────────────────────────────────────────────

    def get_captioning_settings(self) -> "CaptioningSettings":
        """Get validated captioning settings."""
        from app.core.schemas.captioning_settings import CaptioningSettings
        raw = self.get_module_settings("captioning")
        settings = CaptioningSettings.model_validate(raw) if raw else CaptioningSettings()
        settings.migrate_defaults()
        return settings

    def update_captioning_settings(self, settings: "CaptioningSettings"):
        """Save validated captioning settings."""
        self.update_module_settings("captioning", settings.model_dump(mode="json"))

    def get_masking_settings(self) -> "MaskingSettings":
        """Get validated masking settings."""
        from app.core.schemas.masking_settings import MaskingSettings
        raw = self.get_module_settings("masking")
        settings = MaskingSettings.model_validate(raw) if raw else MaskingSettings()
        settings.migrate_defaults()
        return settings

    def update_masking_settings(self, settings: "MaskingSettings"):
        """Save validated masking settings."""
        self.update_module_settings("masking", settings.model_dump(mode="json"))

    def get_training_settings(self) -> "TrainingSettings":
        """Get validated training settings."""
        from app.core.schemas.training_settings import TrainingSettings
        raw = self.get_module_settings("training")
        return TrainingSettings.model_validate(raw) if raw else TrainingSettings()

    def update_training_settings(self, settings: "TrainingSettings"):
        """Save validated training settings."""
        self.update_module_settings("training", settings.model_dump(mode="json"))


def get_settings_manager() -> SettingsManager:
    return SettingsManager.get_instance()


