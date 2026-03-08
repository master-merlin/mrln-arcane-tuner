import importlib
import inspect
import os

from app.engine.models.base import TrainingPlugin
import structlog

logger = structlog.get_logger(__name__)

class PluginManager:
    def __init__(self, plugin_dir: str = "app/engine/models"):
        self.plugin_dir = plugin_dir
        self._plugins: dict[str, TrainingPlugin] = {}

    def discover_plugins(self):
        """
        Scans the plugin directory and loads classes inheriting from TrainingPlugin.
        """
        # clear existing
        self._plugins = {}
        
        # Construct absolute path to plugins directory
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
        plugins_path = os.path.join(base_path, "engine", "models")

        if not os.path.exists(plugins_path):
             logger.error("plugins_directory_not_found", path=plugins_path)
             return

        for filename in os.listdir(plugins_path):
            if filename.endswith(".py") and filename not in ["__init__.py", "base.py"]:
                module_name = f"app.engine.models.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, TrainingPlugin) and 
                            obj is not TrainingPlugin):
                            
                            # Instantiate the plugin
                            plugin_instance = obj()
                            model_id = plugin_instance.get_model_id()
                            self._plugins[model_id] = plugin_instance
                            logger.info("plugin_loaded", model_id=model_id, module=module_name)
                except Exception as e:
                    logger.error("plugin_load_failed", file=filename, error=str(e))

    def get_plugin(self, model_id: str) -> TrainingPlugin:
        return self._plugins.get(model_id)

    def list_plugins(self) -> list[dict[str, str]]:
        return [{"id": pid} for pid in self._plugins.keys()]

# Global instance
plugin_manager = PluginManager()
