"""Central model registry.

Manages auto-discovery of ``ModelFamily`` implementations and loading
of ``ModelDefinition`` YAML files.  The singleton ``registry`` instance
is the single source of truth for all registered families and definitions.
"""

from __future__ import annotations

import importlib
import inspect
import os

import structlog
import yaml

from app.engine.core.definitions import ModelDefinition, ModelFamily

logger = structlog.get_logger(__name__)

class ModelRegistry:
    """Central registry for model families and YAML-based definitions."""

    _families: dict[str, type[ModelFamily]] = {}
    _definitions: dict[str, ModelDefinition] = {}
    _paths: dict[str, str] = {}
    _discovered = False
    _definitions_loaded = False


    @classmethod
    def register_family(cls, name: str, family_class: type[ModelFamily]) -> None:
        """Register a model family class under the given name."""
        cls._families[name] = family_class

    @classmethod
    def initialize(cls) -> None:
        """One-stop initialization: discover families then load definitions."""
        cls.discover_families()
        defs_dir = os.path.join(os.getcwd(), "app/engine/models/definitions")
        cls.load_definitions(defs_dir)

    @classmethod
    def discover_families(cls) -> None:
        """Auto-discover and import all model families in the families directory."""
        if cls._discovered:
            return

        current_dir = os.path.dirname(os.path.abspath(__file__))
        families_dir = os.path.join(current_dir, "families")
        
        if not os.path.exists(families_dir):
            return

        for item in os.listdir(families_dir):
            item_path = os.path.join(families_dir, item)
            
            if os.path.isfile(item_path) and item.endswith(".py") and item != "__init__.py":
                module_name = f"app.engine.models.families.{item[:-3]}"
                cls._load_family_module(module_name, item[:-3])
            elif os.path.isdir(item_path) and item != "__pycache__":
                # Check for family.py in directory
                if os.path.exists(os.path.join(item_path, "family.py")):
                    module_name = f"app.engine.models.families.{item}.family"
                    cls._load_family_module(module_name, item)
        
        cls._discovered = True

    @classmethod
    def _load_family_module(cls, module_name: str, fallback_name: str) -> None:
        """Import a single family module and register any ModelFamily subclasses."""
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, ModelFamily) and 
                    obj is not ModelFamily):
                    
                    family_name = getattr(obj, "family_name", fallback_name)
                    cls.register_family(family_name, obj)
                    logger.info("family_registered", family=family_name, class_name=obj.__name__)
        except (ImportError, AttributeError, TypeError) as e:
            logger.error("family_load_failed", module=module_name, error=str(e))

    @classmethod
    def load_definitions(cls, definitions_dir: str) -> None:
        """Scan central + family-specific directories for YAML definitions."""
        if cls._definitions_loaded:
             return

        cls._definitions = {} # Clear existing
        defs_dir = definitions_dir or os.path.join(os.getcwd(), "app/engine/models/definitions")

        # 1. Scan Central Definitions
        if os.path.exists(defs_dir):
            cls._scan_dir_for_definitions(defs_dir)

        # 2. Scan Family Definitions
        current_dir = os.path.dirname(os.path.abspath(__file__))
        families_dir = os.path.join(current_dir, "families")
        if os.path.exists(families_dir):
            for family in os.listdir(families_dir):
                family_def_dir = os.path.join(families_dir, family, "definitions")
                if os.path.isdir(family_def_dir):
                    cls._scan_dir_for_definitions(family_def_dir)
        
        cls._definitions_loaded = True

    @classmethod
    def load_definition(cls, path: str) -> ModelDefinition:
        """Load a single YAML definition file into the registry."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        # Normalize component shorthands
        if "components" in data:
            for k, v in data["components"].items():
                if isinstance(v, str):
                    data["components"][k] = {"path": v}
        
        definition = ModelDefinition(**data)
        cls._definitions[definition.id] = definition
        cls._paths[definition.id] = path
        return definition

    @classmethod
    def _scan_dir_for_definitions(cls, directory: str):
        for filename in os.listdir(directory):
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                path = os.path.join(directory, filename)
                try:
                    cls.load_definition(path)
                    logger.info("definition_loaded", path=path)
                except (OSError, yaml.YAMLError, ValueError, TypeError) as e:
                    logger.error("definition_load_failed", path=path, error=str(e))

    @classmethod
    def get_family_class(cls, family_id: str) -> type[ModelFamily]:
        """Return the family class for the given ID, or raise ValueError."""
        if family_id not in cls._families:
            raise ValueError(f"ModelFamily '{family_id}' not found in registry.")
        return cls._families[family_id]

    @classmethod
    def get_definition(cls, model_id: str) -> ModelDefinition | None:
        """Return the definition for the given model ID, or ``None``."""
        return cls._definitions.get(model_id)

    @classmethod
    def update_definition(cls, definition_id: str, changes: dict) -> None:
        """Update in-memory definition; uses Pydantic ``model_copy``."""
        if definition_id not in cls._definitions:
            raise ValueError(f"Definition {definition_id} not found.")
        
        current = cls._definitions[definition_id]
        cls._definitions[definition_id] = current.model_copy(update=changes)
        
    @classmethod
    def save_definition(cls, definition_id: str) -> None:
        """Persist definition back to its YAML file."""
        if definition_id not in cls._definitions:
            raise ValueError(f"Definition {definition_id} not found.")
            
        config = cls._definitions[definition_id]
        path = cls._paths.get(definition_id)
        
        if not path:
             raise ValueError(f"No file path known for {definition_id}. Cannot save.")
             
        with open(path, "w") as f:
            yaml.dump(config.model_dump(), f, sort_keys=False)

    @classmethod
    def enrich_definition(
        cls, definition_id: str, components: dict, root_path: str | None = None
    ) -> None:
        """Auto-enrich a definition with introspected component data.

        Fills empty fields and updates ``architecture_params`` with the
        latest values from the repo's config.json files.  If *root_path*
        is provided, also harvests HuggingFace ``config.json`` files for
        additional architecture parameters.
        """
        defn = cls.get_definition(definition_id)
        if not defn:
            logger.warning("enrich_skipped_not_found", id=definition_id)
            return

        from app.engine.utils.introspection import ModelIntrospector
        introspector = ModelIntrospector()
        result = introspector.introspect(components)

        # Config harvesting — merge into introspection result
        if root_path:
            from app.engine.utils.config_harvester import harvest
            harvested = harvest(root_path)
            if harvested:
                # Harvested values WIN — repo is source of truth
                merged = {**result.architecture_params, **harvested}
                result.architecture_params = merged

        # Only fill empty fields
        changes = {}
        if not defn.detected_precision and result.detected_precision:
            changes["detected_precision"] = result.detected_precision
        if result.architecture_params:
            if defn.architecture_params:
                # Harvested values win over stale YAML
                merged = {**defn.architecture_params, **result.architecture_params}

                # Log drift between YAML and harvested values
                for key in result.architecture_params:
                    yaml_val = defn.architecture_params.get(key)
                    if yaml_val is not None and yaml_val != result.architecture_params[key]:
                        logger.warning(
                            "config_drift_detected",
                            key=key,
                            yaml_value=yaml_val,
                            harvested_value=result.architecture_params[key],
                        )

                if merged != defn.architecture_params:
                    changes["architecture_params"] = merged
            else:
                changes["architecture_params"] = result.architecture_params
        if not defn.lora_targetable_modules and result.lora_targetable_modules:
            changes["lora_targetable_modules"] = result.lora_targetable_modules

        # Block topology — derive from architecture_params if not already set
        if not defn.block_topology:
            arch = {**(defn.architecture_params or {}), **(changes.get("architecture_params") or {})}
            topology = _derive_block_topology(defn.family, arch)
            if topology:
                changes["block_topology"] = topology

        if changes:
            cls.update_definition(definition_id, changes)
            try:
                cls.save_definition(definition_id)
                logger.info("definition_enriched", id=definition_id, fields=list(changes.keys()))
            except ValueError:
                # No file path known — definition might be in-memory only
                logger.debug("enrichment_not_persisted", id=definition_id)
        else:
            logger.debug("enrichment_skipped_already_populated", id=definition_id)

    @classmethod
    def list_models(cls) -> list[str]:
        """Return all registered model definition IDs."""
        return list(cls._definitions.keys())

# ── Block Topology Derivation ────────────────────────────────────────────

def _derive_block_topology(family: str, arch: dict) -> list[dict]:
    """Derive block topology from architecture_params for a family.

    Uses ``num_layers`` / ``num_single_layers`` (Flux) or UNet block
    counts (SDXL) from the architecture params to build the metadata
    the frontend needs for block-swap sliders.
    """
    if family in ("flux2", "flux1"):
        topology = []
        num_double = (
            arch.get("transformer.num_layers")
            or arch.get("depth")
            or arch.get("num_layers")
        )
        num_single = (
            arch.get("transformer.num_single_layers")
            or arch.get("depth_single_blocks")
            or arch.get("num_single_layers")
        )
        if num_double:
            topology.append({
                "name": "double_blocks",
                "attr_path": "transformer_blocks",
                "count": int(num_double),
                "approx_vram_mb": 640,
            })
        if num_single:
            topology.append({
                "name": "single_blocks",
                "attr_path": "single_transformer_blocks",
                "count": int(num_single),
                "approx_vram_mb": 320,
            })
        return topology

    if family == "sdxl":
        return [
            {"name": "down_blocks", "attr_path": "down_blocks", "count": 4, "approx_vram_mb": 200},
            {"name": "mid_block", "attr_path": "mid_block", "count": 1, "approx_vram_mb": 100},
            {"name": "up_blocks", "attr_path": "up_blocks", "count": 4, "approx_vram_mb": 200},
        ]

    if family in ("qwen_image", "zimage"):
        num_blocks = arch.get("transformer.num_layers") or arch.get("num_layers") or 24
        return [
            {"name": "transformer_blocks", "attr_path": "transformer_blocks",
             "count": int(num_blocks), "approx_vram_mb": 400},
        ]

    return []


# Global registry instance
registry = ModelRegistry()
