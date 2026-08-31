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


def _roundtrip_yaml():
    """A ruamel.yaml round-trip instance for comment-preserving saves.

    Floats get an explicit mantissa dot in scientific notation (``1.0e-05``,
    not ruamel's default ``1e-05``) — YAML 1.1 loaders like PyYAML otherwise
    re-read them as *strings*, silently corrupting values such as
    ``norm_eps: 1e-05`` on the next load.
    """
    import math

    from ruamel.yaml import YAML

    ryaml = YAML()  # round-trip mode: keeps comments, quotes, key order
    ryaml.preserve_quotes = True
    ryaml.width = 4096  # never re-wrap the authors' long lines

    def _float_representer(representer, value):
        if math.isnan(value):
            text = ".nan"
        elif math.isinf(value):
            text = ".inf" if value > 0 else "-.inf"
        else:
            text = repr(value)
            mantissa, e, exponent = text.partition("e")
            if e and "." not in mantissa:
                text = f"{mantissa}.0e{exponent}"
        return representer.represent_scalar("tag:yaml.org,2002:float", text)

    ryaml.representer.add_representer(float, _float_representer)
    return ryaml


def _merge_mapping(target, source: dict) -> None:
    """Deep-merge ``source`` into a ruamel CommentedMap ``target`` in place.

    Only differing leaves are assigned and extra keys are dropped, so after
    the merge ``target`` parses equal to ``source`` — but comments sitting on
    UNCHANGED sibling keys survive (a wholesale ``target = source`` replace
    would discard every comment inside the mapping).
    """
    for key in [k for k in target if k not in source]:
        del target[key]
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _merge_mapping(target[key], value)
        elif key not in target or target[key] != value:
            target[key] = value


class ModelRegistry:
    """Central registry for model families and YAML-based definitions."""

    _families: dict[str, type[ModelFamily]] = {}
    _definitions: dict[str, ModelDefinition] = {}
    _paths: dict[str, str] = {}
    _discovered = False
    _definitions_loaded = False


    @classmethod
    def _central_definitions_dir(cls) -> str:
        """Central (non family-scoped) definitions directory.

        Anchored on this module, like the family scan below. It used to be
        ``os.getcwd() / "app/engine/models/definitions"``, which only ever
        resolved when the process happened to be launched from ``backend\\`` —
        a service, a different launcher or a test run would silently scan
        nothing and load no central definitions, with no error to show for it.
        """
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "definitions")

    @classmethod
    def register_family(cls, name: str, family_class: type[ModelFamily]) -> None:
        """Register a model family class under the given name."""
        cls._families[name] = family_class

    @classmethod
    def initialize(cls) -> None:
        """One-stop initialization: discover families then load definitions."""
        cls.discover_families()
        cls.load_definitions(cls._central_definitions_dir())

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
        defs_dir = definitions_dir or cls._central_definitions_dir()

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
        """Persist definition back to its YAML file, preserving hand-written
        comments and key order.

        The file is round-tripped through ruamel.yaml and only keys whose
        values actually changed are rewritten; keys absent from the file are
        appended only when they differ from the field default. A plain
        ``yaml.dump(model_dump())`` rewrite (the old behavior) destroyed the
        authors' comments and exploded every default field into the file.
        """
        if definition_id not in cls._definitions:
            raise ValueError(f"Definition {definition_id} not found.")

        config = cls._definitions[definition_id]
        path = cls._paths.get(definition_id)

        if not path:
             raise ValueError(f"No file path known for {definition_id}. Cannot save.")

        ryaml = _roundtrip_yaml()

        doc = None
        on_disk: dict = {}
        try:
            with open(path, encoding="utf-8") as f:
                doc = ryaml.load(f)
            with open(path, encoding="utf-8") as f:
                on_disk = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError, Exception):  # noqa: BLE001 — any parse
            # failure falls through to the full-dump path below
            doc = None

        if doc is None:
            # Missing/unreadable file — nothing to preserve, write it whole.
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(config.model_dump(), f, sort_keys=False)
            return

        dumped = config.model_dump()
        fields = type(config).model_fields
        for key, value in dumped.items():
            if key in on_disk:
                if on_disk[key] != value:
                    if isinstance(doc.get(key), dict) and isinstance(value, dict):
                        # Deep-merge so comments on unchanged nested keys
                        # (e.g. inside architecture_params) survive.
                        _merge_mapping(doc[key], value)
                    else:
                        doc[key] = value
            else:
                # Only append keys that moved off their default — untouched
                # defaults must not leak into the hand-written file.
                field = fields.get(key)
                default = (
                    field.get_default(call_default_factory=True)
                    if field is not None
                    else None
                )
                if value != default:
                    doc[key] = value

        with open(path, "w", encoding="utf-8") as f:
            ryaml.dump(doc, f)

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
                # Harvested values win over stale YAML — EXCEPT keys the
                # definition explicitly pins (deliberate divergence from the
                # checkpoint's own config, e.g. ace_step15's model-card
                # scheduler.shift=3.0 vs the repo scheduler_config's 1.0).
                pinned = set(defn.enrich_pinned_keys or [])
                merged = {**defn.architecture_params, **result.architecture_params}
                for key in pinned:
                    if key in defn.architecture_params:
                        merged[key] = defn.architecture_params[key]

                # Log drift between YAML and harvested values
                for key in result.architecture_params:
                    yaml_val = defn.architecture_params.get(key)
                    if yaml_val is not None and yaml_val != result.architecture_params[key]:
                        logger.warning(
                            "config_drift_detected",
                            key=key,
                            yaml_value=yaml_val,
                            harvested_value=result.architecture_params[key],
                            pinned=key in pinned,
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
        """Return ALL registered model definition IDs, gated ones included.

        This is the INTERNAL view. The registry-wide coverage sweeps (a VRAM
        entry per family, LoRA target lists, TE-loading contracts,
        ``resolve_capabilities``) enumerate it to catch a family that misses a
        surface, so it must never be filtered — that is the guard that makes
        ungating a definition safe later. Anything the USER chooses from wants
        :meth:`list_available_models` instead.
        """
        return list(cls._definitions.keys())

    # ── Availability gate (ECOSYSTEM §6 `unavailable_reason`, LANE-45) ────

    @classmethod
    def is_definition_available(cls, definition_id: str) -> bool:
        """True when *definition_id* may be offered to the user.

        An UNKNOWN id is reported available: this answers "is this definition
        gated", not "does it exist", and the callers that care about existence
        (the job seam, the VRAM route) already have their own 404/lookup path.
        Conflating the two would turn every typo into a gate message.
        """
        defn = cls._definitions.get(definition_id)
        return defn is None or not defn.unavailable_reason

    @classmethod
    def unavailable_reason(cls, definition_id: str) -> str | None:
        """The gate's stated reason for *definition_id*, or None if available."""
        defn = cls._definitions.get(definition_id)
        return defn.unavailable_reason if defn is not None else None

    @classmethod
    def available_definitions(cls) -> dict[str, ModelDefinition]:
        """The USER-FACING view of the registry, insertion order preserved.

        Every surface that enumerates definitions or families FOR THE USER
        builds on this; nothing re-derives ``if defn.unavailable_reason`` at a
        call site, because this project's most-repeated defect is one
        enumeration surface out of step with the others.
        """
        return {
            did: defn
            for did, defn in cls._definitions.items()
            if not defn.unavailable_reason
        }

    @classmethod
    def list_available_models(cls) -> list[str]:
        """Return the definition IDs the user may choose from."""
        return list(cls.available_definitions().keys())

    @classmethod
    def count(cls) -> int:
        """Return the number of registered model definitions.

        Public accessor for callers (e.g. the ``/system/health`` KPI rail)
        that only need the count — avoids reaching into the private
        ``_definitions`` dict directly.
        """
        return len(cls._definitions)

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
