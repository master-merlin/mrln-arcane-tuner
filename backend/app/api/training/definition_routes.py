"""Model definition CRUD, enrichment, capabilities, and VRAM estimation routes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from app.core.events import emit_entity_change, event_manager
from app.core.logger import get_logger
from app.core.schemas.model_overrides import ModelOverride
from app.engine.utils.model_override_manager import ModelOverrideManager
from app.api.schemas.definition_schemas import (
    CreateDefinitionRequest,
    UpdateDefinitionRequest,
    VRAMEstimateRequest,
)

router = APIRouter()
logger = get_logger(__name__)


@router.get("/models/definitions", response_model=list[dict[str, Any]])
async def list_model_definitions():
    """List all available model definitions with introspection data."""
    from app.engine.models.registry import registry

    # R-API-07: batch-load overrides once at the top instead of calling
    # ModelOverrideManager.get_override per definition (was N+1 settings.json reads).
    all_settings = await ModelOverrideManager.get_all_async()
    results = []
    for def_id, defn in registry._definitions.items():
        data = defn.model_dump()
        data["component_paths"] = {
            k: v.get("path", "") for k, v in data.get("components", {}).items()
        }
        override = all_settings.overrides.get(def_id)
        data["source_override"] = override.model_dump() if override else None
        results.append(data)
    return results


@router.post("/models/definitions", response_model=dict[str, Any])
async def create_definition(request: CreateDefinitionRequest):
    """Create a new model definition YAML file."""
    from app.engine.models.registry import registry
    import yaml

    if registry.get_definition(request.id):
        raise HTTPException(status_code=409, detail=f"Definition '{request.id}' already exists.")

    families_dir = (
        Path(__file__).resolve().parents[2] / "engine" / "models" / "families"
    )
    family_def_dir = families_dir / request.family / "definitions"

    def _write_yaml():
        family_def_dir.mkdir(parents=True, exist_ok=True)
        safe_id = request.id.replace("/", "_").replace("\\", "_")
        yaml_path = family_def_dir / f"{safe_id}.yaml"
        data = request.model_dump()
        yaml_path.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")
        return str(yaml_path)

    yaml_path = await asyncio.to_thread(_write_yaml)

    defn = await asyncio.to_thread(registry.load_definition, yaml_path)
    logger.info("definition_created", id=defn.id, family=defn.family, path=yaml_path)
    return defn.model_dump()


@router.put("/models/definitions/{definition_id}", response_model=dict[str, Any])
async def update_definition(definition_id: str, request: UpdateDefinitionRequest):
    """Update an existing model definition (partial update)."""
    from app.engine.models.registry import registry

    defn = registry.get_definition(definition_id)
    if not defn:
        raise HTTPException(status_code=404, detail=f"Definition '{definition_id}' not found.")

    changes = request.model_dump(exclude_none=True)
    if not changes:
        return defn.model_dump()

    registry.update_definition(definition_id, changes)
    await asyncio.to_thread(registry.save_definition, definition_id)
    logger.info("definition_updated", id=definition_id, changed_fields=list(changes.keys()))
    return registry.get_definition(definition_id).model_dump()


@router.delete("/models/definitions/{definition_id}")
async def delete_definition(definition_id: str):
    """Delete a model definition (removes YAML file and registry entry)."""
    from app.engine.models.registry import registry

    defn = registry.get_definition(definition_id)
    if not defn:
        raise HTTPException(status_code=404, detail=f"Definition '{definition_id}' not found.")

    yaml_path = registry._paths.get(definition_id)
    if yaml_path:
        p = Path(yaml_path)
        if p.exists():
            await asyncio.to_thread(p.unlink)

    del registry._definitions[definition_id]
    if definition_id in registry._paths:
        del registry._paths[definition_id]

    # Cascade: remove any source override for this definition
    await ModelOverrideManager.delete_override_async(definition_id)

    logger.info("definition_deleted", id=definition_id)
    return {"status": "deleted", "id": definition_id}


# ── VRAM Estimation ─────────────────────────────────────────────────────


@router.post("/jobs/estimate-vram")
async def estimate_vram(request: VRAMEstimateRequest):
    """Estimate peak VRAM for a training configuration before launching.

    Returns a per-category breakdown and a fit assessment against available GPU VRAM.
    """
    from app.engine.models.registry import registry
    from app.engine.utils.vram_estimator import VRAMEstimator

    defn = registry._definitions.get(request.definition_id)
    if not defn:
        raise HTTPException(status_code=404, detail=f"Definition '{request.definition_id}' not found")

    report = VRAMEstimator.estimate(defn, request.config)
    return report.to_dict()


# ── Model Capabilities ──────────────────────────────────────────────────


@router.get("/models/capabilities/{definition_id}")
async def get_model_capabilities(definition_id: str):
    """Return block topology and trainable layer names for a model definition."""
    from app.engine.models.registry import registry

    defn = registry.get_definition(definition_id)
    if not defn:
        raise HTTPException(
            status_code=404,
            detail=f"Definition '{definition_id}' not found",
        )

    has_topology = bool(defn.block_topology)
    return {
        "enriched": has_topology,
        "block_topology": defn.block_topology,
        "lora_targetable_modules": defn.lora_targetable_modules,
        "trainable_layers": [],
    }


@router.post("/models/definitions/{definition_id}/enrich")
async def enrich_definition(definition_id: str):
    """Trigger enrichment for a model definition.

    Re-runs introspection + config harvesting and populates
    ``block_topology``, ``lora_targetable_modules``, etc.
    """
    from app.engine.models.registry import registry

    defn = registry.get_definition(definition_id)
    if not defn:
        raise HTTPException(
            status_code=404,
            detail=f"Definition '{definition_id}' not found",
        )

    # Clear block_topology to force re-derivation
    registry.update_definition(definition_id, {"block_topology": []})

    components = {k: v.model_dump() for k, v in defn.components.items()}
    repo_comp = defn.components.get("repo")
    root_path = repo_comp.path if repo_comp else None

    await asyncio.to_thread(
        registry.enrich_definition, definition_id, components, root_path,
    )

    updated = registry.get_definition(definition_id)
    return {
        "status": "enriched",
        "id": definition_id,
        "block_topology": updated.block_topology if updated else [],
        "lora_targetable_modules": updated.lora_targetable_modules if updated else [],
    }


# ── Global Model Settings ───────────────────────────────────────────────


@router.get("/models/settings")
async def get_model_settings():
    """Get global model settings (offline mode, default path)."""
    settings = await ModelOverrideManager.get_all_async()
    return {
        "global_offline_mode": settings.global_offline_mode,
        "default_model_path": settings.default_model_path,
    }


@router.put("/models/settings")
async def update_model_settings(body: dict[str, Any]):
    """Update global model settings."""
    settings = await ModelOverrideManager.get_all_async()

    if "global_offline_mode" in body:
        settings.global_offline_mode = bool(body["global_offline_mode"])

    if "default_model_path" in body:
        path_str = body["default_model_path"].strip()
        if path_str and not Path(path_str).is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Path does not exist: {path_str}",
            )
        settings.default_model_path = path_str

    await ModelOverrideManager._save_async(settings)
    logger.info(
        "model_settings_updated",
        offline=settings.global_offline_mode,
        path=settings.default_model_path,
    )
    return {
        "global_offline_mode": settings.global_offline_mode,
        "default_model_path": settings.default_model_path,
    }


# ── Model Source Overrides ──────────────────────────────────────────────


@router.get("/models/definitions/{definition_id}/source")
async def get_model_source(definition_id: str):
    """Get the source override for a model definition."""
    override = await ModelOverrideManager.get_override_async(definition_id)
    if override:
        return override.model_dump()
    return {"source_type": "hf_hub", "local_path": None, "skip_update": False}


@router.put("/models/definitions/{definition_id}/source")
async def set_model_source(definition_id: str, override: ModelOverride):
    """Set a source override for a model definition."""
    from app.engine.models.registry import registry

    defn = registry.get_definition(definition_id)
    if not defn:
        raise HTTPException(
            status_code=404,
            detail=f"Definition '{definition_id}' not found.",
        )

    # Validate local paths exist
    if override.local_path:
        p = Path(override.local_path)
        if not p.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Path does not exist: {override.local_path}",
            )

    # Safetensors mode requires enriched YAML
    if override.source_type == "local_safetensors":
        if not defn.architecture_params:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Local Safetensors mode requires a fully enriched "
                    "model definition (architecture_params must be populated). "
                    "Enrich via HF first."
                ),
            )

    await ModelOverrideManager.set_override_async(definition_id, override)
    logger.info(
        "model_source_updated",
        id=definition_id,
        source=override.source_type,
    )
    # Broadcast for the frontend RegistryStore. We're already on the event
    # loop here, so a direct await is fine — no run_coroutine_threadsafe.
    await emit_entity_change(
        event_manager.broadcast,
        entity="registry_model",
        op="updated",
        id=definition_id,
        payload=override.model_dump(mode="json"),
    )
    return override.model_dump()


@router.delete("/models/definitions/{definition_id}/source")
async def delete_model_source(definition_id: str):
    """Remove source override — revert to YAML default."""
    await ModelOverrideManager.delete_override_async(definition_id)
    logger.info("model_source_override_removed", id=definition_id)
    await emit_entity_change(
        event_manager.broadcast,
        entity="registry_model",
        op="deleted",
        id=definition_id,
        payload=None,
    )
    return {"status": "removed", "id": definition_id}


@router.post("/models/definitions/{definition_id}/validate-path")
async def validate_model_path(
    definition_id: str,
    body: dict[str, Any],
):
    """Probe a local path to determine its type and available components."""
    local_path = body.get("path", "")
    if not local_path:
        raise HTTPException(status_code=400, detail="path is required")

    p = Path(local_path)

    def _probe() -> dict[str, Any]:
        result: dict[str, Any] = {
            "valid": False,
            "type": "unknown",
            "components_found": [],
            "warnings": [],
        }

        if not p.exists():
            return result

        result["valid"] = True

        if p.is_file() and p.suffix == ".safetensors":
            result["type"] = "safetensors"
            result["components_found"] = [p.stem]
            return result

        if p.is_dir():
            has_model_index = (p / "model_index.json").is_file()
            known_subdirs = [
                "transformer", "unet", "vae", "text_encoder",
                "text_encoder_2", "tokenizer", "tokenizer_2",
                "scheduler", "ae",
            ]
            found = [d for d in known_subdirs if (p / d).is_dir()]

            safetensors_files = list(p.glob("*.safetensors"))

            if has_model_index or len(found) >= 2:
                result["type"] = "diffusers"
                result["components_found"] = found
            elif safetensors_files:
                result["type"] = "safetensors"
                result["components_found"] = [f.stem for f in safetensors_files]
                result["warnings"].append(
                    "Raw safetensors detected. Ensure all required "
                    "components are present and the model definition "
                    "has been enriched with architecture_params."
                )
            else:
                result["warnings"].append(
                    "Directory exists but no model files detected."
                )

        return result

    return await asyncio.to_thread(_probe)
