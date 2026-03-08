"""Model definition CRUD, enrichment, capabilities, and VRAM estimation routes."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import APIRouter, HTTPException

from app.core.logger import get_logger
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

    results = []
    for def_id, defn in registry._definitions.items():
        data = defn.model_dump()
        data["component_paths"] = {
            k: v.get("path", "") for k, v in data.get("components", {}).items()
        }
        results.append(data)
    return results


@router.post("/models/definitions", response_model=dict[str, Any])
async def create_definition(request: CreateDefinitionRequest):
    """Create a new model definition YAML file."""
    from app.engine.models.registry import registry
    import yaml

    if registry.get_definition(request.id):
        raise HTTPException(status_code=409, detail=f"Definition '{request.id}' already exists.")

    families_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "engine", "models", "families",
    )
    family_def_dir = os.path.join(families_dir, request.family, "definitions")
    os.makedirs(family_def_dir, exist_ok=True)

    safe_id = request.id.replace("/", "_").replace("\\", "_")
    yaml_path = os.path.join(family_def_dir, f"{safe_id}.yaml")

    data = request.model_dump()
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    defn = registry.load_definition(yaml_path)
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
    registry.save_definition(definition_id)
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
    if yaml_path and os.path.exists(yaml_path):
        os.remove(yaml_path)

    del registry._definitions[definition_id]
    if definition_id in registry._paths:
        del registry._paths[definition_id]

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
