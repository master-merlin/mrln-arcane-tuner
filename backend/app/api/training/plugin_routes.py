"""Plugin discovery routes."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from typing import Any

from app.core.plugin_manager import plugin_manager

router = APIRouter()


@router.get("/plugins", response_model=list[dict[str, str]])
async def list_plugins():
    """List all available training plugins."""
    return await asyncio.to_thread(plugin_manager.list_plugins)


@router.get("/plugins/{model_id}/schema", response_model=dict[str, Any])
async def get_plugin_schema(model_id: str, project_id: str | None = None):
    """Return the JSON configuration schema for a plugin.

    ``project_id`` scopes dynamic options (the dataset dropdown) to a project;
    omit it for the global scope, which offers every dataset.
    """
    plugin = await asyncio.to_thread(plugin_manager.get_plugin, model_id)
    if not plugin:
        raise HTTPException(status_code=404, detail=f"Plugin '{model_id}' not found")

    schema = plugin.get_config_schema().model_json_schema()
    schema = await asyncio.to_thread(plugin.enrich_schema, schema, project_id)
    return schema
