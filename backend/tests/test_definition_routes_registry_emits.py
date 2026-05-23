"""Set/delete model source override endpoints broadcast entity.changed
with entity='registry_model'.

The frontend RegistryStore reconciles per-definition source overrides off
these events. The emission happens directly inside the async route
handlers (no run_coroutine_threadsafe needed — we're already on the loop).
"""
from __future__ import annotations

from unittest.mock import patch, AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.engine.core.definitions import ModelDefinition
from app.engine.models.registry import registry
from app.engine.utils.model_override_manager import ModelOverrideManager
from app.main import app


@pytest.fixture
def seeded_definition():
    """Seed the registry with a throwaway definition so set/delete have a
    target. Cleans up registry + override state on exit so we don't leak
    a junk override into the user's settings.json.
    """
    saved_defs = dict(registry._definitions)
    saved_paths = dict(registry._paths)

    fake_id = "__registry_emit_test"
    try:
        registry._definitions[fake_id] = ModelDefinition(
            id=fake_id,
            name=fake_id,
            family="sdxl",
            components={},
        )
        registry._paths[fake_id] = f"/tmp/{fake_id}.yaml"
        yield fake_id
    finally:
        try:
            ModelOverrideManager.delete_override(fake_id)
        except Exception:
            pass
        registry._definitions.clear()
        registry._paths.clear()
        registry._definitions.update(saved_defs)
        registry._paths.update(saved_paths)


@pytest.mark.asyncio
async def test_set_model_source_broadcasts_updated(seeded_definition):
    """PUT /api/models/definitions/{id}/source emits entity.changed:updated
    (entity='registry_model', id=definition_id, payload=override dict).
    """
    fake_id = seeded_definition
    mock_broadcast = AsyncMock()

    body = {
        "source_type": "hf_hub",
        "local_path": None,
        "skip_update": True,
    }

    transport = ASGITransport(app=app)
    with patch(
        "app.api.training.definition_routes.event_manager.broadcast",
        mock_broadcast,
    ):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.put(
                f"/api/models/definitions/{fake_id}/source",
                json=body,
            )

    assert resp.status_code == 200, resp.text

    entity_calls = [
        c for c in mock_broadcast.await_args_list
        if c.args and c.args[0] == "entity.changed"
    ]
    updates = [
        c for c in entity_calls
        if c.args[1]["op"] == "updated"
        and c.args[1]["entity"] == "registry_model"
    ]
    assert len(updates) == 1, (
        f"expected one updated event, got {len(updates)} from {entity_calls}"
    )
    env = updates[0].args[1]
    assert env["id"] == fake_id
    assert env["payload"] is not None
    assert env["payload"]["source_type"] == "hf_hub"
    assert env["payload"]["skip_update"] is True


@pytest.mark.asyncio
async def test_delete_model_source_broadcasts_deleted(seeded_definition):
    """DELETE /api/models/definitions/{id}/source emits entity.changed:deleted
    with payload=None.
    """
    fake_id = seeded_definition
    mock_broadcast = AsyncMock()

    transport = ASGITransport(app=app)
    with patch(
        "app.api.training.definition_routes.event_manager.broadcast",
        mock_broadcast,
    ):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.delete(
                f"/api/models/definitions/{fake_id}/source",
            )

    assert resp.status_code == 200, resp.text

    entity_calls = [
        c for c in mock_broadcast.await_args_list
        if c.args and c.args[0] == "entity.changed"
    ]
    deletes = [
        c for c in entity_calls
        if c.args[1]["op"] == "deleted"
        and c.args[1]["entity"] == "registry_model"
    ]
    assert len(deletes) == 1, (
        f"expected one deleted event, got {len(deletes)} from {entity_calls}"
    )
    env = deletes[0].args[1]
    assert env["id"] == fake_id
    assert env["payload"] is None
