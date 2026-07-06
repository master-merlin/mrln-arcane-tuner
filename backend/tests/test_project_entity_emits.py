"""Project CRUD + dataset-association endpoints broadcast entity.changed
with entity='project' (B-ARCH-4).

Follows the house pattern from test_definition_routes_registry_emits.py:
patch `app.api.project_routes.event_manager.broadcast` with an AsyncMock
and assert on the entity.changed calls that flow through it.
"""
from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@contextlib.contextmanager
def _isolated_db(tmp_path):
    from app.core.db.engine import DatabaseEngine

    prev = DatabaseEngine._instance
    eng = DatabaseEngine(db_path=str(tmp_path / "projects.db"))
    eng.initialize()
    DatabaseEngine._instance = eng
    try:
        yield eng
    finally:
        eng.close()
        DatabaseEngine._instance = prev


def _entity_calls(mock_broadcast, *, op: str | None = None):
    calls = [
        c for c in mock_broadcast.await_args_list
        if c.args and c.args[0] == "entity.changed" and c.args[1]["entity"] == "project"
    ]
    if op is not None:
        calls = [c for c in calls if c.args[1]["op"] == op]
    return calls


@pytest.mark.asyncio
async def test_create_project_broadcasts_created(tmp_path):
    with _isolated_db(tmp_path):
        mock_broadcast = AsyncMock()
        transport = ASGITransport(app=app)
        with patch("app.api.project_routes.event_manager.broadcast", mock_broadcast):
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/api/projects", json={"name": "Emits Alpha"})
        assert resp.status_code == 201, resp.text
        pid = resp.json()["id"]

        created = _entity_calls(mock_broadcast, op="created")
        assert len(created) == 1, created
        env = created[0].args[1]
        assert env["id"] == pid
        assert env["payload"]["name"] == "Emits Alpha"


@pytest.mark.asyncio
async def test_update_project_broadcasts_updated(tmp_path):
    with _isolated_db(tmp_path):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            created = (await ac.post("/api/projects", json={"name": "Emits Beta"})).json()
            pid = created["id"]

            mock_broadcast = AsyncMock()
            with patch("app.api.project_routes.event_manager.broadcast", mock_broadcast):
                resp = await ac.patch(f"/api/projects/{pid}", json={"description": "hi"})
        assert resp.status_code == 200, resp.text

        updated = _entity_calls(mock_broadcast, op="updated")
        assert len(updated) == 1, updated
        env = updated[0].args[1]
        assert env["id"] == pid
        assert env["payload"]["description"] == "hi"


@pytest.mark.asyncio
async def test_delete_project_broadcasts_deleted(tmp_path):
    with _isolated_db(tmp_path):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            created = (await ac.post("/api/projects", json={"name": "Emits Gamma"})).json()
            pid = created["id"]

            mock_broadcast = AsyncMock()
            with patch("app.api.project_routes.event_manager.broadcast", mock_broadcast):
                resp = await ac.delete(f"/api/projects/{pid}")
        assert resp.status_code == 204, resp.text

        deleted = _entity_calls(mock_broadcast, op="deleted")
        assert len(deleted) == 1, deleted
        env = deleted[0].args[1]
        assert env["id"] == pid
        assert env["payload"] is None


@pytest.mark.asyncio
async def test_dataset_association_change_broadcasts_project_updated(tmp_path):
    """Membership changes (add/remove a dataset) count as project updates."""
    with _isolated_db(tmp_path) as eng:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            created = (await ac.post("/api/projects", json={"name": "Emits Delta"})).json()
            pid = created["id"]

            with eng.write() as conn:
                conn.execute(
                    "INSERT INTO datasets (id, name, path, created_at) "
                    "VALUES ('ds-emit-1', 'emitds', '/tmp/emitds', 1.0)"
                )

            mock_broadcast = AsyncMock()
            with patch("app.api.project_routes.event_manager.broadcast", mock_broadcast):
                add_resp = await ac.post(
                    f"/api/projects/{pid}/datasets", json={"dataset_id": "ds-emit-1"},
                )
            assert add_resp.status_code == 201, add_resp.text
            add_updates = _entity_calls(mock_broadcast, op="updated")
            assert len(add_updates) == 1, add_updates
            assert add_updates[0].args[1]["id"] == pid

            mock_broadcast2 = AsyncMock()
            with patch("app.api.project_routes.event_manager.broadcast", mock_broadcast2):
                rm_resp = await ac.delete(f"/api/projects/{pid}/datasets/ds-emit-1")
            assert rm_resp.status_code == 204, rm_resp.text
            rm_updates = _entity_calls(mock_broadcast2, op="updated")
            assert len(rm_updates) == 1, rm_updates
            assert rm_updates[0].args[1]["id"] == pid


@pytest.mark.asyncio
async def test_remove_nonexistent_dataset_association_emits_nothing(tmp_path):
    """Removing an association that never existed is a 204 no-op — it must
    not broadcast a spurious project-updated event (BL1-8)."""
    with _isolated_db(tmp_path):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            created = (await ac.post("/api/projects", json={"name": "Emits Epsilon"})).json()
            pid = created["id"]

            mock_broadcast = AsyncMock()
            with patch("app.api.project_routes.event_manager.broadcast", mock_broadcast):
                rm_resp = await ac.delete(f"/api/projects/{pid}/datasets/never-existed")
            assert rm_resp.status_code == 204, rm_resp.text
            assert _entity_calls(mock_broadcast) == []
