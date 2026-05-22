"""DatasetManager broadcasts entity.changed on mutations."""
import asyncio
import os
from unittest.mock import patch, AsyncMock

import pytest

from app.core.dataset_manager import DatasetManager


@pytest.fixture
def dataset_manager_with_loop(tmp_path):
    """Build a DatasetManager pointed at a throwaway default_root.

    The session-wide ``_isolate_test_db`` fixture already redirects the
    SQLite singleton to a temp DB, so we just need to give the manager a
    writable filesystem root for created datasets.
    """
    storage_file = str(tmp_path / "dataset_locations.json")
    default_root = str(tmp_path / "datasets")
    os.makedirs(default_root, exist_ok=True)

    # Bypass __init__'s repo-relative path resolution; supply absolute paths.
    mgr = DatasetManager.__new__(DatasetManager)
    mgr.root_dir = str(tmp_path)
    mgr.storage_file = storage_file
    mgr.default_root = default_root

    from app.core.settings_manager import get_settings_manager
    mgr.settings_manager = get_settings_manager()
    mgr.datasets = {}
    mgr._loop = asyncio.get_event_loop()

    from app.core.db import DatabaseEngine
    from app.core.db.repositories.dataset_repo import DatasetRepository
    from app.core.db.repositories.media_item_repo import MediaItemRepository
    mgr._db = DatabaseEngine.get_instance()
    mgr._dataset_repo = DatasetRepository()
    mgr._media_repo = MediaItemRepository()

    return mgr


@pytest.mark.asyncio
async def test_create_dataset_broadcasts_entity_changed(dataset_manager_with_loop):
    mgr = dataset_manager_with_loop
    mock_broadcast = AsyncMock()

    with patch("app.core.dataset_manager.event_manager.broadcast", mock_broadcast):
        ds = mgr.create_dataset(name="emit_create_ds", description="hi", classifier="cls")
        # broadcast is scheduled via run_coroutine_threadsafe; yield so it runs.
        await asyncio.sleep(0.05)

    entity_calls = [
        c for c in mock_broadcast.await_args_list
        if c.args and c.args[0] == "entity.changed"
    ]
    created = [c for c in entity_calls if c.args[1]["op"] == "created"]
    assert len(created) == 1, (
        f"expected one created event, got {len(created)} from {entity_calls}"
    )
    envelope = created[0].args[1]
    assert envelope["entity"] == "dataset"
    assert envelope["id"] == ds.id
    assert envelope["payload"]["id"] == ds.id
    assert envelope["payload"]["name"] == "emit_create_ds"


@pytest.mark.asyncio
async def test_update_dataset_broadcasts_entity_changed(dataset_manager_with_loop):
    mgr = dataset_manager_with_loop
    # Seed: create_dataset already emits a 'created' event; reset broadcast after.
    mgr.create_dataset(name="emit_update_ds", description="d", classifier="")

    mock_broadcast = AsyncMock()
    with patch("app.core.dataset_manager.event_manager.broadcast", mock_broadcast):
        updated = mgr.update_dataset(
            current_name="emit_update_ds",
            new_name="emit_update_ds",
            new_description="changed",
            new_classifier="newcls",
        )
        await asyncio.sleep(0.05)

    entity_calls = [
        c for c in mock_broadcast.await_args_list
        if c.args and c.args[0] == "entity.changed"
    ]
    updates = [c for c in entity_calls if c.args[1]["op"] == "updated"]
    assert len(updates) == 1, (
        f"expected one updated event, got {len(updates)} from {entity_calls}"
    )
    envelope = updates[0].args[1]
    assert envelope["entity"] == "dataset"
    assert envelope["id"] == updated.id
    assert envelope["payload"]["description"] == "changed"
    assert envelope["payload"]["classifier"] == "newcls"


@pytest.mark.asyncio
async def test_delete_dataset_broadcasts_entity_changed(dataset_manager_with_loop):
    mgr = dataset_manager_with_loop
    ds = mgr.create_dataset(name="emit_delete_ds", description="d", classifier="")

    mock_broadcast = AsyncMock()
    with patch("app.core.dataset_manager.event_manager.broadcast", mock_broadcast):
        mgr.delete_dataset("emit_delete_ds")
        await asyncio.sleep(0.05)

    entity_calls = [
        c for c in mock_broadcast.await_args_list
        if c.args and c.args[0] == "entity.changed"
    ]
    deletes = [c for c in entity_calls if c.args[1]["op"] == "deleted"]
    assert len(deletes) == 1, (
        f"expected one deleted event, got {len(deletes)} from {entity_calls}"
    )
    envelope = deletes[0].args[1]
    assert envelope["entity"] == "dataset"
    assert envelope["id"] == ds.id
    assert envelope["payload"] is None
