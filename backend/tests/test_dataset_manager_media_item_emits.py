"""DatasetManager broadcasts entity.changed (entity='media_item') on
single-media-item mutations and deletions.

The emission lives inside ``_persist_media_item`` so every caller —
toggle_image_enabled, save_caption, mask routes, overlay routes, etc. —
gets the broadcast for free. ``delete_media_pair`` emits its own
``op='deleted'`` event since it bypasses the persist chokepoint.
"""
from __future__ import annotations

import asyncio
import os
import time

import pytest
from unittest.mock import patch, AsyncMock

from app.core.dataset_manager import Dataset, DatasetManager
from app.core.db.repositories.dataset_repo import DatasetRepository
from app.core.db.repositories.media_item_repo import MediaItemRepository


@pytest.fixture
def media_item_manager(tmp_path):
    """Build a DatasetManager with one dataset + media item, bound to the
    running event loop so emissions can actually run.

    The session-scope ``_isolate_test_db`` fixture pins DatabaseEngine to
    a temp DB, so the freshly constructed repos transparently hit it.
    """
    mgr = DatasetManager.__new__(DatasetManager)
    mgr.datasets = {}
    mgr._loop = asyncio.get_event_loop()
    mgr._dataset_repo = DatasetRepository()
    mgr._media_repo = MediaItemRepository()

    ds_id = f"ds-mi-emit-{time.time_ns()}"
    ds_name = f"mi-emit-{ds_id}"
    rel_path = "subdir/image.png"
    ds_path = tmp_path / ds_id
    os.makedirs(ds_path / "subdir", exist_ok=True)

    initial_meta = {
        "width": 512,
        "height": 512,
        "aspect_ratio": 1.0,
        "orientation": "square",
        "size_bytes": 1234,
        "has_caption": False,
        "is_video": False,
        "enabled": True,
    }
    ds = Dataset(
        id=ds_id,
        name=ds_name,
        path=str(ds_path),
        created_at=time.time(),
        multimedia_count=1,
        media_metadata={rel_path: dict(initial_meta)},
    )
    mgr.datasets[ds_name] = ds

    # Seed DB so persist's UPDATE path has a row to write to + delete's
    # DELETE has a row to remove.
    mgr._dataset_repo.upsert(ds.model_dump(exclude={"media_metadata"}))
    mgr._media_repo.upsert({"dataset_id": ds_id, "rel_path": rel_path, **initial_meta})

    # Touch the actual file so delete_media_pair's existence check passes.
    (ds_path / rel_path).write_bytes(b"\x89PNG\r\n\x1a\n")

    yield mgr, ds, rel_path

    # Best-effort cleanup
    try:
        mgr._media_repo.delete(ds_id, rel_path)
    except Exception:
        pass


@pytest.mark.asyncio
async def test_persist_media_item_broadcasts_updated(media_item_manager):
    """Mutating + persisting a single media item emits entity.changed:updated.

    Uses ``toggle_image_enabled`` as a representative caller — but the
    emission lives in ``_persist_media_item`` itself, so every caller is
    covered by this codepath.
    """
    mgr, ds, rel_path = media_item_manager

    mock_broadcast = AsyncMock()
    with patch("app.core.dataset_manager.event_manager.broadcast", mock_broadcast):
        mgr.toggle_image_enabled(ds.name, rel_path, False)
        await asyncio.sleep(0.05)  # let run_coroutine_threadsafe land

    entity_calls = [
        c for c in mock_broadcast.await_args_list
        if c.args and c.args[0] == "entity.changed"
    ]
    updates = [c for c in entity_calls if c.args[1]["op"] == "updated"]
    assert len(updates) == 1, (
        f"expected one updated event, got {len(updates)} from {entity_calls}"
    )
    env = updates[0].args[1]
    assert env["entity"] == "media_item"
    assert env["id"] == f"{ds.name}/{rel_path}"
    assert env["payload"]["media_file"] == rel_path
    assert env["payload"]["dataset_name"] == ds.name
    assert env["payload"]["enabled"] is False


@pytest.mark.asyncio
async def test_save_caption_broadcasts_via_persist(media_item_manager, tmp_path):
    """A second caller (save_caption) flows through the same chokepoint.

    Verifies the emission isn't toggle-specific — anything that updates
    media_metadata + calls _persist_media_item emits.
    """
    mgr, ds, rel_path = media_item_manager

    # save_caption writes a .txt next to the image; the dataset path was
    # created in the fixture, but we need the caption file's directory too.
    caption_file = os.path.splitext(rel_path)[0] + ".txt"
    full_caption = os.path.join(ds.path, caption_file)
    os.makedirs(os.path.dirname(full_caption), exist_ok=True)

    mock_broadcast = AsyncMock()
    with patch("app.core.dataset_manager.event_manager.broadcast", mock_broadcast):
        mgr.save_caption(ds.name, caption_file, "a fluffy cat")
        await asyncio.sleep(0.05)

    entity_calls = [
        c for c in mock_broadcast.await_args_list
        if c.args and c.args[0] == "entity.changed"
    ]
    updates = [
        c for c in entity_calls
        if c.args[1]["op"] == "updated" and c.args[1]["entity"] == "media_item"
    ]
    assert len(updates) == 1
    env = updates[0].args[1]
    assert env["id"] == f"{ds.name}/{rel_path}"
    assert env["payload"]["has_caption"] is True


@pytest.mark.asyncio
async def test_delete_media_pair_broadcasts_deleted(media_item_manager):
    """delete_media_pair emits entity.changed:deleted with payload=None."""
    mgr, ds, rel_path = media_item_manager

    mock_broadcast = AsyncMock()
    with patch("app.core.dataset_manager.event_manager.broadcast", mock_broadcast):
        mgr.delete_media_pair(ds.name, rel_path)
        await asyncio.sleep(0.05)

    entity_calls = [
        c for c in mock_broadcast.await_args_list
        if c.args and c.args[0] == "entity.changed"
    ]
    deletes = [
        c for c in entity_calls
        if c.args[1]["op"] == "deleted" and c.args[1]["entity"] == "media_item"
    ]
    assert len(deletes) == 1, (
        f"expected one deleted event, got {len(deletes)} from {entity_calls}"
    )
    env = deletes[0].args[1]
    assert env["id"] == f"{ds.name}/{rel_path}"
    assert env["payload"] is None
