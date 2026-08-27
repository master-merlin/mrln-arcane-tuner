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
import pytest_asyncio
from unittest.mock import patch, AsyncMock

from app.core.dataset_manager import Dataset, DatasetManager
from app.core.db.repositories.dataset_repo import DatasetRepository
from app.core.db.repositories.media_item_repo import MediaItemRepository


@pytest_asyncio.fixture
async def media_item_manager(tmp_path):
    """Build a DatasetManager with one dataset + media item, bound to the
    running event loop so emissions can actually run.

    The session-scope ``_isolate_test_db`` fixture pins DatabaseEngine to
    a temp DB, so the freshly constructed repos transparently hit it.
    """
    mgr = DatasetManager.__new__(DatasetManager)
    mgr.datasets = {}
    # Async fixture: `get_running_loop()` binds to the loop the test awaits on.
    # 0.21 installed a current loop in the main thread; 1.x does not.
    mgr._loop = asyncio.get_running_loop()
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
async def test_save_caption_increments_caption_count_and_emits_dataset_event(
    media_item_manager,
):
    """save_caption: false→true transition increments dataset.caption_count
    and broadcasts entity.changed for the dataset; a repeat save on the
    same image is a no-op for both the counter and the dataset broadcast.

    Regression for the stale-Library-counter bug: previously
    ``save_caption`` only updated the media_item row + emitted a
    ``media_item`` event, so ``dataset.caption_count`` (the aggregate
    field used by DatasetStore in the library view) only refreshed on
    the next full scan.
    """
    mgr, ds, rel_path = media_item_manager
    assert ds.caption_count == 0
    assert ds.media_metadata[rel_path].get("has_caption") is False

    caption_file = os.path.splitext(rel_path)[0] + ".txt"
    full_caption = os.path.join(ds.path, caption_file)
    os.makedirs(os.path.dirname(full_caption), exist_ok=True)

    mock_broadcast = AsyncMock()
    with patch("app.core.dataset_manager.event_manager.broadcast", mock_broadcast):
        # First save: has_caption flips False → True.
        mgr.save_caption(ds.name, caption_file, "a fluffy cat")
        await asyncio.sleep(0.05)

        # Second save: has_caption stays True → True (overwrite).
        mgr.save_caption(ds.name, caption_file, "a fluffier cat")
        await asyncio.sleep(0.05)

    # ── caption_count must have moved by exactly 1, not 2. ──────────────
    assert ds.caption_count == 1, (
        f"caption_count should increment exactly once for first save, "
        f"then stay put on overwrite; got {ds.caption_count}"
    )
    assert ds.media_metadata[rel_path]["has_caption"] is True

    # ── Exactly one dataset-level entity.changed across both saves. ─────
    entity_calls = [
        c for c in mock_broadcast.await_args_list
        if c.args and c.args[0] == "entity.changed"
    ]
    dataset_updates = [
        c for c in entity_calls
        if c.args[1]["entity"] == "dataset" and c.args[1]["op"] == "updated"
    ]
    assert len(dataset_updates) == 1, (
        f"expected exactly one dataset 'updated' event across two saves, "
        f"got {len(dataset_updates)} from {entity_calls}"
    )
    env = dataset_updates[0].args[1]
    assert env["id"] == ds.id
    # Payload must carry the fresh caption_count so the frontend's
    # EntityStore can reconcile without a full refetch.
    assert env["payload"]["caption_count"] == 1
    assert env["payload"]["id"] == ds.id
    assert env["payload"]["name"] == ds.name
    # Payload must be the FULL dataset row, not a 3-field stub —
    # the frontend EntityStore upserts (replaces) on `updated`, so
    # a partial payload would wipe multimedia_count / preview_image
    # etc. on every caption save. Spot-check a few non-summary
    # fields that any subscriber would read.
    assert env["payload"]["multimedia_count"] == 1
    assert "media_metadata" in env["payload"]
    assert env["payload"]["path"] == ds.path

    # ── DB row reflects the new caption_count too (survives restart). ──
    row = mgr._dataset_repo.get_by_id(ds.id)
    assert row is not None
    assert row["caption_count"] == 1


@pytest.mark.asyncio
async def test_save_caption_recounts_when_media_metadata_lacks_has_caption(
    media_item_manager,
):
    """Legacy media_metadata entries written before build_media_entry started
    seeding ``has_caption`` lack the key entirely. The save path used to call
    ``meta.get("has_caption")`` (truthy=False) on every save and increment
    every time, so a single image with two saves would land at caption_count=2
    instead of 1. Counting truthy ``has_caption`` flags after each save makes
    the path idempotent across schema vintage.
    """
    mgr, ds, rel_path = media_item_manager
    # Mimic a legacy entry: pop has_caption entirely.
    ds.media_metadata[rel_path].pop("has_caption", None)
    ds.caption_count = 0

    caption_file = os.path.splitext(rel_path)[0] + ".txt"
    full_caption = os.path.join(ds.path, caption_file)
    os.makedirs(os.path.dirname(full_caption), exist_ok=True)

    mock_broadcast = AsyncMock()
    with patch("app.core.dataset_manager.event_manager.broadcast", mock_broadcast):
        mgr.save_caption(ds.name, caption_file, "first")
        await asyncio.sleep(0.05)
        mgr.save_caption(ds.name, caption_file, "second")
        await asyncio.sleep(0.05)

    assert ds.caption_count == 1, (
        f"two saves on a single image must yield caption_count=1, "
        f"got {ds.caption_count}"
    )


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
