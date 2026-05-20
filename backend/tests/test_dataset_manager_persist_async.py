"""R-API-07: DatasetManager._persist_media_item_async must offload
the sync sqlite write so FastAPI routes don't block the event loop.
"""
from __future__ import annotations

import time

import pytest

from app.core.dataset_manager import Dataset, DatasetManager
from app.core.db.repositories.dataset_repo import DatasetRepository
from app.core.db.repositories.media_item_repo import MediaItemRepository


@pytest.fixture
def dataset_manager_with_item(tmp_path):
    """Build a DatasetManager bound to the session test DB with one dataset + media item.

    The session-scope ``_isolate_test_db`` fixture in conftest.py pins
    DatabaseEngine._instance to a tmp DB, so a freshly constructed
    DatasetRepository / MediaItemRepository transparently hits it.

    We skip DatasetManager.__init__ to avoid the storage-file / disk dance
    and only wire up the attributes _persist_media_item touches.
    """
    mgr = DatasetManager.__new__(DatasetManager)
    mgr.datasets = {}
    mgr._loop = None
    mgr._dataset_repo = DatasetRepository()
    mgr._media_repo = MediaItemRepository()

    ds_id = f"ds-async-{time.time_ns()}"
    rel_path = "image.png"
    initial_meta = {
        "width": 512,
        "height": 512,
        "aspect_ratio": 1.0,
        "orientation": "square",
        "size_bytes": 1234,
        "has_caption": False,
        "is_video": False,
    }
    ds = Dataset(
        id=ds_id,
        name=f"async-test-{ds_id}",
        path=str(tmp_path / ds_id),
        created_at=time.time(),
        media_metadata={rel_path: dict(initial_meta)},
    )

    # Seed the DB so the UPDATE path inside _persist_media_item has a row
    # to write to (the method calls _media_repo.update, not upsert).
    mgr._dataset_repo.upsert(ds.model_dump(exclude={"media_metadata"}))
    mgr._media_repo.upsert({"dataset_id": ds_id, "rel_path": rel_path, **initial_meta})

    yield mgr, ds, rel_path

    # Cleanup so subsequent fixtures don't see stragglers
    mgr._media_repo.delete(ds_id, rel_path)


@pytest.mark.asyncio
async def test_persist_media_item_async_matches_sync(dataset_manager_with_item) -> None:
    """R-API-07: async variant must produce the same DB state as sync."""
    mgr, ds, rel_path = dataset_manager_with_item

    # Mutate metadata, persist sync, snapshot the row.
    ds.media_metadata[rel_path]["width"] = 768
    mgr._persist_media_item(ds, rel_path)
    sync_state = mgr._media_repo.get_by_path(ds.id, rel_path)

    # Mutate again, persist async, snapshot.
    ds.media_metadata[rel_path]["width"] = 768  # same value -> identical row expected
    await mgr._persist_media_item_async(ds, rel_path)
    async_state = mgr._media_repo.get_by_path(ds.id, rel_path)

    # The repo's _prepare() defaults added_at to time.time() on every write,
    # so that one field will always differ between two successive calls.
    # Compare every other field: the async path must produce identical state
    # to the sync path on the columns _persist_media_item actually controls.
    assert sync_state is not None
    assert async_state is not None
    sync_no_ts = {k: v for k, v in sync_state.items() if k != "added_at"}
    async_no_ts = {k: v for k, v in async_state.items() if k != "added_at"}
    assert sync_no_ts == async_no_ts
    assert async_state["width"] == 768


@pytest.mark.asyncio
async def test_persist_media_item_async_writes_row(dataset_manager_with_item) -> None:
    """R-API-07: async variant actually writes the row."""
    mgr, ds, rel_path = dataset_manager_with_item

    ds.media_metadata[rel_path]["height"] = 1024
    await mgr._persist_media_item_async(ds, rel_path)

    row = mgr._media_repo.get_by_path(ds.id, rel_path)
    assert row is not None
    assert row["height"] == 1024
