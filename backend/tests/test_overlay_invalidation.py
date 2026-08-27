"""PR11 invalidation cross-check — overlay reconciliation on destructive edits
+ per-item events on enable-all.

When a base image's pixels change (crop / adjustment), the previously rendered
overlay PNG is stale but the *recipe* in ``overlays.json`` is still valid intent,
so the overlay is re-rendered from the recipe against the new pixels (or dropped
if it can't be re-rendered). On image deletion the overlay is cleaned up so it
doesn't orphan. ``enable_all_images`` emits a single coarse ``dataset.invalidated``
(clients reconcile via refreshDataset) rather than per-item events.
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock
from PIL import Image

from app.core.dataset_manager import Dataset, DatasetManager
from app.core.db.repositories.dataset_repo import DatasetRepository
from app.core.db.repositories.media_item_repo import MediaItemRepository


def _entity_events(mock_broadcast, *, entity: str, op: str):
    return [
        c.args[1]
        for c in mock_broadcast.await_args_list
        if c.args and c.args[0] == "entity.changed"
        and c.args[1]["entity"] == entity and c.args[1]["op"] == op
    ]


@pytest_asyncio.fixture
async def overlay_manager(tmp_path):
    """DatasetManager with one dataset containing two REAL images (so crop /
    adjust can open them), bound to the running loop so emissions fire."""
    mgr = DatasetManager.__new__(DatasetManager)
    mgr.datasets = {}
    # Async fixture: `get_running_loop()` binds to the loop the test awaits on.
    # 0.21 installed a current loop in the main thread; 1.x does not.
    mgr._loop = asyncio.get_running_loop()
    mgr._dataset_repo = DatasetRepository()
    mgr._media_repo = MediaItemRepository()

    ds_id = f"ds-ovl-{time.time_ns()}"
    ds_name = f"ovl-{ds_id}"
    ds_path = tmp_path / ds_id
    os.makedirs(ds_path, exist_ok=True)

    rel_a, rel_b = "a.png", "b.png"
    for rel in (rel_a, rel_b):
        Image.new("RGB", (512, 512), (120, 80, 40)).save(ds_path / rel)

    base_meta = {
        "width": 512, "height": 512, "aspect_ratio": 1.0, "orientation": "squared",
        "size_bytes": 100, "has_caption": False, "is_video": False, "enabled": True,
    }
    ds = Dataset(
        id=ds_id, name=ds_name, path=str(ds_path), created_at=time.time(),
        multimedia_count=2,
        media_metadata={rel_a: dict(base_meta), rel_b: dict(base_meta)},
    )
    mgr.datasets[ds_name] = ds
    mgr._dataset_repo.upsert(ds.model_dump(exclude={"media_metadata"}))
    for rel in (rel_a, rel_b):
        mgr._media_repo.upsert({"dataset_id": ds_id, "rel_path": rel, **base_meta})

    yield mgr, ds, rel_a, rel_b


def _seed_overlay(ds, rel_path, ops):
    """Create an overlay PNG + overlays.json recipe + metadata for an image."""
    stem = os.path.splitext(rel_path)[0]
    overlays_dir = os.path.join(ds.path, "overlays")
    os.makedirs(overlays_dir, exist_ok=True)
    Image.new("RGB", (512, 512), (10, 10, 10)).save(os.path.join(overlays_dir, f"{stem}.png"))
    with open(os.path.join(ds.path, "overlays.json"), "w", encoding="utf-8") as f:
        json.dump({rel_path: {"overlay_file": f"overlays/{stem}.png",
                              "created_at": "x", "operations": ops}}, f)
    ds.media_metadata[rel_path]["has_overlay"] = True
    ds.media_metadata[rel_path]["overlay_dimensions"] = [512, 512]


# ── Crop / adjust: re-render overlay from recipe ────────────────────────────

@pytest.mark.asyncio
async def test_crop_rerenders_overlay_from_recipe(overlay_manager):
    """A crop re-applies the overlay recipe to the cropped pixels and emits
    overlay/updated (rather than discarding the edit)."""
    mgr, ds, rel_a, _ = overlay_manager
    _seed_overlay(ds, rel_a, [{"type": "vignette", "enabled": True, "params": {"amount": 0.3}}])

    mock_broadcast = AsyncMock()
    fake = ((256, 256), "deadbeef", [{"type": "vignette", "enabled": True, "params": {"amount": 0.3}}])
    with patch("app.core.dataset_manager.event_manager.broadcast", mock_broadcast), \
         patch("app.core.dataset_manager.rerender_overlay_from_recipe", return_value=fake) as rr:
        mgr.crop_media(ds.name, rel_a, 256, 256)
        await asyncio.sleep(0.05)

    rr.assert_called_once()
    assert ds.media_metadata[rel_a]["has_overlay"] is True
    assert ds.media_metadata[rel_a]["overlay_dimensions"] == [256, 256]
    updated = _entity_events(mock_broadcast, entity="overlay", op="updated")
    assert len(updated) == 1
    assert updated[0]["payload"]["dimensions"] == [256, 256]
    assert updated[0]["payload"]["hash"] == "deadbeef"
    # No deletion event when the overlay survives.
    assert not _entity_events(mock_broadcast, entity="overlay", op="deleted")


@pytest.mark.asyncio
async def test_crop_drops_overlay_when_rerender_fails(overlay_manager):
    """If re-rendering raises (e.g. GPU OOM), the stale overlay is removed +
    an overlay/deleted event fires — the edit never fails on the overlay."""
    mgr, ds, rel_a, _ = overlay_manager
    _seed_overlay(ds, rel_a, [{"type": "vignette", "enabled": True, "params": {}}])
    overlay_png = os.path.join(ds.path, "overlays", "a.png")
    assert os.path.exists(overlay_png)

    mock_broadcast = AsyncMock()
    with patch("app.core.dataset_manager.event_manager.broadcast", mock_broadcast), \
         patch("app.core.dataset_manager.rerender_overlay_from_recipe",
               side_effect=RuntimeError("CUDA OOM")):
        mgr.crop_media(ds.name, rel_a, 256, 256)
        await asyncio.sleep(0.05)

    assert not os.path.exists(overlay_png)  # stale overlay removed
    assert json.loads(open(os.path.join(ds.path, "overlays.json")).read()) == {}
    assert len(_entity_events(mock_broadcast, entity="overlay", op="deleted")) == 1


@pytest.mark.asyncio
async def test_crop_without_overlay_emits_no_overlay_event(overlay_manager):
    """Cropping an image that has no overlay emits no overlay event."""
    mgr, ds, rel_a, _ = overlay_manager
    mock_broadcast = AsyncMock()
    with patch("app.core.dataset_manager.event_manager.broadcast", mock_broadcast):
        mgr.crop_media(ds.name, rel_a, 256, 256)
        await asyncio.sleep(0.05)
    assert not _entity_events(mock_broadcast, entity="overlay", op="updated")
    assert not _entity_events(mock_broadcast, entity="overlay", op="deleted")


# ── Delete: clean up the overlay ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_removes_overlay_and_emits(overlay_manager):
    """Deleting an image removes its overlay PNG + recipe and emits
    overlay/deleted (so the file/recipe don't orphan)."""
    mgr, ds, rel_a, _ = overlay_manager
    _seed_overlay(ds, rel_a, [{"type": "vignette", "enabled": True, "params": {}}])
    overlay_png = os.path.join(ds.path, "overlays", "a.png")

    mock_broadcast = AsyncMock()
    with patch("app.core.dataset_manager.event_manager.broadcast", mock_broadcast):
        mgr.delete_media_pair(ds.name, rel_a)
        await asyncio.sleep(0.05)

    assert not os.path.exists(overlay_png)
    assert "a.png" not in json.loads(open(os.path.join(ds.path, "overlays.json")).read())
    assert len(_entity_events(mock_broadcast, entity="overlay", op="deleted")) == 1


# ── Enable-all: per-item events ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_enable_all_emits_dataset_invalidated(overlay_manager):
    """enable_all_images flips every image and emits ONE coarse
    dataset.invalidated (clients reconcile via refreshDataset) — not per-item
    media_item/updated events. Replaces the old O(N) per-item emission."""
    mgr, ds, rel_a, rel_b = overlay_manager
    ds.media_metadata[rel_a]["enabled"] = False
    ds.media_metadata[rel_b]["enabled"] = False

    mock_broadcast = AsyncMock()
    with patch("app.core.dataset_manager.event_manager.broadcast", mock_broadcast):
        result = mgr.enable_all_images(ds.name)
        await asyncio.sleep(0.05)

    assert result["reset_count"] == 2
    # No per-item events any more …
    assert not _entity_events(mock_broadcast, entity="media_item", op="updated")
    # … just one coarse dataset.invalidated for this dataset.
    invalidated = [
        c.args[1]
        for c in mock_broadcast.await_args_list
        if c.args and c.args[0] == "dataset.invalidated"
    ]
    assert invalidated == [{"name": ds.name}]


# ── overlay_recipe.rerender_overlay_from_recipe (real pipeline, CPU op) ──────

def test_rerender_overlay_from_recipe_real(tmp_path):
    """The real re-render runs a CPU recipe op against the current base and
    rewrites the overlay PNG, returning (dims, hash, ops)."""
    from app.core.dataset.overlay_recipe import rerender_overlay_from_recipe

    ds_path = tmp_path / "ds"
    os.makedirs(ds_path)
    Image.new("RGB", (320, 200), (200, 120, 60)).save(ds_path / "img.png")
    os.makedirs(ds_path / "overlays")
    Image.new("RGB", (999, 999), (0, 0, 0)).save(ds_path / "overlays" / "img.png")
    ops = [{"type": "vignette", "enabled": True, "params": {"amount": 0.4}}]
    with open(ds_path / "overlays.json", "w", encoding="utf-8") as f:
        json.dump({"img.png": {"overlay_file": "overlays/img.png",
                               "created_at": "x", "operations": ops}}, f)

    result = rerender_overlay_from_recipe(str(ds_path), "img.png")
    assert result is not None
    dims, overlay_hash, returned_ops = result
    assert dims == (320, 200)  # re-rendered against the CURRENT base (320x200)
    assert len(overlay_hash) == 64
    assert returned_ops == ops
    # Overlay PNG was rewritten to the new dimensions.
    with Image.open(ds_path / "overlays" / "img.png") as im:
        assert im.size == (320, 200)


def test_rerender_overlay_from_recipe_no_recipe_returns_none(tmp_path):
    """No recipe → None (nothing to reconcile)."""
    from app.core.dataset.overlay_recipe import rerender_overlay_from_recipe

    ds_path = tmp_path / "ds"
    os.makedirs(ds_path)
    Image.new("RGB", (64, 64), (1, 2, 3)).save(ds_path / "img.png")
    assert rerender_overlay_from_recipe(str(ds_path), "img.png") is None
