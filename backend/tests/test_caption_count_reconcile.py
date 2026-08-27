"""A caption save reconciles the media item it belongs to, even if that item
predates the last scan.

``save_caption`` derives ``caption_count`` by counting ``has_caption`` flags in
``media_metadata``. That is only correct while ``media_metadata`` is complete:
an image added to the dataset folder after the last scan has no entry, so the
flag loop matched nothing, the recount returned the same number as before, and
nothing was persisted or broadcast. The caption was on disk and the library card
kept showing the old total until someone ran a rescan by hand.

The write path now adopts the sibling media file (a single-item incremental
rescan) instead of silently skipping it.
"""
from __future__ import annotations

import os
import time

import pytest

from app.core.dataset_manager import Dataset, DatasetManager
from app.core.db.repositories.dataset_repo import DatasetRepository
from app.core.db.repositories.media_item_repo import MediaItemRepository


def _png(path: str, size: tuple[int, int] = (64, 48)) -> None:
    from PIL import Image

    Image.new("RGB", size, (10, 20, 30)).save(path)


@pytest.fixture
def unscanned_manager(tmp_path):
    """Dataset whose folder holds an image the last scan never saw."""
    mgr = DatasetManager.__new__(DatasetManager)
    mgr.datasets = {}
    # No loop: these are sync tests asserting on counts, never on broadcasts.
    # Every _loop consumer in DatasetManager guards `is None` — the documented
    # "no-op before the loop is wired" path — so None is the honest value here.
    # It was only ever a loop because 0.21's get_event_loop() handed one over.
    mgr._loop = None
    mgr._dataset_repo = DatasetRepository()
    mgr._media_repo = MediaItemRepository()

    ds_id = f"ds-recon-{time.time_ns()}"
    ds_name = f"recon-{ds_id}"
    ds_path = tmp_path / ds_id
    os.makedirs(ds_path, exist_ok=True)
    _png(str(ds_path / "photo.png"))

    ds = Dataset(
        id=ds_id,
        name=ds_name,
        path=str(ds_path),
        created_at=time.time(),
        multimedia_count=0,
        caption_count=0,
        media_metadata={},          # never scanned since the image landed
    )
    mgr.datasets[ds_name] = ds
    mgr._dataset_repo.upsert(ds.model_dump(exclude={"media_metadata"}))
    return mgr, ds


def test_caption_for_an_unscanned_image_updates_the_count(unscanned_manager):
    mgr, ds = unscanned_manager

    mgr.save_caption(ds.name, "photo.txt", "a red car")

    assert "photo.png" in ds.media_metadata, (
        "the caption's media file was never adopted, so the count cannot follow"
    )
    assert ds.media_metadata["photo.png"]["has_caption"] is True
    assert ds.caption_count == 1


def test_the_adopted_entry_survives_a_reload(unscanned_manager):
    """The recount has to be persisted, not just held in memory."""
    mgr, ds = unscanned_manager

    mgr.save_caption(ds.name, "photo.txt", "a red car")

    row = mgr._dataset_repo.get_by_id(ds.id)
    assert row is not None
    assert row["caption_count"] == 1
    assert mgr._media_repo.get_by_path(ds.id, "photo.png") is not None


def test_an_orphan_caption_invents_nothing(unscanned_manager):
    """No media file with that stem — the caption is written, but it counts
    toward nothing and must not fabricate a media item."""
    mgr, ds = unscanned_manager

    mgr.save_caption(ds.name, "ghost.txt", "no such image")

    assert "ghost.png" not in ds.media_metadata
    assert not any(k.startswith("ghost") for k in ds.media_metadata)
    assert ds.caption_count == 0


def test_an_already_known_image_is_not_re_adopted(tmp_path):
    """The existing fast path is untouched: a scanned image just flips its flag."""
    mgr = DatasetManager.__new__(DatasetManager)
    mgr.datasets = {}
    # No loop: these are sync tests asserting on counts, never on broadcasts.
    # Every _loop consumer in DatasetManager guards `is None` — the documented
    # "no-op before the loop is wired" path — so None is the honest value here.
    # It was only ever a loop because 0.21's get_event_loop() handed one over.
    mgr._loop = None
    mgr._dataset_repo = DatasetRepository()
    mgr._media_repo = MediaItemRepository()

    ds_id = f"ds-known-{time.time_ns()}"
    ds_name = f"known-{ds_id}"
    ds_path = tmp_path / ds_id
    os.makedirs(ds_path, exist_ok=True)
    _png(str(ds_path / "photo.png"), size=(32, 32))

    meta = {
        "width": 32, "height": 32, "aspect_ratio": 1.0, "orientation": "square",
        "size_bytes": 1, "has_caption": False, "is_video": False, "enabled": True,
    }
    ds = Dataset(
        id=ds_id, name=ds_name, path=str(ds_path), created_at=time.time(),
        multimedia_count=1, caption_count=0,
        media_metadata={"photo.png": dict(meta)},
    )
    mgr.datasets[ds_name] = ds
    mgr._dataset_repo.upsert(ds.model_dump(exclude={"media_metadata"}))
    mgr._media_repo.upsert({"dataset_id": ds_id, "rel_path": "photo.png", **meta})

    mgr.save_caption(ds.name, "photo.txt", "a red car")

    assert ds.caption_count == 1
    # Dimensions come from the existing entry, not a re-read of the file.
    assert ds.media_metadata["photo.png"]["width"] == 32
