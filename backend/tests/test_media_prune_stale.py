"""Regression tests for stale ``media_items`` row eviction.

Harmonize renames/converts files and a rescan rebuilds ``media_metadata`` from
disk, but persistence used to only UPSERT — never DELETE — so rows for the old
filenames lingered in the DB. After a restart those ghost rows (with stale
perceptual hashes) hydrated back into memory and resurfaced in Dataset Analysis
as near-duplicates against files no longer on disk (broken thumbnails).

These lock down that a scan/persist now makes ``media_items`` mirror the
current on-disk set: rows whose ``rel_path`` is gone are pruned.
"""

import os
import time
import uuid

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from app.core.dataset_manager import DatasetManager
from app.core.db.engine import DatabaseEngine
from app.core.db.migrations import run_migrations
from app.core.db.repositories.dataset_repo import DatasetRepository
from app.core.db.repositories.media_item_repo import MediaItemRepository


@pytest.fixture()
def db_engine(tmp_path):
    engine = DatabaseEngine(str(tmp_path / "test.db"))
    run_migrations(engine)
    return engine


@pytest.fixture()
def media_repo(db_engine):
    repo = MediaItemRepository()
    with patch("app.core.db.repositories.media_item_repo.get_db", return_value=db_engine):
        yield repo


@pytest.fixture()
def dataset_repo(db_engine):
    repo = DatasetRepository()
    with patch("app.core.db.repositories.dataset_repo.get_db", return_value=db_engine):
        yield repo


def _make_dataset_row(name: str) -> dict:
    return {
        "id": str(uuid.uuid4()), "name": name, "path": f"/tmp/{name}",
        "description": "", "created_at": time.time(), "last_scanned_at": None,
        "file_count": 0, "total_size_bytes": 0, "multimedia_count": 0,
        "caption_count": 0, "mask_count": 0, "caption_coverage": False,
        "missing": False, "preview_image": None, "majority_ar": 1.0,
        "harmonization_score": 0.0, "classifier": "", "version": "1.0.0",
        "has_cache": False,
    }


def _row(ds_id: str, rel_path: str) -> dict:
    return {
        "dataset_id": ds_id, "rel_path": rel_path, "width": 100, "height": 100,
        "aspect_ratio": 1.0, "orientation": "squared", "size_bytes": 1024,
    }


# ── Repository-level prune ────────────────────────────────────────────────


class TestPruneMissing:
    def test_prunes_rows_absent_from_keep_set(self, media_repo, dataset_repo, db_engine):
        ds = _make_dataset_row("prune_ds")
        dataset_repo.upsert(ds)
        for name in ("old1.jpg", "old2.jpg", "kept.jpg"):
            media_repo.upsert(_row(ds["id"], name))

        with db_engine.write() as conn:
            deleted = media_repo.prune_missing_with_conn(conn, ds["id"], {"kept.jpg"})

        assert deleted == 2
        remaining = set(media_repo.to_metadata_dict(ds["id"]).keys())
        assert remaining == {"kept.jpg"}

    def test_no_op_when_all_present(self, media_repo, dataset_repo, db_engine):
        ds = _make_dataset_row("prune_noop")
        dataset_repo.upsert(ds)
        media_repo.upsert(_row(ds["id"], "a.jpg"))
        with db_engine.write() as conn:
            deleted = media_repo.prune_missing_with_conn(conn, ds["id"], {"a.jpg", "b.jpg"})
        assert deleted == 0
        assert set(media_repo.to_metadata_dict(ds["id"]).keys()) == {"a.jpg"}


# ── Full manager rename-eviction regression ───────────────────────────────


@pytest.fixture()
def real_manager(tmp_path, db_engine):
    mock_settings = MagicMock()
    mock_settings.get_module_settings.return_value = {}
    default_root = str(tmp_path / "datasets")
    os.makedirs(default_root, exist_ok=True)
    with patch.object(DatasetManager, "__init__", lambda self, **kw: None):
        mgr = DatasetManager()
    mgr.root_dir = str(tmp_path)
    mgr.storage_file = str(tmp_path / "dataset_locations.json")
    mgr.default_root = default_root
    mgr.settings_manager = mock_settings
    mgr.datasets = {}
    mgr._loop = None
    mgr._db = db_engine
    mgr._dataset_repo = DatasetRepository()
    mgr._media_repo = MediaItemRepository()
    return mgr


def _patch_db(db_engine):
    return (
        patch("app.core.db.repositories.media_item_repo.get_db", return_value=db_engine),
        patch("app.core.db.repositories.dataset_repo.get_db", return_value=db_engine),
        patch.object(DatabaseEngine, "get_instance", return_value=db_engine),
    )


def _create_image(path: str):
    Image.new("RGB", (100, 100), "red").save(path)


def test_renamed_file_does_not_leave_a_ghost_row(real_manager, db_engine, tmp_path):
    """CORE REGRESSION (the Analysis ghost-duplicate bug).

    Scan two images, then rename one on disk (as harmonize does) and rescan.
    The old filename's row must be evicted — not linger with a stale hash that
    a fresh reload would resurface in Dataset Analysis.
    """
    p_media, p_dataset, p_singleton = _patch_db(db_engine)
    with p_media, p_dataset, p_singleton:
        ds = real_manager.create_dataset("harmonize_ds")
        _create_image(os.path.join(ds.path, "IMG_0001.jpg"))
        _create_image(os.path.join(ds.path, "keep.jpg"))
        real_manager.scan_dataset("harmonize_ds")
        assert set(real_manager.datasets["harmonize_ds"].media_metadata) == {
            "IMG_0001.jpg", "keep.jpg",
        }

        # Simulate harmonize renaming IMG_0001.jpg → canonical_00001.jpg on disk.
        os.rename(
            os.path.join(ds.path, "IMG_0001.jpg"),
            os.path.join(ds.path, "canonical_00001.jpg"),
        )
        real_manager.scan_dataset("harmonize_ds")

        # In-memory reflects the rename...
        assert set(real_manager.datasets["harmonize_ds"].media_metadata) == {
            "canonical_00001.jpg", "keep.jpg",
        }
        # ...and so does the DB — no ghost row survives a restart.
        repo = MediaItemRepository()
        persisted = set(repo.to_metadata_dict(ds.id).keys())
        assert persisted == {"canonical_00001.jpg", "keep.jpg"}
        assert "IMG_0001.jpg" not in persisted
