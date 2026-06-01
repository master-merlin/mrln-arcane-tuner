"""Regression tests for per-image ``enabled`` persistence.

The per-image ``enabled`` flag (the "exclude from training" toggle) must
survive a backend restart. It used to live only in the in-memory
``media_metadata`` dict; this suite locks down that it is now a durable
column on ``media_items`` that round-trips through the repository and a
fresh ``DatasetManager.load()``.

Covers:
- V8 migration: adds ``enabled INTEGER NOT NULL DEFAULT 1`` (existing rows
  backfill to enabled).
- MediaItemRepository: ``enabled`` in ``_COLUMNS``, ``_prepare`` int
  coercion, ``to_metadata_dict`` bool coercion.
- Full round-trip: toggle an image disabled, reload a new manager against
  the same DB, assert it is STILL disabled and ``excluded_count == 1``.
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


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def db_engine(tmp_path):
    """Isolated DatabaseEngine backed by a temp SQLite file (migrated)."""
    engine = DatabaseEngine(str(tmp_path / "test.db"))
    run_migrations(engine)
    return engine


@pytest.fixture()
def media_repo(db_engine):
    repo = MediaItemRepository()
    with patch(
        "app.core.db.repositories.media_item_repo.get_db",
        return_value=db_engine,
    ):
        yield repo


@pytest.fixture()
def dataset_repo(db_engine):
    repo = DatasetRepository()
    with patch(
        "app.core.db.repositories.dataset_repo.get_db",
        return_value=db_engine,
    ):
        yield repo


def _make_dataset_row(name: str = "test_ds") -> dict:
    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "path": f"/tmp/{name}",
        "description": "",
        "created_at": time.time(),
        "last_scanned_at": None,
        "file_count": 0,
        "total_size_bytes": 0,
        "multimedia_count": 0,
        "caption_count": 0,
        "mask_count": 0,
        "caption_coverage": False,
        "missing": False,
        "preview_image": None,
        "majority_ar": 1.0,
        "harmonization_score": 0.0,
        "classifier": "",
        "version": "1.0.0",
        "has_cache": False,
    }


# ── V8 Migration ─────────────────────────────────────────────────────────


class TestV8EnabledMigration:
    """The new migration adds an ``enabled`` column defaulting to 1."""

    def test_enabled_column_exists_with_default_1(self, db_engine):
        """After migrations, media_items has ``enabled`` defaulting to 1."""
        ds_id = "ds-enabled"
        with db_engine.write() as conn:
            conn.execute(
                "INSERT INTO datasets (id, name, path, created_at) VALUES (?, ?, ?, ?)",
                (ds_id, "ds_enabled", "/tmp/ds_enabled", time.time()),
            )
            # Insert WITHOUT specifying enabled — must fall back to DEFAULT 1
            conn.execute(
                "INSERT INTO media_items "
                "(dataset_id, rel_path, width, height, aspect_ratio, "
                " orientation, size_bytes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ds_id, "legacy.png", 100, 100, 1.0, "squared", 1024),
            )

        with db_engine.connection() as conn:
            row = conn.execute(
                "SELECT enabled FROM media_items WHERE rel_path = ?",
                ("legacy.png",),
            ).fetchone()
            assert row["enabled"] == 1

    def test_run_migrations_reaches_v8(self, db_engine):
        """Full run_migrations should bring schema to v8."""
        with db_engine.connection() as conn:
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            assert row["version"] == 8


# ── Repository round-trip ─────────────────────────────────────────────────


class TestMediaItemRepoEnabled:
    """``enabled`` is a first-class column with int<->bool coercion."""

    def test_columns_include_enabled(self):
        assert "enabled" in MediaItemRepository._COLUMNS

    def test_prepare_coerces_enabled_to_int(self):
        assert MediaItemRepository._prepare({"enabled": True})["enabled"] == 1
        assert MediaItemRepository._prepare({"enabled": False})["enabled"] == 0

    def test_to_metadata_dict_converts_enabled_to_bool(self, media_repo, dataset_repo):
        ds = _make_dataset_row("enabled_bool")
        dataset_repo.upsert(ds)
        media_repo.upsert(
            {
                "dataset_id": ds["id"],
                "rel_path": "off.png",
                "width": 100,
                "height": 100,
                "aspect_ratio": 1.0,
                "orientation": "squared",
                "size_bytes": 1024,
                "enabled": False,
            }
        )

        meta = media_repo.to_metadata_dict(ds["id"])["off.png"]
        assert meta["enabled"] is False
        assert isinstance(meta["enabled"], bool)

    def test_enabled_round_trips_through_db(self, media_repo, dataset_repo):
        """Disabled flag persists across a fresh read from the same DB."""
        ds = _make_dataset_row("enabled_rt")
        dataset_repo.upsert(ds)
        media_repo.upsert(
            {
                "dataset_id": ds["id"],
                "rel_path": "disabled.png",
                "width": 100,
                "height": 100,
                "aspect_ratio": 1.0,
                "orientation": "squared",
                "size_bytes": 1024,
                "enabled": False,
            }
        )

        # Fresh read (no in-memory state).
        meta = media_repo.to_metadata_dict(ds["id"])
        assert meta["disabled.png"]["enabled"] is False


# ── Full DatasetManager reload round-trip (core regression) ───────────────


@pytest.fixture()
def real_manager(tmp_path, db_engine):
    """A DatasetManager wired to a REAL temp DB (not mocked).

    Both repos and ``DatabaseEngine.get_instance`` resolve to ``db_engine``
    so persistence actually hits SQLite, enabling a true restart simulation.
    """
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
    """Context-manager bundle pointing every get_db / singleton at db_engine."""
    return (
        patch(
            "app.core.db.repositories.media_item_repo.get_db",
            return_value=db_engine,
        ),
        patch(
            "app.core.db.repositories.dataset_repo.get_db",
            return_value=db_engine,
        ),
        patch.object(DatabaseEngine, "get_instance", return_value=db_engine),
    )


def _create_image(path: str, width: int = 100, height: int = 100):
    Image.new("RGB", (width, height), "red").save(path)


def test_enabled_survives_manager_reload(real_manager, db_engine, tmp_path):
    """CORE REGRESSION: a disabled image stays disabled after a fresh load.

    1. Create + scan a dataset with two images.
    2. Disable one via ``toggle_image_enabled`` (persists the media row).
    3. Build a *new* DatasetManager against the same DB and ``load()``.
    4. Assert the image is STILL ``enabled=False`` and ``excluded_count == 1``.
    """
    p_media, p_dataset, p_singleton = _patch_db(db_engine)
    with p_media, p_dataset, p_singleton:
        ds = real_manager.create_dataset("reload_ds")
        _create_image(os.path.join(ds.path, "a.png"))
        _create_image(os.path.join(ds.path, "b.png"))
        real_manager.scan_dataset("reload_ds")

        # Sanity: both enabled at start.
        assert real_manager.datasets["reload_ds"].excluded_count == 0

        real_manager.toggle_image_enabled("reload_ds", "a.png", enabled=False)
        assert real_manager.datasets["reload_ds"].excluded_count == 1

        # ── Simulate a backend restart: brand-new manager, same DB ──
        with patch.object(DatasetManager, "__init__", lambda self, **kw: None):
            fresh = DatasetManager()
        fresh.root_dir = str(tmp_path)
        fresh.default_root = real_manager.default_root
        fresh.settings_manager = real_manager.settings_manager
        fresh.datasets = {}
        fresh._loop = None
        fresh._db = db_engine
        fresh._dataset_repo = DatasetRepository()
        fresh._media_repo = MediaItemRepository()
        fresh.load()

        reloaded = fresh.datasets["reload_ds"]
        assert reloaded.media_metadata["a.png"]["enabled"] is False
        assert reloaded.media_metadata["b.png"]["enabled"] is True
        assert reloaded.excluded_count == 1
