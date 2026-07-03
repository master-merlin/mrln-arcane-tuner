"""Tests for paired-image (edit/kontext) dataset support — dataset layer.

Covers the PR1 contract:
- V15 migration: ``datasets.kind`` + ``media_items.control_count``/``control_info``
- Repository column handling for the new fields
- Control slot detection in ``build_media_entry`` (stem-matched ``control*/`` dirs)
- The extended ``/pairs`` payload with logical role resolution
  (``role_order`` permutes which physical slot is the training target)
- Control sidecar cleanup on media deletion
- The pixel-edit staleness stamp (``control_info.target_edited_at``)
"""

import os
import time
import uuid

import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

from app.core.db.engine import DatabaseEngine
from app.core.db.migrations import run_migrations, _migrate_v15
from app.core.db.repositories.media_item_repo import MediaItemRepository
from app.core.db.repositories.dataset_repo import DatasetRepository
from app.core.dataset.scan_helpers import build_media_entry
from app.core.dataset.control_helpers import (
    CONTROL_SLOTS,
    detect_control_slots,
    resolve_effective_roles,
)
from app.core.dataset.media_helpers import refresh_media_metadata_after_change
from app.core.dataset_manager import Dataset, DatasetManager
from app.api.schemas.dataset_schemas import DatasetPairResponse


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def db_engine(tmp_path):
    """Isolated DatabaseEngine backed by a temp SQLite file."""
    return DatabaseEngine(str(tmp_path / "test.db"))


@pytest.fixture()
def migrated_engine(db_engine):
    run_migrations(db_engine)
    return db_engine


@pytest.fixture()
def media_repo(migrated_engine):
    repo = MediaItemRepository()
    with patch(
        "app.core.db.repositories.media_item_repo.get_db",
        return_value=migrated_engine,
    ):
        yield repo


@pytest.fixture()
def dataset_repo(migrated_engine):
    repo = DatasetRepository()
    with patch(
        "app.core.db.repositories.dataset_repo.get_db",
        return_value=migrated_engine,
    ):
        yield repo


@pytest.fixture()
def mock_settings():
    mock_instance = MagicMock()
    mock_instance.get_module_settings.return_value = {}
    mock_instance.update_module_settings = MagicMock()
    with patch(
        "app.core.dataset_manager.get_settings_manager",
        return_value=mock_instance,
    ):
        yield mock_instance


@pytest.fixture()
def manager(tmp_path, mock_settings):
    """DatasetManager rooted in tmp_path with mocked persistence."""
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
    mgr._db = MagicMock()
    mgr._dataset_repo = MagicMock()
    mgr._media_repo = MagicMock()
    return mgr


def _create_image(path: str, width: int = 64, height: int = 64, color: str = "red"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (width, height), color).save(path)


def _create_text(path: str, text: str = "make it a watercolor painting"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _make_edit_dataset(manager, name: str = "editds", slots: int = 2):
    """Create a dataset with one target image + N control slots on disk."""
    ds = manager.create_dataset(name, kind="edit")
    _create_image(os.path.join(ds.path, "img1.png"))
    _create_text(os.path.join(ds.path, "img1.txt"))
    if slots >= 1:
        _create_image(os.path.join(ds.path, "control", "img1.jpg"), color="blue")
    if slots >= 2:
        _create_image(os.path.join(ds.path, "control_2", "img1.png"), color="green")
    manager.scan_dataset(name)
    return ds


# ── V15 migration ────────────────────────────────────────────────────────


class TestV15Migration:
    def test_run_migrations_reaches_latest(self, db_engine):
        # V17 (legacy definition_id repair) is the current schema tip.
        run_migrations(db_engine)
        with db_engine.connection() as conn:
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            assert row["version"] == 17

    def test_dataset_kind_defaults_standard(self, migrated_engine):
        with migrated_engine.write() as conn:
            conn.execute(
                "INSERT INTO datasets (id, name, path, created_at) "
                "VALUES (?, ?, ?, ?)",
                ("ds-kind", "kind_ds", "/tmp/kind_ds", time.time()),
            )
        with migrated_engine.connection() as conn:
            row = conn.execute(
                "SELECT kind FROM datasets WHERE id = ?", ("ds-kind",)
            ).fetchone()
            assert row["kind"] == "standard"

    def test_media_item_control_columns(self, migrated_engine):
        with migrated_engine.write() as conn:
            conn.execute(
                "INSERT INTO datasets (id, name, path, created_at) "
                "VALUES (?, ?, ?, ?)",
                ("ds-ctl", "ctl_ds", "/tmp/ctl_ds", time.time()),
            )
            conn.execute(
                "INSERT INTO media_items ("
                " dataset_id, rel_path, width, height, aspect_ratio,"
                " orientation, size_bytes, control_count, control_info"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("ds-ctl", "img.png", 64, 64, 1.0, "squared", 10,
                 2, '{"slots": {}}'),
            )
        with migrated_engine.connection() as conn:
            row = conn.execute(
                "SELECT control_count, control_info FROM media_items LIMIT 1"
            ).fetchone()
            assert row["control_count"] == 2
            assert row["control_info"] == '{"slots": {}}'

    def test_control_count_defaults_zero(self, migrated_engine):
        with migrated_engine.write() as conn:
            conn.execute(
                "INSERT INTO datasets (id, name, path, created_at) "
                "VALUES (?, ?, ?, ?)",
                ("ds-def", "def_ds", "/tmp/def_ds", time.time()),
            )
            conn.execute(
                "INSERT INTO media_items ("
                " dataset_id, rel_path, width, height, aspect_ratio,"
                " orientation, size_bytes"
                ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("ds-def", "img.png", 64, 64, 1.0, "squared", 10),
            )
        with migrated_engine.connection() as conn:
            row = conn.execute(
                "SELECT control_count, control_info FROM media_items LIMIT 1"
            ).fetchone()
            assert row["control_count"] == 0
            assert row["control_info"] is None

    def test_v15_is_idempotent(self, migrated_engine):
        # Already applied by run_migrations — a second call must not raise.
        with migrated_engine.write() as conn:
            _migrate_v15(conn)


# ── Repository columns ───────────────────────────────────────────────────


class TestRepoControlColumns:
    def test_media_columns_include_control_fields(self):
        cols = MediaItemRepository._COLUMNS
        assert "control_count" in cols
        assert "control_info" in cols

    def test_dataset_columns_include_kind(self):
        assert "kind" in DatasetRepository._COLUMNS

    def test_prepare_serializes_control_info(self):
        prepared = MediaItemRepository._prepare(
            {"control_info": {"slots": {"control": {"rel_path": "control/a.jpg"}}}}
        )
        assert isinstance(prepared["control_info"], str)
        assert "control/a.jpg" in prepared["control_info"]

    def test_to_metadata_dict_parses_control_info(self, media_repo, dataset_repo):
        ds_id = str(uuid.uuid4())
        dataset_repo.upsert({
            "id": ds_id, "name": "ctl_roundtrip", "path": "/tmp/x",
            "created_at": time.time(),
        })
        media_repo.upsert({
            "dataset_id": ds_id, "rel_path": "img.png",
            "width": 64, "height": 64, "aspect_ratio": 1.0,
            "orientation": "squared", "size_bytes": 10, "frame_count": 0,
            "control_count": 1,
            "control_info": {"slots": {"control": {"rel_path": "control/img.jpg"}},
                             "role_order": ["control", "root"]},
        })
        meta = media_repo.to_metadata_dict(ds_id)["img.png"]
        assert meta["control_count"] == 1
        assert isinstance(meta["control_info"], dict)
        assert meta["control_info"]["role_order"] == ["control", "root"]

    def test_dataset_kind_roundtrip(self, dataset_repo):
        ds_id = str(uuid.uuid4())
        dataset_repo.upsert({
            "id": ds_id, "name": "edit_ds", "path": "/tmp/e",
            "created_at": time.time(), "kind": "edit",
        })
        row = dataset_repo.get_by_id(ds_id)
        assert row["kind"] == "edit"


# ── Dataset model + manager kind handling ────────────────────────────────


class TestDatasetKind:
    def test_model_defaults_standard(self):
        ds = Dataset(id="1", name="t", path="/p", created_at=1.0)
        assert ds.kind == "standard"

    def test_model_roundtrip_kind(self):
        ds = Dataset(id="1", name="t", path="/p", created_at=1.0, kind="edit")
        assert Dataset.model_validate(ds.model_dump()).kind == "edit"

    def test_create_dataset_with_kind(self, manager):
        ds = manager.create_dataset("editset", kind="edit")
        assert ds.kind == "edit"

    def test_create_dataset_defaults_standard(self, manager):
        assert manager.create_dataset("plain").kind == "standard"

    def test_update_dataset_changes_kind(self, manager):
        manager.create_dataset("flip")
        ds = manager.update_dataset("flip", "flip", "", new_kind="edit")
        assert ds.kind == "edit"

    def test_update_dataset_none_preserves_kind(self, manager):
        manager.create_dataset("keep", kind="edit")
        ds = manager.update_dataset("keep", "keep", "desc")
        assert ds.kind == "edit"


# ── Control slot detection (scan) ────────────────────────────────────────


class TestControlSlotDetection:
    def _entry(self, tmp_path, existing_meta=None):
        ds = str(tmp_path / "ds")
        img = os.path.join(ds, "img1.png")
        _create_image(img)
        return ds, lambda: build_media_entry(
            img, "img1", ".png", ds, existing_meta or {}, 64, 64,
        )

    def test_no_controls(self, tmp_path):
        _, build = self._entry(tmp_path)
        entry = build()
        assert entry["control_count"] == 0
        assert entry.get("control_info") is None

    def test_detects_slot_one(self, tmp_path):
        ds, build = self._entry(tmp_path)
        _create_image(os.path.join(ds, "control", "img1.jpg"), 32, 48)
        entry = build()
        assert entry["control_count"] == 1
        slot = entry["control_info"]["slots"]["control"]
        assert slot["rel_path"] == "control/img1.jpg"
        assert slot["width"] == 32
        assert slot["height"] == 48

    def test_detects_multiple_slots_in_order(self, tmp_path):
        ds, build = self._entry(tmp_path)
        _create_image(os.path.join(ds, "control", "img1.jpg"))
        _create_image(os.path.join(ds, "control_2", "img1.png"))
        _create_image(os.path.join(ds, "control_3", "img1.webp"))
        entry = build()
        assert entry["control_count"] == 3
        assert list(entry["control_info"]["slots"].keys()) == list(CONTROL_SLOTS)

    def test_ext_priority_jpg_over_png(self, tmp_path):
        ds, build = self._entry(tmp_path)
        _create_image(os.path.join(ds, "control", "img1.png"))
        _create_image(os.path.join(ds, "control", "img1.jpg"))
        entry = build()
        assert entry["control_info"]["slots"]["control"]["rel_path"] == "control/img1.jpg"

    def test_unmatched_stem_ignored(self, tmp_path):
        ds, build = self._entry(tmp_path)
        _create_image(os.path.join(ds, "control", "other.jpg"))
        entry = build()
        assert entry["control_count"] == 0

    def test_preserves_role_order_and_stamp(self, tmp_path):
        existing = {"control_info": {"role_order": ["control", "root"],
                                     "target_edited_at": 123.0}}
        ds, build = self._entry(tmp_path, existing_meta=existing)
        _create_image(os.path.join(ds, "control", "img1.jpg"))
        entry = build()
        assert entry["control_info"]["role_order"] == ["control", "root"]
        assert entry["control_info"]["target_edited_at"] == 123.0

    def test_detect_control_slots_helper(self, tmp_path):
        ds = str(tmp_path / "ds")
        _create_image(os.path.join(ds, "control_2", "img1.png"), 20, 30)
        slots = detect_control_slots(ds, "img1")
        assert list(slots.keys()) == ["control_2"]
        assert slots["control_2"]["rel_path"] == "control_2/img1.png"


# ── Role resolution ──────────────────────────────────────────────────────


class TestRoleResolution:
    SLOT_FILES = {"control": "control/img1.jpg", "control_2": "control_2/img1.png"}

    def test_default_order(self):
        target, controls = resolve_effective_roles(
            "img1.png", self.SLOT_FILES, None,
        )
        assert target == "img1.png"
        assert controls == ["control/img1.jpg", "control_2/img1.png"]

    def test_permuted_order(self):
        target, controls = resolve_effective_roles(
            "img1.png", self.SLOT_FILES, ["control", "root", "control_2"],
        )
        assert target == "control/img1.jpg"
        assert controls == ["img1.png", "control_2/img1.png"]

    def test_partial_order_appends_missing_slots(self):
        target, controls = resolve_effective_roles(
            "img1.png", self.SLOT_FILES, ["control_2", "root"],
        )
        assert target == "control_2/img1.png"
        assert controls == ["img1.png", "control/img1.jpg"]

    def test_invalid_slot_falls_back_to_default(self):
        target, controls = resolve_effective_roles(
            "img1.png", self.SLOT_FILES, ["control_3", "root"],
        )
        assert target == "img1.png"
        assert controls == ["control/img1.jpg", "control_2/img1.png"]


# ── /pairs contract ──────────────────────────────────────────────────────


class TestPairsContract:
    def test_standard_dataset_has_empty_control_fields(self, manager):
        ds = manager.create_dataset("plainpairs")
        _create_image(os.path.join(ds.path, "img1.png"))
        _create_text(os.path.join(ds.path, "img1.txt"))
        manager.scan_dataset("plainpairs")

        pairs = manager.get_dataset_pairs("plainpairs")
        assert len(pairs) == 1
        p = pairs[0]
        assert p["control_files"] == []
        assert p["role_order"] is None
        assert p["effective_target"] == "img1.png"
        assert p["effective_controls"] == []

    def test_edit_dataset_pairs_carry_controls(self, manager):
        _make_edit_dataset(manager, "editpairs", slots=2)
        pairs = manager.get_dataset_pairs("editpairs")
        assert len(pairs) == 1
        p = pairs[0]
        assert p["control_files"] == ["control/img1.jpg", "control_2/img1.png"]
        assert p["effective_target"] == "img1.png"
        assert p["effective_controls"] == ["control/img1.jpg", "control_2/img1.png"]
        assert p["control_info"]["slots"]["control"]["rel_path"] == "control/img1.jpg"

    def test_role_order_resolves_effective_roles(self, manager):
        ds = _make_edit_dataset(manager, "swapped", slots=1)
        meta = ds.media_metadata["img1.png"]
        meta["control_info"]["role_order"] = ["control", "root"]

        p = manager.get_dataset_pairs("swapped")[0]
        assert p["role_order"] == ["control", "root"]
        assert p["effective_target"] == "control/img1.jpg"
        assert p["effective_controls"] == ["img1.png"]

    def test_invalid_role_order_falls_back(self, manager):
        ds = _make_edit_dataset(manager, "badorder", slots=1)
        ds.media_metadata["img1.png"]["control_info"]["role_order"] = [
            "control_3", "root",
        ]
        p = manager.get_dataset_pairs("badorder")[0]
        assert p["effective_target"] == "img1.png"
        assert p["effective_controls"] == ["control/img1.jpg"]

    def test_pairs_rows_validate_against_schema(self, manager):
        _make_edit_dataset(manager, "schemards", slots=2)
        for row in manager.get_dataset_pairs("schemards"):
            resp = DatasetPairResponse.model_validate(row)
            assert resp.effective_target == "img1.png"
            assert resp.control_files == [
                "control/img1.jpg", "control_2/img1.png",
            ]


# ── Deletion cleanup ─────────────────────────────────────────────────────


class TestDeleteCleansControls:
    def test_delete_media_pair_removes_control_files(self, manager):
        ds = _make_edit_dataset(manager, "delpairs", slots=2)
        ctl1 = os.path.join(ds.path, "control", "img1.jpg")
        ctl2 = os.path.join(ds.path, "control_2", "img1.png")
        assert os.path.exists(ctl1) and os.path.exists(ctl2)

        manager.delete_media_pair("delpairs", "img1.png")
        assert not os.path.exists(ctl1)
        assert not os.path.exists(ctl2)


# ── Pixel-edit staleness stamp ───────────────────────────────────────────


class TestTargetEditedStamp:
    def test_stamp_set_when_controls_present(self, tmp_path):
        img = str(tmp_path / "img1.png")
        _create_image(img)
        metadata = {
            "img1.png": {
                "size_bytes": 1, "has_mask": False,
                "control_count": 1,
                "control_info": {"slots": {"control": {"rel_path": "control/img1.jpg"}}},
            }
        }
        refresh_media_metadata_after_change(metadata, "img1.png", img)
        stamp = metadata["img1.png"]["control_info"].get("target_edited_at")
        assert stamp is not None
        assert abs(stamp - time.time()) < 5

    def test_no_stamp_without_control_info(self, tmp_path):
        img = str(tmp_path / "img1.png")
        _create_image(img)
        metadata = {"img1.png": {"size_bytes": 1}}
        refresh_media_metadata_after_change(metadata, "img1.png", img)
        assert "control_info" not in metadata["img1.png"]

    def test_pixel_edit_does_not_delete_control_files(self, tmp_path):
        ds = str(tmp_path / "ds")
        img = os.path.join(ds, "img1.png")
        ctl = os.path.join(ds, "control", "img1.jpg")
        _create_image(img)
        _create_image(ctl)
        metadata = {"img1.png": {"size_bytes": 1, "control_info": {"slots": {}}}}
        refresh_media_metadata_after_change(metadata, "img1.png", img)
        assert os.path.exists(ctl)
