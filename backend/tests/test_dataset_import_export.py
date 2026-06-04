"""Round-trip tests for dataset export + import (manager + routes)."""

import os
from unittest.mock import MagicMock, patch

import pytest

from app.core.dataset_manager import DatasetManager
from app.core.db.repositories.dataset_repo import DatasetRepository
from app.core.db.repositories.media_item_repo import MediaItemRepository


@pytest.fixture()
def real_repo_manager(tmp_path):
    """A DatasetManager wired to the isolated test DB (real repositories).

    Relies on conftest's session-scoped ``_isolate_test_db`` fixture, which
    points the DatabaseEngine singleton at a throwaway SQLite file.
    """
    from app.core.db.engine import DatabaseEngine

    default_root = str(tmp_path / "datasets")
    os.makedirs(default_root, exist_ok=True)

    with patch.object(DatasetManager, "__init__", lambda self, **kw: None):
        mgr = DatasetManager()
    mgr.root_dir = str(tmp_path)
    mgr.storage_file = str(tmp_path / "dataset_locations.json")
    mgr.default_root = default_root
    mgr.settings_manager = MagicMock()
    mgr.datasets = {}
    mgr._loop = None
    mgr._db = DatabaseEngine.get_instance()
    mgr._dataset_repo = DatasetRepository()
    mgr._media_repo = MediaItemRepository()
    return mgr


def test_register_imported_dataset_persists_metadata_without_scan(real_repo_manager):
    mgr = real_repo_manager
    manifest = {
        "format_version": 1,
        "dataset": {
            "name": "ShouldBeIgnored", "description": "d", "version": "2.3.4",
            "trigger_word": "tw", "tags": ["x", "y"], "classifier": "style",
            "harmonization_score": 0.7, "created_at": 5.0,
        },
        "media": {
            "a.jpg": {"width": 10, "height": 10, "quality_score": 0.42,
                      "enabled": True, "has_caption": True},
            "b.jpg": {"width": 10, "height": 10, "quality_score": 0.11,
                      "enabled": False, "has_mask": True},
        },
    }
    target = os.path.join(mgr.default_root, "Imported")
    os.makedirs(target, exist_ok=True)

    ds = mgr.register_imported_dataset("Imported", manifest, path=target)

    # name comes from the argument, not the manifest's stale name
    assert ds.name == "Imported"
    assert ds.version == "2.3.4"
    assert ds.trigger_word == "tw"
    assert ds.tags == ["x", "y"]
    # a fresh id was generated (manifest never carries one)
    assert ds.id and ds.id != "ShouldBeIgnored"

    # metadata actually landed in SQLite, verbatim, with no HPS recompute
    rows = MediaItemRepository().to_metadata_dict(ds.id)
    assert rows["a.jpg"]["quality_score"] == 0.42
    assert rows["a.jpg"]["enabled"] is True
    assert rows["b.jpg"]["enabled"] is False
    assert rows["b.jpg"]["has_mask"] is True


def _seed_dataset_on_disk_and_db(tmp_root, name="Roundtrip"):
    """Create a real dataset folder + DB rows via the app's singleton manager.

    Returns the dataset_manager and the dataset's on-disk root.
    """
    from app.core.dataset_manager import dataset_manager
    from PIL import Image

    dataset_manager.default_root = str(tmp_root)
    dataset_manager.datasets = {}  # start clean for the test

    root = os.path.join(tmp_root, name)
    os.makedirs(os.path.join(root, "masks"), exist_ok=True)
    os.makedirs(os.path.join(root, ".thumbnails"), exist_ok=True)
    Image.new("RGB", (16, 16), "red").save(os.path.join(root, "a.jpg"))
    with open(os.path.join(root, "a.txt"), "w", encoding="utf-8") as f:
        f.write("a caption")
    Image.new("RGB", (16, 16), "black").save(os.path.join(root, "masks", "a.png"))
    with open(os.path.join(root, ".thumbnails", "a.webp"), "wb") as f:
        f.write(b"thumb")  # must be excluded from export

    ds = dataset_manager.register_imported_dataset(
        name,
        {
            "format_version": 1,
            "dataset": {"version": "1.5.0", "trigger_word": "rt",
                        "tags": ["t1"], "harmonization_score": 0.6},
            "media": {"a.jpg": {"width": 16, "height": 16, "quality_score": 0.5,
                                "enabled": True, "has_caption": True,
                                "has_mask": True}},
        },
        path=root,
    )
    return dataset_manager, root, ds


def test_export_endpoint_returns_zip_with_manifest(client, tmp_path):
    import io
    import json
    import zipfile

    _seed_dataset_on_disk_and_db(tmp_path, "Roundtrip")

    resp = client.get("/api/datasets/Roundtrip/export")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "a.jpg" in names and "a.txt" in names and "masks/a.png" in names
        assert not any(n.startswith(".thumbnails/") for n in names)
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["dataset"]["trigger_word"] == "rt"
        assert manifest["media"]["a.jpg"]["quality_score"] == 0.5


def test_import_upload_roundtrip_restores_metadata(client, tmp_path):
    import io
    # Build an export from a seeded dataset, then delete it and re-import.
    dm, root, ds = _seed_dataset_on_disk_and_db(tmp_path, "Roundtrip")
    export = client.get("/api/datasets/Roundtrip/export").content
    dm.delete_dataset("Roundtrip", delete_files=True)
    assert dm.get_dataset("Roundtrip") is None

    resp = client.post(
        "/api/datasets/import",
        files={"file": ("Roundtrip_1.5.0.zip", io.BytesIO(export), "application/zip")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Roundtrip"
    assert body["trigger_word"] == "rt"
    assert body["version"] == "1.5.0"
    assert body["media_metadata"]["a.jpg"]["quality_score"] == 0.5


def test_import_conflict_returns_409_then_overwrite_succeeds(client, tmp_path):
    import io
    dm, root, ds = _seed_dataset_on_disk_and_db(tmp_path, "Dupe")
    export = client.get("/api/datasets/Dupe/export").content

    # Name still exists -> 409 with structured body
    conflict = client.post(
        "/api/datasets/import",
        files={"file": ("Dupe.zip", io.BytesIO(export), "application/zip")},
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["conflict"] is True
    assert conflict.json()["detail"]["name"] == "Dupe"

    # Retry with overwrite -> replaces it
    ok = client.post(
        "/api/datasets/import",
        files={"file": ("Dupe.zip", io.BytesIO(export), "application/zip")},
        data={"on_conflict": "overwrite"},
    )
    assert ok.status_code == 200
    assert ok.json()["name"] == "Dupe"


def test_import_conflict_rename_creates_suffixed_dataset(client, tmp_path):
    import io
    dm, root, ds = _seed_dataset_on_disk_and_db(tmp_path, "Solo")
    export = client.get("/api/datasets/Solo/export").content

    ok = client.post(
        "/api/datasets/import",
        files={"file": ("Solo.zip", io.BytesIO(export), "application/zip")},
        data={"on_conflict": "rename"},
    )
    assert ok.status_code == 200
    assert ok.json()["name"] == "Solo (imported)"


def test_import_path_roundtrip(client, tmp_path):
    dm, root, ds = _seed_dataset_on_disk_and_db(tmp_path, "ByPath")
    export = client.get("/api/datasets/ByPath/export").content
    archive_path = os.path.join(tmp_path, "ByPath_export.zip")
    with open(archive_path, "wb") as f:
        f.write(export)
    dm.delete_dataset("ByPath", delete_files=True)

    resp = client.post("/api/datasets/import-path", json={"archive_path": archive_path})
    assert resp.status_code == 200
    assert resp.json()["name"] == "ByPath"
