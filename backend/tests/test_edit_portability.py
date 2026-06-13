"""PR9 — paired edit-dataset portability round-trip.

Pins that export/import carries the three edit-specific pieces (no code beyond
the existing manifest passthrough is needed, but this guards against a future
regression):
- control/ folders ride along in the archive tree
- dataset ``kind`` round-trips
- per-image ``control_info.role_order`` round-trips verbatim (manifest is
  authoritative on import — no rescan resets it)
- a legacy archive with no ``kind`` imports as ``standard``
"""

from __future__ import annotations

import os
import zipfile
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.core.dataset import portable
from app.core.dataset_manager import Dataset, DatasetManager


@pytest.fixture()
def manager(tmp_path):
    inst = MagicMock()
    inst.get_module_settings.return_value = {}
    with patch("app.core.dataset_manager.get_settings_manager", return_value=inst):
        with patch.object(DatasetManager, "__init__", lambda self, **kw: None):
            mgr = DatasetManager()
    mgr.root_dir = str(tmp_path)
    mgr.default_root = str(tmp_path / "datasets")
    mgr.datasets = {}
    mgr._loop = None
    mgr._db = MagicMock()
    mgr._dataset_repo = MagicMock()
    mgr._media_repo = MagicMock()
    return mgr


def _edit_dataset(tmp_path) -> Dataset:
    return Dataset(
        id="d1", name="EditSet", path=str(tmp_path / "EditSet"),
        created_at=1.0, kind="edit",
        media_metadata={
            "img1.png": {
                "width": 64, "height": 64, "enabled": True, "is_video": False,
                "control_count": 1,
                "control_info": {
                    "slots": {"control": {"rel_path": "control/img1.jpg",
                                          "width": 64, "height": 64}},
                    "role_order": ["control", "root"],
                },
            },
        },
    )


def test_manifest_carries_kind_and_role_order(tmp_path):
    manifest = portable.build_manifest(_edit_dataset(tmp_path), "0.6.0")
    assert manifest["dataset"]["kind"] == "edit"
    role_order = manifest["media"]["img1.png"]["control_info"]["role_order"]
    assert role_order == ["control", "root"]


def test_export_zip_includes_control_folder(tmp_path):
    root = tmp_path / "EditSet"
    os.makedirs(root / "control")
    Image.new("RGB", (64, 64), "red").save(root / "img1.png")
    Image.new("RGB", (64, 64), "blue").save(root / "control" / "img1.jpg")

    manifest = portable.build_manifest(_edit_dataset(tmp_path), "0.6.0")
    buf = portable.write_export_zip(root, manifest)
    with zipfile.ZipFile(buf) as zf:
        names = zf.namelist()
    assert "control/img1.jpg" in names
    assert "img1.png" in names
    assert "manifest.json" in names


def test_import_restores_kind_and_role_order(manager, tmp_path):
    manifest = portable.build_manifest(_edit_dataset(tmp_path), "0.6.0")
    target = str(tmp_path / "Imported")
    os.makedirs(target, exist_ok=True)

    ds = manager.register_imported_dataset("Imported", manifest, path=target)
    assert ds.kind == "edit"
    info = ds.media_metadata["img1.png"]["control_info"]
    assert info["role_order"] == ["control", "root"]


def test_legacy_manifest_without_kind_defaults_standard(manager, tmp_path):
    # A pre-edit archive: dataset section has no `kind`.
    manifest = portable.build_manifest(_edit_dataset(tmp_path), "0.6.0")
    manifest["dataset"].pop("kind", None)
    target = str(tmp_path / "Legacy")
    os.makedirs(target, exist_ok=True)

    ds = manager.register_imported_dataset("Legacy", manifest, path=target)
    assert ds.kind == "standard"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
