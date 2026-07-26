"""W1 fix wave — Finding 1: ``delete_media_pair`` was an unguarded
arbitrary-file-DELETE primitive.

Pre-fix, every filesystem target derived from the client-supplied
``media_file`` (the main media path, plus the caption/mask/masked/control
sidecars derived from its ``stem``) was built with a plain ``os.path.join``
and handed straight to ``os.remove`` — no containment check anywhere on the
path. A ``filename`` containing ``../`` segments (reachable via
``DELETE /api/datasets/{name}/pairs/{filename:path}``, whose ``:path``
converter allows slashes) could delete an arbitrary file outside the
dataset directory.

All filesystem targets now resolve through the shared
``validate_path_within`` guard (``app/api/_path_guard.py``), which raises
``HTTPException(403)`` on escape and returns the resolved ``Path`` that is
actually used for the ``os.remove``/existence check.

Fixture mirrors ``test_dataset_manager.py`` / ``test_caption_path_guard.py``:
a ``DatasetManager`` rooted in ``tmp_path`` with the DB layer mocked out.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi import HTTPException
from unittest.mock import MagicMock, patch

from app.core.dataset.control_helpers import CONTROL_SLOTS
from app.core.dataset_manager import DatasetManager


# ── Fixtures (mirrors test_dataset_manager.py / test_caption_path_guard.py) ──


@pytest.fixture()
def mock_settings():
    """Patch SettingsManager so DatasetManager never touches real disk."""
    mock_instance = MagicMock()
    mock_instance.get_module_settings.return_value = {}
    mock_instance.update_module_settings = MagicMock()
    with patch(
        "app.core.dataset_manager.get_settings_manager", return_value=mock_instance
    ):
        yield mock_instance


@pytest.fixture()
def manager(tmp_path, mock_settings):
    """Create a DatasetManager rooted in tmp_path."""
    storage_file = str(tmp_path / "dataset_locations.json")
    default_root = str(tmp_path / "datasets")
    os.makedirs(default_root, exist_ok=True)

    with patch.object(DatasetManager, "__init__", lambda self, **kw: None):
        mgr = DatasetManager()

    mgr.root_dir = str(tmp_path)
    mgr.storage_file = storage_file
    mgr.default_root = default_root
    mgr.settings_manager = mock_settings
    mgr.datasets = {}
    mgr._loop = None
    mgr._db = MagicMock()
    mgr._dataset_repo = MagicMock()
    mgr._media_repo = MagicMock()
    return mgr


def _create_image(path: str, width: int = 32, height: int = 32):
    from PIL import Image

    img = Image.new("RGB", (width, height), "red")
    img.save(path)


# ── The vulnerability: real sibling file outside the dataset ───────────────


def test_delete_media_pair_rejects_traversal_and_preserves_outside_file(manager):
    """A "../"-style filename must not be able to delete a real file that
    sits next to (outside) the dataset directory — the core CRITICAL finding.
    """
    ds = manager.create_dataset("foo")
    _create_image(str(Path(ds.path) / "a.jpg"))

    # A real sibling file OUTSIDE the dataset directory — the traversal target.
    outside_file = Path(ds.path).parent / "secret.bin"
    outside_file.write_bytes(b"do not delete me")

    with pytest.raises(HTTPException):
        manager.delete_media_pair("foo", "../secret.bin")

    assert outside_file.exists(), "traversal must not delete the outside file"
    assert outside_file.read_bytes() == b"do not delete me"


def test_delete_media_pair_route_does_not_swallow_traversal_into_404(
    manager, monkeypatch
):
    """Same attack through the real route coroutine (bypassing HTTP/URL
    dot-segment normalization ambiguity by calling the handler directly),
    confirming the route's ``except (ValueError, FileNotFoundError)`` clause
    does NOT catch the guard's ``HTTPException(403)`` and remap it into a
    misleading 404 — it must propagate as-is."""
    import asyncio

    import app.api.dataset.crud_routes as crud_routes

    ds = manager.create_dataset("foo")
    _create_image(str(Path(ds.path) / "a.jpg"))
    outside_file = Path(ds.path).parent / "secret_route.bin"
    outside_file.write_bytes(b"do not delete me either")

    monkeypatch.setattr(crud_routes, "dataset_manager", manager)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            crud_routes.delete_media_pair(name="foo", filename="../secret_route.bin")
        )

    assert excinfo.value.status_code == 403
    assert outside_file.exists()


# ── Regression: legitimate deletes (including all sidecars) still work ────


def test_delete_media_pair_removes_all_contained_sidecars(manager):
    """The per-target guards must not break normal, contained deletes —
    every sidecar class (caption, mask, masked image+caption, control
    slots) should still be removed for a legitimate same-dataset filename.
    """
    ds = manager.create_dataset("bar")
    ds_path = Path(ds.path)
    _create_image(str(ds_path / "a.jpg"))
    (ds_path / "a.txt").write_text("a caption", encoding="utf-8")

    masks_dir = ds_path / "masks"
    masks_dir.mkdir()
    _create_image(str(masks_dir / "a.png"))

    masked_dir = ds_path / "masked"
    masked_dir.mkdir()
    _create_image(str(masked_dir / "a.jpg"))
    (masked_dir / "a.txt").write_text("masked caption", encoding="utf-8")

    control_dir = ds_path / CONTROL_SLOTS[0]
    control_dir.mkdir()
    _create_image(str(control_dir / "a.jpg"))

    manager.delete_media_pair("bar", "a.jpg")

    assert not (ds_path / "a.jpg").exists()
    assert not (ds_path / "a.txt").exists()
    assert not (masks_dir / "a.png").exists()
    assert not (masked_dir / "a.jpg").exists()
    assert not (masked_dir / "a.txt").exists()
    assert not (control_dir / "a.jpg").exists()
