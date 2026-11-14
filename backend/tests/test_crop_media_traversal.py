"""A crop request cannot reach an image outside its dataset.

``crop_media`` takes the client's ``path`` from ``POST /datasets/{name}/crop``
and from the batch-crop worker, opens it and saves the cropped pixels back
over it. Every path it uses comes from ``resolve_edit_targets``, which runs the
client path through ``validate_path_within`` before any filesystem call: a
path that leaves the dataset (``..``, an absolute path, a link) is a 403 and
nothing outside the dataset is read, written or deleted.

Every outside target here is a REAL image larger than the crop target, so a
crop that reached it would change its bytes.

Fixture mirrors ``test_delete_media_pair_traversal.py``: a ``DatasetManager``
rooted in ``tmp_path`` with the DB layer mocked out.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from PIL import Image

from app.core.dataset_manager import DatasetManager
from tests.support.dir_links import link_dir

OUTSIDE_SIZE = (64, 48)
TARGET = (16, 16)
DENIED = "Access denied: path is outside the allowed directory."


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_settings():
    mock_instance = MagicMock()
    mock_instance.get_module_settings.return_value = {}
    mock_instance.update_module_settings = MagicMock()
    with patch(
        "app.core.dataset_manager.get_settings_manager", return_value=mock_instance
    ):
        yield mock_instance


@pytest.fixture()
def manager(tmp_path, mock_settings):
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


@pytest.fixture()
def dataset(manager):
    """Dataset ``foo`` holding a real ``a.jpg`` and its metadata entry."""
    ds = manager.create_dataset("foo")
    _image(Path(ds.path) / "a.jpg", (40, 30))
    ds.media_metadata["a.jpg"] = {
        "width": 40, "height": 30, "size_bytes": 1, "enabled": True,
    }
    return ds


@pytest.fixture()
def outside(dataset):
    """A real image next to (outside) the dataset directory."""
    path = Path(dataset.path).parent / "x.png"
    _image(path, OUTSIDE_SIZE)
    return path


@pytest.fixture()
def routed(manager, monkeypatch):
    """Point the crop route at this test's manager."""
    import app.api.dataset.crop_routes as crop_routes

    monkeypatch.setattr(crop_routes, "dataset_manager", manager)
    return manager


def _image(path: Path, size: tuple[int, int], color: str = "red") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _snapshot(path: Path) -> tuple[bytes, int]:
    return path.read_bytes(), path.stat().st_mtime_ns


def _crop(manager, rel: str) -> None:
    manager.crop_media("foo", rel, *TARGET)


def _assert_refused(manager, rel: str) -> None:
    with pytest.raises(HTTPException) as excinfo:
        _crop(manager, rel)
    assert excinfo.value.status_code == 403
    assert excinfo.value.detail == DENIED


def _assert_no_path_in(text: str, tmp_path: Path) -> None:
    """No form of the test tree's location leaks into a response body."""
    unescaped = text.replace("\\\\", "\\").lower()
    resolved = tmp_path.resolve()
    for form in {str(tmp_path), str(resolved), tmp_path.as_posix(),
                 resolved.as_posix(), tmp_path.name}:
        assert form.lower() not in unescaped, form


# ── The vulnerability ───────────────────────────────────────────────────────


def test_crop_media_rejects_dotdot_and_leaves_the_outside_image_untouched(
    manager, dataset, outside,
):
    before = _snapshot(outside)

    _assert_refused(manager, "../x.png")

    assert _snapshot(outside) == before
    with Image.open(outside) as im:
        assert im.size == OUTSIDE_SIZE


def test_crop_media_rejects_an_absolute_path_outside_the_dataset(
    manager, dataset, outside,
):
    before = _snapshot(outside)

    _assert_refused(manager, str(outside))
    _assert_refused(manager, outside.as_posix())

    assert _snapshot(outside) == before


@pytest.mark.skipif(sys.platform != "win32", reason="a backslash separates only on Windows")
def test_crop_media_rejects_a_backslash_traversal_on_windows(manager, dataset, outside):
    before = _snapshot(outside)

    _assert_refused(manager, "..\\x.png")
    _assert_refused(manager, "sub\\..\\..\\x.png")
    _assert_refused(manager, "../sub/..\\x.png")

    assert _snapshot(outside) == before


def test_crop_media_rejects_a_source_link_that_resolves_outside_the_dataset(
    manager, dataset, tmp_path,
):
    elsewhere = tmp_path / "elsewhere"
    target = elsewhere / "x.png"
    _image(target, OUTSIDE_SIZE)
    link_dir(Path(dataset.path) / "linked", elsewhere)
    before = _snapshot(target)

    _assert_refused(manager, "linked/x.png")

    assert _snapshot(target) == before


def test_crop_media_gives_403_for_an_outside_path_whether_or_not_it_exists(
    manager, dataset, outside,
):
    """No oracle: the answer does not tell whether an outside file exists."""
    assert outside.exists()
    _assert_refused(manager, "../x.png")
    assert not (outside.parent / "ghost.png").exists()
    _assert_refused(manager, "../ghost.png")
    _assert_refused(manager, str(outside.parent / "ghost.png"))


def test_crop_media_refusal_runs_no_side_effect(manager, dataset, outside, monkeypatch):
    import app.core.dataset_manager as dm_mod

    calls = MagicMock()
    monkeypatch.setattr(dm_mod, "invalidate_mask_files", calls.invalidate_mask_files)
    monkeypatch.setattr(
        dm_mod, "refresh_media_metadata_after_change", calls.refresh_metadata,
    )
    monkeypatch.setattr(dm_mod.Image, "open", calls.image_open)
    monkeypatch.setattr(manager, "_persist_media_item", calls.persist)
    monkeypatch.setattr(manager, "_reconcile_overlay_after_edit", calls.reconcile)
    monkeypatch.setattr(manager, "_loop", MagicMock())

    _assert_refused(manager, "../x.png")
    _assert_refused(manager, "../ghost.png")

    assert calls.mock_calls == []
    manager._media_repo.update.assert_not_called()
    manager._loop.assert_not_called()


# ── The route ───────────────────────────────────────────────────────────────


def test_crop_route_answers_403_and_does_not_remap_the_refusal(
    routed, dataset, outside, client, tmp_path,
):
    before = _snapshot(outside)

    for rel in ("../x.png", "../ghost.png", str(outside)):
        resp = client.post("/api/datasets/foo/crop", json={
            "path": rel, "target_width": TARGET[0], "target_height": TARGET[1],
        })
        assert resp.status_code == 403, (rel, resp.text)
        assert resp.json()["detail"] == DENIED
        _assert_no_path_in(resp.text, tmp_path)

    assert _snapshot(outside) == before


def test_crop_route_treats_percent_encoded_dots_as_a_literal_name(
    routed, dataset, outside, client,
):
    """A JSON body is not URL-decoded: ``%2e%2e%2fx.png`` names a file in the
    dataset that does not exist."""
    before = _snapshot(outside)

    resp = client.post("/api/datasets/foo/crop", json={
        "path": "%2e%2e%2fx.png", "target_width": TARGET[0], "target_height": TARGET[1],
    })

    assert resp.status_code == 404
    assert _snapshot(outside) == before


# ── Positive controls ──────────────────────────────────────────────────────


def test_crop_media_still_crops_a_root_level_image(routed, dataset, client):
    a = Path(dataset.path) / "a.jpg"

    resp = client.post("/api/datasets/foo/crop", json={
        "path": "a.jpg", "target_width": TARGET[0], "target_height": TARGET[1],
    })

    assert resp.status_code == 200
    assert resp.json() == {"status": "cropped", "file": "a.jpg"}
    with Image.open(a) as im:
        assert im.size == TARGET
    meta = dataset.media_metadata["a.jpg"]
    assert (meta["width"], meta["height"]) == TARGET
    assert meta["size_bytes"] == a.stat().st_size
    routed._media_repo.update.assert_called_once()
    from app.core.dataset import thumbnails

    with Image.open(thumbnails.thumbnail_path_for(dataset.path, "a.jpg")) as im:
        assert im.size == TARGET


def test_crop_media_still_crops_a_nested_image_and_keeps_its_key(manager, dataset):
    nested = Path(dataset.path) / "sub" / "a.jpg"
    _image(nested, (40, 30))
    dataset.media_metadata["sub/a.jpg"] = {
        "width": 40, "height": 30, "size_bytes": 1, "enabled": True,
    }
    keys_before = set(dataset.media_metadata)

    assert manager.crop_media("foo", "sub/a.jpg", *TARGET) is True

    with Image.open(nested) as im:
        assert im.size == TARGET
    assert set(dataset.media_metadata) == keys_before
    meta = dataset.media_metadata["sub/a.jpg"]
    assert (meta["width"], meta["height"]) == TARGET
    assert meta["size_bytes"] == nested.stat().st_size


def test_the_outside_image_is_real_and_bigger_than_the_target(dataset, outside):
    """The refusal tests prove something only because a crop that reached the
    outside image would change it."""
    with Image.open(outside) as im:
        assert im.size[0] > TARGET[0] and im.size[1] > TARGET[1]
    assert Path(dataset.path).resolve() not in outside.resolve().parents


# ── The batch worker reaches the same method ───────────────────────────────


def test_crop_batch_refuses_a_traversal_item_and_crops_the_rest(
    manager, dataset, outside, monkeypatch,
):
    import app.core.dataset_manager as dm_mod
    from app.core.dataset import crop_batch
    from app.core.tasks.task_manager import task_manager

    task_manager.set_loop(None)
    monkeypatch.setattr(dm_mod, "dataset_manager", manager)
    before = _snapshot(outside)
    items = [
        {"path": "../x.png", "target_width": TARGET[0], "target_height": TARGET[1]},
        {"path": "a.jpg", "target_width": TARGET[0], "target_height": TARGET[1]},
    ]
    t = task_manager.create(type="crop_batch", title="x", total=2, dataset_name="foo")

    crop_batch.run_crop_batch(t.id, dataset_name="foo", items=items, origin="center")

    task = task_manager.get(t.id)
    assert task.status.value == "completed"
    assert (task.ok, task.failed) == (1, 1)
    assert _snapshot(outside) == before
    with Image.open(Path(dataset.path) / "a.jpg") as im:
        assert im.size == TARGET
