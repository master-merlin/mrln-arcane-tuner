"""A crop's errors keep their codes; an oversize target is a 400.

``crop_media`` raised "Target dimensions cannot be larger than source." as a
``ValueError`` inside the ``try`` that turns every failure into a
``RuntimeError``, so the route's ``except ValueError -> 400`` never saw it and
the user got a 500. Only that check leaves as a ``ValueError`` now: a
``ValueError`` PIL itself raises is still a ``RuntimeError`` and a 500 with
the generic envelope.

The shared ``client`` re-raises server exceptions, so every 500 here goes
through a local ``TestClient(app, raise_server_exceptions=False)``.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.dataset_manager import DatasetManager

SOURCE_SIZE = (40, 30)
TOO_LARGE = "Target dimensions cannot be larger than source."
GENERIC_500 = "A critical error occurred. Please check server logs."


@pytest.fixture()
def manager(tmp_path, monkeypatch):
    import app.api.dataset.crop_routes as crop_routes

    settings = MagicMock()
    settings.get_module_settings.return_value = {}
    (tmp_path / "datasets").mkdir()
    with patch("app.core.dataset_manager.get_settings_manager", return_value=settings), \
         patch.object(DatasetManager, "__init__", lambda self, **kw: None):
        mgr = DatasetManager()
    mgr.root_dir = str(tmp_path)
    mgr.storage_file = str(tmp_path / "dataset_locations.json")
    mgr.default_root = str(tmp_path / "datasets")
    mgr.settings_manager = settings
    mgr.datasets = {}
    mgr._loop = None
    mgr._db = MagicMock()
    mgr._dataset_repo = MagicMock()
    mgr._media_repo = MagicMock()
    monkeypatch.setattr(crop_routes, "dataset_manager", mgr)
    return mgr


@pytest.fixture()
def dataset(manager):
    ds = manager.create_dataset("foo")
    Image.new("RGB", SOURCE_SIZE, "red").save(Path(ds.path) / "a.jpg")
    ds.media_metadata["a.jpg"] = {
        "width": SOURCE_SIZE[0], "height": SOURCE_SIZE[1], "size_bytes": 1, "enabled": True,
    }
    return ds


@pytest.fixture()
def client_no_raise():
    from app.main import app

    return TestClient(app, raise_server_exceptions=False)


def _post(client, path: str, w: int, h: int, name: str = "foo"):
    return client.post(f"/api/datasets/{name}/crop", json={
        "path": path, "target_width": w, "target_height": h,
    })


def _assert_no_path_in(text: str, tmp_path: Path) -> None:
    unescaped = text.replace("\\\\", "\\").lower()
    resolved = tmp_path.resolve()
    for form in {str(tmp_path), str(resolved), tmp_path.as_posix(),
                 resolved.as_posix(), tmp_path.name}:
        assert form.lower() not in unescaped, form


@pytest.mark.parametrize("w,h", [(41, 30), (40, 31), (80, 60)])
def test_crop_route_answers_400_for_a_target_larger_than_the_source(
    dataset, client, tmp_path, w, h,
):
    a = Path(dataset.path) / "a.jpg"
    before = a.read_bytes()

    resp = _post(client, "a.jpg", w, h)

    assert resp.status_code == 400
    assert resp.json()["detail"] == TOO_LARGE
    _assert_no_path_in(resp.text, tmp_path)
    assert a.read_bytes() == before


def test_crop_media_raises_value_error_for_a_target_larger_than_the_source(
    manager, dataset,
):
    with pytest.raises(ValueError) as excinfo:
        manager.crop_media("foo", "a.jpg", 41, 30)

    assert str(excinfo.value) == TOO_LARGE
    assert not isinstance(excinfo.value, RuntimeError)
    manager._media_repo.update.assert_not_called()


def test_a_value_error_from_pil_itself_stays_a_runtime_error(manager, dataset):
    """PIL opens a PNG by its bytes but refuses to save to an unknown
    extension with a ``ValueError``: that is a failed crop, not a bad request."""
    odd = Path(dataset.path) / "a.xyz"
    Image.new("RGB", SOURCE_SIZE, "red").save(odd, format="PNG")
    before = odd.read_bytes()
    with Image.open(odd) as im, pytest.raises(ValueError):
        im.save(odd.with_name("probe.xyz"))

    with pytest.raises(RuntimeError) as excinfo:
        manager.crop_media("foo", "a.xyz", 16, 16)

    assert not isinstance(excinfo.value, ValueError)
    assert odd.read_bytes() == before


def test_a_pil_failure_answers_500_with_the_generic_envelope(
    dataset, client_no_raise, tmp_path,
):
    Image.new("RGB", SOURCE_SIZE, "red").save(Path(dataset.path) / "a.xyz", format="PNG")

    resp = _post(client_no_raise, "a.xyz", 16, 16)

    assert resp.status_code == 500
    assert resp.json()["detail"] == GENERIC_500
    _assert_no_path_in(resp.text, tmp_path)


def test_an_undecodable_image_is_still_a_500_without_a_path_in_the_body(
    dataset, client_no_raise, tmp_path,
):
    (Path(dataset.path) / "broken.jpg").write_bytes(b"not an image at all")

    resp = _post(client_no_raise, "broken.jpg", 16, 16)

    assert resp.status_code == 500
    assert resp.json()["detail"] == GENERIC_500
    _assert_no_path_in(resp.text, tmp_path)


def test_unknown_dataset_missing_file_and_audio_keep_their_codes(dataset, client):
    (Path(dataset.path) / "song.wav").write_bytes(b"RIFF....WAVE")

    assert _post(client, "a.jpg", 16, 16, name="ghost").status_code == 400
    assert _post(client, "missing.jpg", 16, 16).status_code == 404
    audio = _post(client, "song.wav", 16, 16)
    assert audio.status_code == 400
    assert audio.json()["detail"] == "Crop is not supported for audio files."


def test_a_same_size_target_is_still_a_no_op(manager, dataset):
    a = Path(dataset.path) / "a.jpg"
    before = (a.read_bytes(), os.stat(a).st_mtime_ns)

    assert manager.crop_media("foo", "a.jpg", *SOURCE_SIZE) is None

    assert (a.read_bytes(), os.stat(a).st_mtime_ns) == before
    manager._media_repo.update.assert_not_called()
