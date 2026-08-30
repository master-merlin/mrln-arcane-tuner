import time
from unittest.mock import MagicMock, patch

import pytest

from app.core.dataset_manager import Dataset


def _make_dataset(**overrides) -> Dataset:
    defaults = {
        "id": "test-id",
        "name": "test",
        "path": "/tmp/test",
        "description": "",
        "created_at": time.time(),
        "file_count": 0,
    }
    defaults.update(overrides)
    return Dataset(**defaults)


# ── Dataset CRUD ─────────────────────────────────────────────────────────


@patch("app.api.dataset.crud_routes.dataset_manager")
def test_list_datasets(mock_manager, client):
    mock_manager.list_datasets.return_value = []
    response = client.get("/api/datasets")
    assert response.status_code == 200
    assert response.json() == []
    assert "X-Trace-ID" in response.headers


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_get_dataset_found(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.get_dataset.return_value = _make_dataset(name="found")
    response = client.get("/api/datasets/found")
    assert response.status_code == 200


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_get_dataset_not_found(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.get_dataset.return_value = None
    response = client.get("/api/datasets/nonexistent")
    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset not found"


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_create_dataset_success(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.create_dataset.return_value = _make_dataset(name="new_ds")
    response = client.post("/api/datasets", json={"name": "new_ds"})
    assert response.status_code == 200


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_create_dataset_duplicate(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.create_dataset.side_effect = ValueError("already exists")
    response = client.post("/api/datasets", json={"name": "dup"})
    assert response.status_code == 400  # ValueError → 400


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_update_dataset_success(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.update_dataset.return_value = _make_dataset(name="updated")
    # Route is PATCH, not PUT
    response = client.patch("/api/datasets/old", json={"name": "updated", "description": "new"})
    assert response.status_code == 200


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_update_dataset_not_found(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.update_dataset.side_effect = ValueError("not found")
    response = client.patch("/api/datasets/ghost", json={"name": "ghost", "description": ""})
    assert response.status_code == 400  # ValueError → 400


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_delete_dataset_success(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    response = client.delete("/api/datasets/myds")
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_delete_dataset_not_found(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.delete_dataset.side_effect = ValueError("not found")
    response = client.delete("/api/datasets/ghost")
    assert response.status_code == 404


# ── Scanning ─────────────────────────────────────────────────────────────


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_scan_dataset_success(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.scan_dataset.return_value = _make_dataset(name="scanned", file_count=5)
    response = client.post("/api/datasets/scanned/scan")
    assert response.status_code == 200


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_scan_dataset_not_found(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.scan_dataset.side_effect = ValueError("not found")
    response = client.post("/api/datasets/ghost/scan")
    assert response.status_code == 404


# ── File Upload ──────────────────────────────────────────────────────────


@patch("app.api.dataset.crud_routes.dataset_manager")
def test_upload_file(mock_manager, client, tmp_path):
    """Happy path against a REAL temp directory (not a mocked open()) so this
    also exercises the tmp-write + os.replace path end-to-end (W4.T7)."""
    mock_dataset = MagicMock()
    mock_dataset.path = str(tmp_path)
    mock_manager.get_dataset.return_value = mock_dataset

    response = client.post(
        "/api/datasets/test/upload",
        files={"file": ("test.txt", b"hello world")}
    )
    assert response.status_code == 200
    assert response.json()["filename"] == "test.txt"
    assert response.json()["status"] == "uploaded"

    saved = tmp_path / "test.txt"
    assert saved.read_bytes() == b"hello world"
    # No leftover .tmp file after a successful upload.
    assert not (tmp_path / "test.txt.tmp").exists()


@pytest.mark.asyncio
async def test_upload_file_disconnect_leaves_no_partial_file(tmp_path):
    """W4.T7: a client disconnect mid-upload (UploadFile.read raising after
    delivering some bytes) must NOT leave a truncated file at the final path
    — the next dataset scan would index it as real media."""
    from app.api.dataset.crud_routes import upload_file

    class _DisconnectingUpload:
        filename = "photo.png"

        def __init__(self):
            self._served_first = False

        async def read(self, size: int = -1) -> bytes:
            if not self._served_first:
                self._served_first = True
                return b"partial-bytes-before-disconnect"
            raise ConnectionResetError("client disconnected mid-upload")

    mock_dataset = MagicMock()
    mock_dataset.path = str(tmp_path)

    # ConnectionResetError is an OSError subclass, so the route converts it
    # to a clean 500 rather than crashing the worker — either way, no
    # partial file must land on disk.
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        await upload_file(
            name="ds",
            file=_DisconnectingUpload(),
            slot=0,
            target_stem=None,
            dataset=mock_dataset,
        )

    final_path = tmp_path / "photo.png"
    assert not final_path.exists()
    # No orphaned .tmp file left behind either.
    assert list(tmp_path.iterdir()) == []


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_upload_file_dataset_not_found(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.get_dataset.return_value = None
    response = client.post(
        "/api/datasets/ghost/upload",
        files={"file": ("test.txt", b"data")}
    )
    assert response.status_code == 404


# ── Pairs & Media ────────────────────────────────────────────────────────


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_get_dataset_pairs(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.get_dataset_pairs.return_value = [
        {"stem": "img1", "media_file": "img1.png", "caption_file": "img1.txt", "media_type": "image"}
    ]
    response = client.get("/api/datasets/myds/pairs")
    assert response.status_code == 200
    assert len(response.json()) == 1


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_get_dataset_media_not_found(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.get_dataset.return_value = None
    response = client.get("/api/datasets/ghost/media?image_rel_path=img.png")
    assert response.status_code == 404


# ── Captions ─────────────────────────────────────────────────────────────


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_get_caption_success(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.read_caption.return_value = "a caption"
    response = client.get("/api/datasets/myds/captions/img.txt")
    assert response.status_code == 200
    assert response.json()["content"] == "a caption"


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_get_caption_dataset_not_found(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.read_caption.side_effect = ValueError("not found")
    response = client.get("/api/datasets/ghost/captions/img.txt")
    assert response.status_code == 404


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_save_caption_dataset_not_found(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.save_caption.side_effect = ValueError("not found")
    response = client.put("/api/datasets/ghost/captions/img.txt", json={"content": "text"})
    # W4.T13: matches the GET twin (test_get_caption_dataset_not_found) — an
    # unknown dataset is a 404, not an INTERNAL_ERROR 500.
    assert response.status_code == 404


# ── Lyrics (audio sidecar) ───────────────────────────────────────────────
# Sibling routes to captions above — see crud_routes.py "Lyrics" section.


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_get_lyrics_success(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.read_caption.return_value = "la la la"
    response = client.get("/api/datasets/myds/lyrics/song.lyrics.txt")
    assert response.status_code == 200
    assert response.json()["content"] == "la la la"
    mock_manager.read_caption.assert_called_once_with("myds", "song.lyrics.txt")


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_get_lyrics_dataset_not_found(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.read_caption.side_effect = ValueError("not found")
    response = client.get("/api/datasets/ghost/lyrics/song.lyrics.txt")
    assert response.status_code == 404


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_save_lyrics_success(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    response = client.put(
        "/api/datasets/myds/lyrics/song.lyrics.txt", json={"content": "verse one"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "saved"
    mock_manager.save_lyrics.assert_called_once_with("myds", "song.lyrics.txt", "verse one")


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_save_lyrics_dataset_not_found(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.save_lyrics.side_effect = ValueError("not found")
    response = client.put(
        "/api/datasets/ghost/lyrics/song.lyrics.txt", json={"content": "x"},
    )
    # W4.T13: matches the GET twin (test_get_lyrics_dataset_not_found) — an
    # unknown dataset is a 404, not an INTERNAL_ERROR 500.
    assert response.status_code == 404


# ── Image Enable/Disable ────────────────────────────────────────────────


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_toggle_image_not_found(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.toggle_image_enabled.side_effect = ValueError("not found")
    # Route is PATCH
    response = client.patch("/api/datasets/ghost/images/img.png/enabled", json={"enabled": False})
    assert response.status_code == 404


# ── Cropping ─────────────────────────────────────────────────────────────


@patch("app.api.dataset.crop_routes.dataset_manager")
@patch("app.api.dataset.crop_routes.asyncio.to_thread")
def test_crop_media_dataset_not_found(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.crop_media.side_effect = FileNotFoundError("not found")
    response = client.post("/api/datasets/ghost/crop", json={
        "path": "img.png", "target_width": 512, "target_height": 512
    })
    assert response.status_code == 404


# ── Analysis / Harmonization ─────────────────────────────────────────────


def _make_orientation_analysis(**overrides) -> dict:
    """A well-formed DatasetAnalysisResponse.<orientation> payload —
    matches the shape DatasetManager.analyze_harmonization actually returns."""
    data = {
        "majority_ar": 1.5,
        "majority_ar_display": "3:2",
        "max_long_side_found": 1536,
        "target_resolution": [1536, 1024],
        "count_total": 2,
        "count_majority": 2,
        "images": [
            {
                "path": "a.png",
                "width": 1536,
                "height": 1024,
                "aspect_ratio": 1.5,
                "target_width": 1536,
                "target_height": 1024,
                "similar_count": 0,
                "similar_images": [],
            },
        ],
    }
    data.update(overrides)
    return data


@patch("app.api.dataset.analysis_routes.dataset_manager")
@patch("app.api.dataset.analysis_routes.asyncio.to_thread")
def test_analysis_success(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.analyze_harmonization.return_value = {
        "landscape": _make_orientation_analysis()
    }
    response = client.get("/api/datasets/myds/analysis")
    assert response.status_code == 200
    body = response.json()
    assert body["landscape"]["majority_ar"] == 1.5
    assert body["portrait"] is None


@patch("app.api.dataset.analysis_routes.dataset_manager")
@patch("app.api.dataset.analysis_routes.asyncio.to_thread")
def test_analysis_not_found(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.analyze_harmonization.side_effect = ValueError("not found")
    response = client.get("/api/datasets/ghost/analysis")
    assert response.status_code == 404


@patch("app.api.dataset.analysis_routes.dataset_manager")
@patch("app.api.dataset.analysis_routes.asyncio.to_thread")
def test_analysis_rejects_garbage_resolutions(mock_to_thread, mock_manager, client):
    """W5.T10: resolutions is validated BEFORE the (potentially expensive)
    harmonization analysis runs — garbage input 400s without ever calling
    analyze_harmonization."""

    async def run_sync(func, *args, **kw):
        return func(*args, **kw)

    mock_to_thread.side_effect = run_sync

    response = client.get(
        "/api/datasets/myds/analysis", params={"resolutions": "1024,abc"}
    )
    assert response.status_code == 400
    mock_manager.analyze_harmonization.assert_not_called()


@patch("app.api.dataset.analysis_routes.dataset_manager")
@patch("app.api.dataset.analysis_routes.asyncio.to_thread")
def test_analysis_rejects_non_positive_resolutions(
    mock_to_thread, mock_manager, client
):
    """A resolution of 0 or negative is well-formed as an int but nonsensical
    as a bucket base resolution — must 400, not silently pass through."""

    async def run_sync(func, *args, **kw):
        return func(*args, **kw)

    mock_to_thread.side_effect = run_sync

    response = client.get(
        "/api/datasets/myds/analysis", params={"resolutions": "1024,0"}
    )
    assert response.status_code == 400
    mock_manager.analyze_harmonization.assert_not_called()


@patch("app.api.dataset.analysis_routes.dataset_manager")
@patch("app.api.dataset.analysis_routes.asyncio.to_thread")
def test_analysis_accepts_valid_resolutions(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)

    mock_to_thread.side_effect = run_sync
    mock_manager.analyze_harmonization.return_value = {
        "landscape": _make_orientation_analysis()
    }
    response = client.get(
        "/api/datasets/myds/analysis", params={"resolutions": "1024, 768"}
    )
    assert response.status_code == 200


# ── Delete Media Pair ────────────────────────────────────────────────────


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_delete_media_pair_success(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    response = client.delete("/api/datasets/myds/pairs/img.png")
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_delete_media_pair_not_found(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.delete_media_pair.side_effect = ValueError("not found")
    response = client.delete("/api/datasets/ghost/pairs/img.png")
    assert response.status_code == 404


# ── Enable All ───────────────────────────────────────────────────────────


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_enable_all_images(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    # Mirror the real enable_all_images shape (see test_enable_all_resets_* /
    # test_enable_all_noop_*); the route now validates it via EnableAllResponse.
    mock_manager.enable_all_images.return_value = {"reset_count": 3}
    response = client.post("/api/datasets/myds/images/enable-all")
    assert response.status_code == 200
    assert response.json() == {"reset_count": 3}


# ── Save Caption (success) ───────────────────────────────────────────────


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_save_caption_success(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    response = client.put("/api/datasets/myds/captions/img.txt", json={"content": "new caption"})
    assert response.status_code == 200
    assert response.json()["status"] == "saved"


# ── Calc Crop Targets ────────────────────────────────────────────────────


@patch("app.api.dataset.crop_routes.dataset_manager")
def test_calc_crop_targets_success(mock_manager, client):
    mock_manager.calculate_target_dims.return_value = (1024, 768)
    response = client.post("/api/datasets/myds/calc-crop-targets", json={
        "width": 1024, "height": 768, "aspect_ratio": 1.33,
    })
    assert response.status_code == 200
    data = response.json()
    assert "target_width" in data
    assert "target_height" in data


# ── Download Dataset ─────────────────────────────────────────────────────


class TestDownloadDatasetEndpoint:
    """W4.T11: download streams via a temp file (FileResponse + BackgroundTask
    cleanup) instead of building the whole zip in an in-memory BytesIO."""

    def test_download_dataset_returns_zip_excludes_caches_and_cleans_up_temp_file(
        self, client, tmp_path, monkeypatch,
    ):
        import io
        import zipfile
        from pathlib import Path

        from app.core.dataset_manager import dataset_manager
        import app.api.dataset.crud_routes as crud_mod

        ds_root = tmp_path / "datasets"
        ds_root.mkdir()
        monkeypatch.setattr(dataset_manager, "default_root", str(ds_root))

        ds_path = ds_root / "dl_ds"
        ds_path.mkdir()
        (ds_path / "a.jpg").write_bytes(b"img-a")
        (ds_path / ".cache").mkdir()
        (ds_path / ".cache" / "latents.bin").write_bytes(b"NOPE")

        dataset_manager.create_dataset("dl_ds", path=str(ds_path))

        # Spy on the temp path handed to zipfile so we can assert it's gone
        # (BackgroundTask cleanup) once the response has been fully sent.
        recorded: dict[str, str] = {}
        orig_mkstemp = crud_mod.tempfile.mkstemp

        def _spy_mkstemp(*a, **k):
            fd, p = orig_mkstemp(*a, **k)
            recorded["path"] = p
            return fd, p

        monkeypatch.setattr(crud_mod.tempfile, "mkstemp", _spy_mkstemp)

        try:
            response = client.get("/api/datasets/dl_ds/download")
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/zip"
            assert ".zip" in response.headers.get("content-disposition", "")

            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                names = set(zf.namelist())
            assert "a.jpg" in names
            assert not any(n.startswith(".cache/") for n in names)

            assert "path" in recorded, "route never built a temp-file zip"
            assert not Path(recorded["path"]).exists(), (
                "temp zip was not cleaned up after the response was sent"
            )
        finally:
            dataset_manager.delete_dataset("dl_ds", delete_files=True)

    def test_download_dataset_404_when_directory_missing(
        self, client, tmp_path, monkeypatch,
    ):
        from app.core.dataset_manager import dataset_manager

        ds_root = tmp_path / "datasets"
        ds_root.mkdir()
        monkeypatch.setattr(dataset_manager, "default_root", str(ds_root))

        ds_path = ds_root / "dl_missing"
        ds_path.mkdir()
        dataset_manager.create_dataset("dl_missing", path=str(ds_path))
        import shutil
        shutil.rmtree(ds_path)

        try:
            response = client.get("/api/datasets/dl_missing/download")
            assert response.status_code == 404
        finally:
            dataset_manager.delete_dataset("dl_missing", delete_files=True)


# ── Bump Version ─────────────────────────────────────────────────────────


@patch("app.api.dataset.analysis_routes.dataset_manager")
@patch("app.api.dataset.analysis_routes.asyncio.to_thread")
def test_bump_version_success(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.bump_dataset_version.return_value = "1.0.1"
    response = client.post("/api/datasets/myds/bump?type=patch")
    assert response.status_code == 200
    assert response.json()["version"] == "1.0.1"


@patch("app.api.dataset.analysis_routes.dataset_manager")
@patch("app.api.dataset.analysis_routes.asyncio.to_thread")
def test_bump_version_not_found(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.bump_dataset_version.return_value = None
    response = client.post("/api/datasets/ghost/bump")
    assert response.status_code == 404


# ── Set Version (manual edit) ────────────────────────────────────────────


@patch("app.api.dataset.analysis_routes.dataset_manager")
@patch("app.api.dataset.analysis_routes.asyncio.to_thread")
def test_set_version_success(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.set_dataset_version.return_value = "2.0.0"
    response = client.post("/api/datasets/myds/version", json={"version": "2.0.0"})
    assert response.status_code == 200
    assert response.json()["version"] == "2.0.0"


@patch("app.api.dataset.analysis_routes.dataset_manager")
@patch("app.api.dataset.analysis_routes.asyncio.to_thread")
def test_set_version_not_found(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.set_dataset_version.return_value = None
    response = client.post("/api/datasets/ghost/version", json={"version": "2.0.0"})
    assert response.status_code == 404


@patch("app.api.dataset.analysis_routes.dataset_manager")
@patch("app.api.dataset.analysis_routes.asyncio.to_thread")
def test_set_version_invalid_semver(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.set_dataset_version.side_effect = ValueError("bad")
    response = client.post("/api/datasets/myds/version", json={"version": "v1"})
    assert response.status_code == 400
    assert "bad" in response.json()["detail"]


# ── Thumbnail Endpoint ───────────────────────────────────────────────────


class TestThumbnailEndpoint:
    def test_get_thumbnail_returns_webp(self, client, tmp_path, monkeypatch):
        from PIL import Image
        from app.core.dataset_manager import dataset_manager

        ds_root = tmp_path / "datasets"
        ds_root.mkdir()
        monkeypatch.setattr(dataset_manager, "default_root", str(ds_root))

        ds_path = ds_root / "ep_ds"
        ds_path.mkdir()
        Image.new("RGB", (640, 480), "blue").save(ds_path / "img.jpg")

        dataset_manager.create_dataset("ep_ds", path=str(ds_path))

        response = client.get(
            "/api/datasets/ep_ds/thumbnail", params={"image_rel_path": "img.jpg"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/webp"
        assert "etag" in {k.lower() for k in response.headers}
        assert response.headers.get("cache-control") == "public, max-age=3600"
        assert len(response.content) > 0

        # Cleanup
        dataset_manager.delete_dataset("ep_ds", delete_files=True)

    def test_get_thumbnail_404_when_source_missing(self, client, tmp_path, monkeypatch):
        from app.core.dataset_manager import dataset_manager

        ds_root = tmp_path / "datasets"
        ds_root.mkdir()
        monkeypatch.setattr(dataset_manager, "default_root", str(ds_root))

        ds_path = ds_root / "ep_404"
        ds_path.mkdir()
        dataset_manager.create_dataset("ep_404", path=str(ds_path))

        response = client.get(
            "/api/datasets/ep_404/thumbnail", params={"image_rel_path": "ghost.jpg"},
        )
        assert response.status_code == 404

        dataset_manager.delete_dataset("ep_404", delete_files=True)

    def test_get_thumbnail_rejects_path_traversal(self, client, tmp_path, monkeypatch):
        from app.core.dataset_manager import dataset_manager

        ds_root = tmp_path / "datasets"
        ds_root.mkdir()
        monkeypatch.setattr(dataset_manager, "default_root", str(ds_root))

        ds_path = ds_root / "ep_trav"
        ds_path.mkdir()
        dataset_manager.create_dataset("ep_trav", path=str(ds_path))

        response = client.get(
            "/api/datasets/ep_trav/thumbnail",
            params={"image_rel_path": "../../etc/passwd"},
        )
        # validate_path_within raises HTTPException(403) on escape
        assert response.status_code in (400, 403, 404)

        dataset_manager.delete_dataset("ep_trav", delete_files=True)

    def test_get_thumbnail_serves_the_requested_rendition(
        self, client, tmp_path, monkeypatch,
    ):
        from PIL import Image
        from app.core.dataset_manager import dataset_manager

        ds_root = tmp_path / "datasets"
        ds_root.mkdir()
        monkeypatch.setattr(dataset_manager, "default_root", str(ds_root))

        ds_path = ds_root / "ep_edge"
        ds_path.mkdir()
        Image.new("RGB", (2000, 1200), "blue").save(ds_path / "img.jpg")
        dataset_manager.create_dataset("ep_edge", path=str(ds_path))

        response = client.get(
            "/api/datasets/ep_edge/thumbnail",
            params={"image_rel_path": "img.jpg", "max_edge": 512},
        )
        assert response.status_code == 200
        assert (ds_path / ".thumbnails" / "512" / "img.webp").exists()
        # The size is a directory: it must never appear in the filename,
        # where `foo.png`@512 and `foo@512.png` collided.
        assert not (ds_path / ".thumbnails" / "img@512.webp").exists()

        dataset_manager.delete_dataset("ep_edge", delete_files=True)

    def test_get_thumbnail_rejects_an_unlisted_edge(self, client, tmp_path, monkeypatch):
        """`max_edge` lands in a filename, so it is allowlisted, not clamped."""
        from app.core.dataset_manager import dataset_manager

        ds_root = tmp_path / "datasets"
        ds_root.mkdir()
        monkeypatch.setattr(dataset_manager, "default_root", str(ds_root))

        ds_path = ds_root / "ep_badedge"
        ds_path.mkdir()
        dataset_manager.create_dataset("ep_badedge", path=str(ds_path))

        response = client.get(
            "/api/datasets/ep_badedge/thumbnail",
            params={"image_rel_path": "img.jpg", "max_edge": 9999},
        )
        assert response.status_code == 400

        dataset_manager.delete_dataset("ep_badedge", delete_files=True)


# ── Aggregate excluded_count ─────────────────────────────────────────────


def test_excluded_count_zero_when_no_metadata():
    """A dataset with no media_metadata reports excluded_count == 0."""
    ds = _make_dataset(name="empty")
    assert ds.excluded_count == 0


def test_excluded_count_zero_when_all_enabled():
    """All images enabled (or missing the flag) → excluded_count == 0."""
    ds = _make_dataset(
        name="all_on",
        media_metadata={
            "a.png": {"enabled": True},
            "b.png": {"enabled": True},
            "c.png": {},  # legacy entry, no flag → treated as enabled
        },
    )
    assert ds.excluded_count == 0


def test_excluded_count_counts_disabled():
    """Only entries with enabled is False count as excluded."""
    ds = _make_dataset(
        name="mixed",
        media_metadata={
            "a.png": {"enabled": True},
            "b.png": {"enabled": False},
            "c.png": {"enabled": False},
            "d.png": {},  # legacy → enabled
        },
    )
    assert ds.excluded_count == 2


@patch("app.api.dataset.crud_routes.dataset_manager")
def test_excluded_count_reaches_list_response(mock_manager, client):
    """The LIST endpoint payload exposes excluded_count per row."""
    ds = _make_dataset(
        name="listed",
        media_metadata={
            "a.png": {"enabled": True},
            "b.png": {"enabled": False},
        },
    )
    mock_manager.list_datasets.return_value = [ds]
    response = client.get("/api/datasets")
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 1
    assert rows[0]["excluded_count"] == 1


# ── Enable All — bulk-persist optimization ───────────────────────────────


def test_enable_all_bulk_persists_once_and_invalidates(monkeypatch):
    from app.core.dataset_manager import Dataset, dataset_manager

    ds = Dataset(id="ea-i", name="ea", path="/tmp/ea", created_at=0.0,
                 media_metadata={
                     "a.jpg": {"enabled": False},
                     "b.jpg": {"enabled": False},
                     "c.jpg": {"enabled": True},
                 })
    dataset_manager.datasets["ea"] = ds
    calls = {"persist_dataset": 0, "persist_item": 0, "invalidate": 0}
    monkeypatch.setattr(dataset_manager, "_persist_dataset",
                        lambda d: calls.__setitem__("persist_dataset", calls["persist_dataset"] + 1))
    monkeypatch.setattr(dataset_manager, "_persist_media_item",
                        lambda d, k: calls.__setitem__("persist_item", calls["persist_item"] + 1))
    monkeypatch.setattr(dataset_manager, "_emit_dataset_invalidated",
                        lambda name: calls.__setitem__("invalidate", calls["invalidate"] + 1))
    try:
        res = dataset_manager.enable_all_images("ea")
        assert res["reset_count"] == 2
        assert ds.media_metadata["a.jpg"]["enabled"] is True
        assert ds.media_metadata["b.jpg"]["enabled"] is True
        assert calls["persist_dataset"] == 1     # one bulk write
        assert calls["persist_item"] == 0        # NO per-item writes
        assert calls["invalidate"] == 1          # one coarse broadcast
    finally:
        dataset_manager.datasets.pop("ea", None)


def test_enable_all_noop_when_all_enabled(monkeypatch):
    from app.core.dataset_manager import Dataset, dataset_manager

    ds = Dataset(id="ea2-i", name="ea2", path="/tmp/ea2", created_at=0.0,
                 media_metadata={"a.jpg": {"enabled": True}})
    dataset_manager.datasets["ea2"] = ds
    calls = {"persist_dataset": 0, "invalidate": 0}
    monkeypatch.setattr(dataset_manager, "_persist_dataset",
                        lambda d: calls.__setitem__("persist_dataset", calls["persist_dataset"] + 1))
    monkeypatch.setattr(dataset_manager, "_emit_dataset_invalidated",
                        lambda name: calls.__setitem__("invalidate", calls["invalidate"] + 1))
    try:
        res = dataset_manager.enable_all_images("ea2")
        assert res["reset_count"] == 0
        assert calls["persist_dataset"] == 0     # nothing changed → no write
        assert calls["invalidate"] == 0
    finally:
        dataset_manager.datasets.pop("ea2", None)

