import time
from unittest.mock import MagicMock, patch
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


@patch("app.api.dataset.crud_routes.dataset_manager")
@patch("app.api.dataset.crud_routes.asyncio.to_thread")
def test_scan_all_datasets(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.scan_all_datasets.return_value = [_make_dataset(name="ds1")]
    response = client.post("/api/datasets/scan-all")
    assert response.status_code == 200


# ── File Upload ──────────────────────────────────────────────────────────


@patch("app.api.dataset.crud_routes.dataset_manager")
def test_upload_file(mock_manager, client):
    mock_dataset = MagicMock()
    mock_dataset.path = "/tmp/test"
    mock_manager.get_dataset.return_value = mock_dataset

    with patch("app.api.dataset.crud_routes.open", create=True):
        with patch("app.api.dataset.crud_routes.shutil.copyfileobj"):
            response = client.post(
                "/api/datasets/test/upload",
                files={"file": ("test.txt", b"hello world")}
            )
            assert response.status_code == 200
            assert response.json()["filename"] == "test.txt"
            assert response.json()["status"] == "uploaded"


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
    assert response.status_code == 500  # ValueError → 500 in save_caption route


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


@patch("app.api.dataset.analysis_routes.dataset_manager")
@patch("app.api.dataset.analysis_routes.asyncio.to_thread")
def test_analysis_success(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.analyze_harmonization.return_value = {"landscape": {"majority_ar": 1.5}}
    response = client.get("/api/datasets/myds/analysis")
    assert response.status_code == 200


@patch("app.api.dataset.analysis_routes.dataset_manager")
@patch("app.api.dataset.analysis_routes.asyncio.to_thread")
def test_analysis_not_found(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.analyze_harmonization.side_effect = ValueError("not found")
    response = client.get("/api/datasets/ghost/analysis")
    assert response.status_code == 404


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
    mock_manager.enable_all_images.return_value = {"status": "all_enabled"}
    response = client.post("/api/datasets/myds/images/enable-all")
    assert response.status_code == 200


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


# ── Harmonize Files ──────────────────────────────────────────────────────


@patch("app.api.dataset.analysis_routes.dataset_manager")
@patch("app.api.dataset.analysis_routes.asyncio.to_thread")
def test_harmonize_files_success(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.harmonize_files.return_value = {"status": "harmonized", "renamed": 3}
    response = client.post("/api/datasets/myds/harmonize")
    assert response.status_code == 200


@patch("app.api.dataset.analysis_routes.dataset_manager")
@patch("app.api.dataset.analysis_routes.asyncio.to_thread")
def test_harmonize_files_error(mock_to_thread, mock_manager, client):
    async def run_sync(func, *args, **kw):
        return func(*args, **kw)
    mock_to_thread.side_effect = run_sync
    mock_manager.harmonize_files.side_effect = FileNotFoundError("not found")
    response = client.post("/api/datasets/ghost/harmonize")
    assert response.status_code == 404

