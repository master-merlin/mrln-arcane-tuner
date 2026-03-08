from unittest.mock import MagicMock, patch


# ── Generate Mask ────────────────────────────────────────────────────────


@patch("app.api.masking_routes.masking_service")
@patch("app.api.masking_routes.dataset_manager")
@patch("app.api.masking_routes.os")
@patch("app.api.masking_routes.asyncio.to_thread")
def test_generate_mask_success(mock_to_thread, mock_os, mock_manager, mock_service_instance, client):
    async def run_sync(func, *args, **kwargs):
        return func(*args, **kwargs)
    mock_to_thread.side_effect = run_sync

    mock_os.path.join.side_effect = lambda *args: "/".join(args)
    mock_os.path.exists.return_value = True
    mock_os.path.splitext.return_value = ("image", ".jpg")
    mock_os.path.basename.return_value = "image.jpg"
    mock_os.makedirs = MagicMock()

    mock_mask_image = MagicMock()
    mock_service_instance.generate_mask.return_value = mock_mask_image

    mock_dataset = MagicMock()
    mock_dataset.path = "/tmp/ds"
    mock_manager.get_dataset.return_value = mock_dataset

    payload = {
        "dataset_name": "test_ds",
        "image_rel_path": "image.jpg",
        "model_id": "sam3",
        "params": {}
    }

    response = client.post("/api/datasets/test_ds/masking/generate", json=payload)

    assert response.status_code == 200
    assert response.json()["mask_path"] == "masks/image.png"
    mock_mask_image.save.assert_called_once()
    mock_manager.scan_dataset.assert_called_once_with("test_ds")


@patch("app.api.masking_routes.masking_service")
@patch("app.api.masking_routes.dataset_manager")
@patch("app.api.masking_routes.asyncio.to_thread")
def test_generate_mask_dataset_not_found(mock_to_thread, mock_manager, mock_service, client):
    async def run_sync(func, *args, **kwargs):
        return func(*args, **kwargs)
    mock_to_thread.side_effect = run_sync
    mock_manager.get_dataset.return_value = None

    payload = {
        "dataset_name": "ghost",
        "image_rel_path": "image.jpg",
        "model_id": "sam3",
        "params": {}
    }
    response = client.post("/api/datasets/ghost/masking/generate", json=payload)
    assert response.status_code == 404


# ── Delete Mask ──────────────────────────────────────────────────────────


@patch("app.api.masking_routes.masking_service")
@patch("app.api.masking_routes.dataset_manager")
@patch("app.api.masking_routes.os")
@patch("app.api.masking_routes.asyncio.to_thread")
def test_delete_mask_success(mock_to_thread, mock_os, mock_manager, mock_service_instance, client):
    async def run_sync(func, *args, **kwargs):
        return func(*args, **kwargs)
    mock_to_thread.side_effect = run_sync

    mock_os.path.join.side_effect = lambda *args: "/".join(args)
    mock_os.path.exists.return_value = True
    mock_os.path.splitext.return_value = ("image", ".jpg")
    mock_os.path.basename.return_value = "image.jpg"

    mock_dataset = MagicMock()
    mock_dataset.path = "/tmp/ds"
    mock_manager.get_dataset.return_value = mock_dataset

    response = client.delete("/api/datasets/test_ds/masking/delete?image_rel_path=image.jpg")

    assert response.status_code == 200
    assert response.json()["status"] == "deleted"
    mock_os.remove.assert_called_once()


@patch("app.api.masking_routes.masking_service")
@patch("app.api.masking_routes.dataset_manager")
@patch("app.api.masking_routes.os")
@patch("app.api.masking_routes.asyncio.to_thread")
def test_delete_mask_not_found(mock_to_thread, mock_os, mock_manager, mock_service, client):
    async def run_sync(func, *args, **kwargs):
        return func(*args, **kwargs)
    mock_to_thread.side_effect = run_sync

    mock_os.path.join.side_effect = lambda *args: "/".join(args)
    mock_os.path.exists.return_value = False  # Mask file doesn't exist
    mock_os.path.splitext.return_value = ("image", ".jpg")
    mock_os.path.basename.return_value = "image.jpg"

    mock_dataset = MagicMock()
    mock_dataset.path = "/tmp/ds"
    mock_manager.get_dataset.return_value = mock_dataset

    response = client.delete("/api/datasets/test_ds/masking/delete?image_rel_path=image.jpg")
    assert response.status_code == 404


# ── Apply Mask ───────────────────────────────────────────────────────────


@patch("app.api.masking_routes.masking_service")
@patch("app.api.masking_routes.dataset_manager")
@patch("app.api.masking_routes.os")
@patch("app.api.masking_routes.asyncio.to_thread")
def test_apply_mask_success(mock_to_thread, mock_os, mock_manager, mock_service, client):
    async def run_sync(func, *args, **kwargs):
        return func(*args, **kwargs)
    mock_to_thread.side_effect = run_sync

    mock_os.path.join.side_effect = lambda *args: "/".join(args)
    mock_os.path.exists.return_value = True
    mock_os.path.splitext.return_value = ("image", ".jpg")
    mock_os.path.basename.return_value = "image.jpg"

    mock_dataset = MagicMock()
    mock_dataset.path = "/tmp/ds"
    mock_manager.get_dataset.return_value = mock_dataset

    payload = {
        "dataset_name": "test_ds",
        "image_rel_path": "image.jpg",
        "opacity": 0.5
    }
    response = client.post("/api/datasets/test_ds/masking/apply", json=payload)
    assert response.status_code == 200


# ── Preview Mask ─────────────────────────────────────────────────────────


@patch("app.api.masking_routes.masking_service")
@patch("app.api.masking_routes.dataset_manager")
@patch("app.api.masking_routes.asyncio.to_thread")
def test_preview_mask_not_found(mock_to_thread, mock_manager, mock_service_instance, client):
    async def run_sync(func, *args, **kwargs):
        return func(*args, **kwargs)
    mock_to_thread.side_effect = run_sync
    mock_manager.get_dataset.return_value = None

    response = client.get("/api/datasets/test_ds/masking/preview?image_rel_path=image.jpg")

    assert response.status_code == 404


@patch("app.api.masking_routes.masking_service")
@patch("app.api.masking_routes.dataset_manager")
@patch("app.api.masking_routes.os")
@patch("app.api.masking_routes.asyncio.to_thread")
def test_preview_mask_success(mock_to_thread, mock_os, mock_manager, mock_service, client):
    async def run_sync(func, *args, **kwargs):
        return func(*args, **kwargs)
    mock_to_thread.side_effect = run_sync

    mock_os.path.join.side_effect = lambda *args: "/".join(args)
    mock_os.path.exists.return_value = True

    mock_dataset = MagicMock()
    mock_dataset.path = "/tmp/ds"
    mock_manager.get_dataset.return_value = mock_dataset

    # generate_preview returns a PIL Image
    from PIL import Image
    mock_img = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
    mock_service.generate_preview.return_value = mock_img

    response = client.get("/api/datasets/test_ds/masking/preview?image_rel_path=image.jpg&opacity=0.5")
    assert response.status_code == 200
