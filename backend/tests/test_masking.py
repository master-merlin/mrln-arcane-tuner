from unittest.mock import AsyncMock, MagicMock, patch


# ── Generate Mask ────────────────────────────────────────────────────────


@patch("app.api.masking_routes.masking_service")
@patch("app.api.masking_routes.dataset_manager")
@patch("app.api.masking_routes.asyncio.to_thread")
def test_generate_mask_success(mock_to_thread, mock_manager, mock_service_instance, client, tmp_path):
    async def run_sync(func, *args, **kwargs):
        return func(*args, **kwargs)
    mock_to_thread.side_effect = run_sync

    # Create a real temp file, so Path(...).exists() returns True
    img_file = tmp_path / "image.jpg"
    img_file.write_bytes(b"\xff\xd8\xff\xe0")  # minimal JPEG header

    mock_mask_image = MagicMock()
    mock_service_instance.generate_mask.return_value = mock_mask_image

    mock_dataset = MagicMock()
    mock_dataset.path = str(tmp_path)
    mock_dataset.media_metadata = {"image.jpg": {"has_mask": False}}
    mock_manager.get_dataset.return_value = mock_dataset
    mock_manager.update_media_flags_async = AsyncMock()

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
    # W4.T14: mutate+persist moved into DatasetManager.update_media_flags —
    # the route now just calls it with the field(s) to set.
    mock_manager.update_media_flags_async.assert_called_once_with(
        "test_ds", "image.jpg", has_mask=True,
    )


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


@patch("app.api.masking_routes.safe_remove")
@patch("app.api.masking_routes.masking_service")
@patch("app.api.masking_routes.dataset_manager")
@patch("app.api.masking_routes.asyncio.to_thread")
def test_delete_mask_success(mock_to_thread, mock_manager, mock_service_instance, mock_safe_remove, client, tmp_path):
    async def run_sync(func, *args, **kwargs):
        return func(*args, **kwargs)
    mock_to_thread.side_effect = run_sync

    # Create a mask file so Path(...).exists() returns True
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    mask_file = masks_dir / "image.png"
    mask_file.write_bytes(b"\x89PNG")

    mock_dataset = MagicMock()
    mock_dataset.path = str(tmp_path)
    mock_dataset.media_metadata = {
        "image.jpg": {"has_mask": True, "has_masked": True, "has_masked_caption": True, "mask_info": {}},
    }
    mock_manager.get_dataset.return_value = mock_dataset
    mock_manager.update_media_flags_async = AsyncMock()

    response = client.delete("/api/datasets/test_ds/masking/delete?image_rel_path=image.jpg")

    assert response.status_code == 200
    assert response.json()["status"] == "deleted"
    # W4.T14: mutate+persist moved into DatasetManager.update_media_flags —
    # REMOVE_FIELD (accessed via the mocked manager) pops mask_info.
    mock_manager.update_media_flags_async.assert_called_once_with(
        "test_ds", "image.jpg",
        has_mask=False, has_masked=False, has_masked_caption=False,
        mask_info=mock_manager.REMOVE_FIELD,
    )


@patch("app.api.masking_routes.safe_remove")
@patch("app.api.masking_routes.masking_service")
@patch("app.api.masking_routes.dataset_manager")
@patch("app.api.masking_routes.asyncio.to_thread")
def test_delete_mask_not_found(mock_to_thread, mock_manager, mock_service, mock_safe_remove, client, tmp_path):
    async def run_sync(func, *args, **kwargs):
        return func(*args, **kwargs)
    mock_to_thread.side_effect = run_sync

    # No mask file exists in tmp_path, so Path(...).exists() returns False
    mock_dataset = MagicMock()
    mock_dataset.path = str(tmp_path)
    mock_manager.get_dataset.return_value = mock_dataset

    response = client.delete("/api/datasets/test_ds/masking/delete?image_rel_path=image.jpg")
    assert response.status_code == 404


# ── Apply Mask ───────────────────────────────────────────────────────────


@patch("app.api.masking_routes.masking_service")
@patch("app.api.masking_routes.dataset_manager")
@patch("app.api.masking_routes.asyncio.to_thread")
def test_apply_mask_success(mock_to_thread, mock_manager, mock_service, client, tmp_path):
    async def run_sync(func, *args, **kwargs):
        return func(*args, **kwargs)
    mock_to_thread.side_effect = run_sync

    # Create mask file so Path(...).exists() returns True
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    (masks_dir / "image.png").write_bytes(b"\x89PNG")

    mock_dataset = MagicMock()
    mock_dataset.path = str(tmp_path)
    mock_dataset.media_metadata = {"image.jpg": {"has_masked": False}}
    mock_manager.get_dataset.return_value = mock_dataset
    mock_manager.update_media_flags_async = AsyncMock()

    payload = {
        "dataset_name": "test_ds",
        "image_rel_path": "image.jpg",
        "opacity": 0.5
    }
    response = client.post("/api/datasets/test_ds/masking/apply", json=payload)
    assert response.status_code == 200
    mock_manager.update_media_flags_async.assert_called_once_with(
        "test_ds", "image.jpg", has_masked=True,
    )


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
@patch("app.api.masking_routes.asyncio.to_thread")
def test_preview_mask_success(mock_to_thread, mock_manager, mock_service, client, tmp_path):
    async def run_sync(func, *args, **kwargs):
        return func(*args, **kwargs)
    mock_to_thread.side_effect = run_sync

    # Create mask file so Path(...).exists() returns True
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    (masks_dir / "image.png").write_bytes(b"\x89PNG")

    mock_dataset = MagicMock()
    mock_dataset.path = str(tmp_path)
    mock_manager.get_dataset.return_value = mock_dataset

    # generate_preview returns a PIL Image
    from PIL import Image
    mock_img = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
    mock_service.generate_preview.return_value = mock_img

    response = client.get("/api/datasets/test_ds/masking/preview?image_rel_path=image.jpg&opacity=0.5")
    assert response.status_code == 200
