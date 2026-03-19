from unittest.mock import MagicMock, patch
from pathlib import Path
from PIL import Image
from app.core.captioning.models.qwen3_vl import Qwen3VLModel
from app.core.captioning.models.joycaption import JoyCaptionModel
from app.core.captioning.models.youtu_vl import YoutuVLModel
from app.core.captioning.models.florence2 import Florence2Model


# ── API Route Tests ──────────────────────────────────────────────────────


@patch("app.api.caption_routes.CaptionService")
@patch("app.core.dataset_manager.dataset_manager")
@patch("app.api.caption_routes.validate_path_within")
@patch("app.api.caption_routes.asyncio.to_thread")
def test_generate_caption_success(mock_to_thread, mock_validate, mock_manager, mock_service_cls, client):
    # Mock to_thread to return an awaitable
    async def run_sync(func, *args, **kwargs):
        return func(*args, **kwargs)
    mock_to_thread.side_effect = run_sync

    # Mock path validation (return a fake Path that "exists")
    fake_path = MagicMock(spec=Path)
    fake_path.exists.return_value = True
    mock_validate.return_value = fake_path

    # Mock CaptionService
    mock_service_instance = MagicMock()
    mock_service_instance.generate_caption.return_value = "A beautiful sunset"
    mock_service_cls.get_instance.return_value = mock_service_instance

    # Mock Dataset Manager
    mock_dataset = MagicMock()
    mock_dataset.path = "/tmp/datasets/test_ds"
    mock_manager.datasets.get.return_value = mock_dataset
    mock_manager.datasets.__contains__.return_value = True
    mock_manager.datasets.__getitem__.return_value = mock_dataset

    payload = {
        "dataset_name": "test_ds",
        "image_rel_path": "image.jpg",
        "model_id": "moondream",
        "params": {"temperature": 0.7}
    }

    response = client.post("/api/captions/generate", json=payload)
    
    assert response.status_code == 200
    assert response.json() == {"caption": "A beautiful sunset"}
    mock_service_instance.generate_caption.assert_called_once()


@patch("app.api.caption_routes.CaptionService")
@patch("app.core.dataset_manager.dataset_manager")
@patch("app.api.caption_routes.asyncio.to_thread")
def test_generate_caption_dataset_not_found(mock_to_thread, mock_manager, mock_service_cls, client):
    async def run_sync(func, *args, **kwargs):
        return func(*args, **kwargs)
    mock_to_thread.side_effect = run_sync

    # Mock Manager to return None
    mock_manager.get_dataset.return_value = None

    payload = {
        "dataset_name": "unknown_ds",
        "image_rel_path": "image.jpg",
        "model_id": "moondream",
        "params": {}
    }

    response = client.post("/api/captions/generate", json=payload)
    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset not found"

@patch("app.api.caption_routes.CaptionService")
@patch("app.api.caption_routes.asyncio.to_thread")
def test_unload_models(mock_to_thread, mock_service_cls, client):
    async def run_sync(func, *args, **kwargs):
        return func(*args, **kwargs)
    mock_to_thread.side_effect = run_sync

    mock_service_instance = MagicMock()
    mock_service_cls.get_instance.return_value = mock_service_instance

    response = client.delete("/api/captions/unload")
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_service_instance.unload_models.assert_called_once()


@patch("app.api.caption_routes.CaptionService")
@patch("app.core.dataset_manager.dataset_manager")
@patch("app.api.caption_routes.validate_path_within")
@patch("app.api.caption_routes.asyncio.to_thread")
def test_system_prompt_propagation_through_route(
    mock_to_thread, mock_validate, mock_manager, mock_service_cls, client
):
    """Verify that system_prompt from the request body is merged into params."""
    async def run_sync(func, *args, **kwargs):
        return func(*args, **kwargs)
    mock_to_thread.side_effect = run_sync

    fake_path = MagicMock(spec=Path)
    fake_path.exists.return_value = True
    mock_validate.return_value = fake_path

    mock_service_instance = MagicMock()
    mock_service_instance.generate_caption.return_value = "test caption"
    mock_service_cls.get_instance.return_value = mock_service_instance

    mock_dataset = MagicMock()
    mock_dataset.path = "/tmp/datasets/test_ds"
    mock_manager.datasets.get.return_value = mock_dataset
    mock_manager.datasets.__contains__.return_value = True

    payload = {
        "dataset_name": "test_ds",
        "image_rel_path": "image.jpg",
        "model_id": "qwen3-vl-4B-Instruct",
        "params": {"temperature": 0.5},
        "system_prompt": "Respond in JSON format with keys: subject, scene, style"
    }

    response = client.post("/api/captions/generate", json=payload)
    assert response.status_code == 200

    # Verify system_prompt was merged into the params dict
    call_kwargs = mock_service_instance.generate_caption.call_args
    passed_params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params")
    assert passed_params["system_prompt"] == "Respond in JSON format with keys: subject, scene, style"
    assert passed_params["temperature"] == 0.5


# ── QWEN VL Image Resize Tests ──────────────────────────────────────────


class TestQwen3VLResize:
    """Tests for Qwen3VL image resize on inference."""

    def setup_method(self):
        self.service = MagicMock()
        self.model = Qwen3VLModel(self.service)

    def test_resize_large_landscape_image(self):
        """Images >1280px on the long side should be resized."""
        image = Image.new("RGB", (2560, 1440))  # landscape, long side 2560
        result = self.model._resize_for_inference(image)

        assert max(result.width, result.height) == 1280
        assert result.width == 1280
        assert result.height == 720  # aspect ratio preserved

    def test_resize_large_portrait_image(self):
        """Portrait images >1280px should also be resized."""
        image = Image.new("RGB", (960, 1920))  # portrait, long side 1920
        result = self.model._resize_for_inference(image)

        assert max(result.width, result.height) == 1280
        assert result.height == 1280
        assert result.width == 640

    def test_no_resize_small_image(self):
        """Images ≤1280px should be returned untouched."""
        image = Image.new("RGB", (1024, 768))
        result = self.model._resize_for_inference(image)

        assert result is image  # exact same object, not a copy
        assert result.width == 1024
        assert result.height == 768

    def test_no_resize_exact_boundary(self):
        """Images exactly at 1280px boundary should not be resized."""
        image = Image.new("RGB", (1280, 720))
        result = self.model._resize_for_inference(image)

        assert result is image
        assert result.width == 1280

    def test_resize_square_image(self):
        """Square images >1280px should be resized to 1280x1280."""
        image = Image.new("RGB", (2000, 2000))
        result = self.model._resize_for_inference(image)

        assert result.width == 1280
        assert result.height == 1280

    def test_resize_custom_max_long_side(self):
        """max_long_side from params should control the resize threshold."""
        image = Image.new("RGB", (1024, 768))
        # With default (1280), this image should NOT be resized
        result_default = self.model._resize_for_inference(image)
        assert result_default is image

        # With 768, it SHOULD be resized (long side 1024 > 768)
        result_768 = self.model._resize_for_inference(image, max_long_side=768)
        assert result_768 is not image
        assert max(result_768.width, result_768.height) == 768


# ── System Prompt Routing Tests ──────────────────────────────────────────


class TestSystemPromptRouting:
    """Tests for resolve_prompt() across all captioning models."""

    def test_qwen3_vl_custom_prompt_in_system_role(self):
        """Qwen3VL should place the custom prompt in the system role."""
        model = Qwen3VLModel(MagicMock())
        params = {"system_prompt": "Return JSON: {subject, scene, style}"}

        result = model.resolve_prompt(params)
        assert result == "Return JSON: {subject, scene, style}"

    def test_qwen3_vl_default_prompt(self):
        """Qwen3VL should return default system prompt when none provided."""
        model = Qwen3VLModel(MagicMock())
        result = model.resolve_prompt({})
        assert "image descriptions" in result.lower()

    def test_joycaption_custom_prompt(self):
        """JoyCaption should accept a custom prompt via resolve_prompt."""
        model = JoyCaptionModel(MagicMock())
        params = {"system_prompt": "Write a Midjourney-style prompt."}

        result = model.resolve_prompt(params)
        assert result == "Write a Midjourney-style prompt."

    def test_youtu_vl_custom_prompt(self):
        """YoutuVL should accept a custom prompt via resolve_prompt."""
        model = YoutuVLModel(MagicMock())
        params = {"system_prompt": "Describe technical details."}

        result = model.resolve_prompt(params)
        assert result == "Describe technical details."

    def test_youtu_vl_default_prompt(self):
        """YoutuVL without custom prompt falls back to default."""
        model = YoutuVLModel(MagicMock())
        result = model.resolve_prompt({})
        assert "describe" in result.lower()

    def test_florence2_returns_none(self):
        """Florence2 should return None — it uses task-type tags only."""
        model = Florence2Model(MagicMock())
        result = model.resolve_prompt({"system_prompt": "Anything"})
        assert result is None
