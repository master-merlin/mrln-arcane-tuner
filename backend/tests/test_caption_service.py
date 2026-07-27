"""
Tests for CaptionService — covers singleton, plugin dispatch, model switching, error handling.
"""

import pytest
from unittest.mock import MagicMock
from PIL import Image


class TestCaptionServiceSingleton:
    def test_get_instance_returns_same(self):
        from app.core.captioning.caption_service import CaptionService
        CaptionService._instance = None
        a = CaptionService.get_instance()
        b = CaptionService.get_instance()
        assert a is b
        CaptionService._instance = None

    def test_reset_instance_clears(self):
        from app.core.captioning.caption_service import CaptionService
        CaptionService._instance = None
        CaptionService.get_instance()
        CaptionService.reset_instance()
        assert CaptionService._instance is None


class TestCaptionServiceGenerate:
    def test_unsupported_model_raises(self):
        from app.core.captioning.caption_service import CaptionService
        CaptionService._instance = None
        svc = CaptionService.get_instance()
        with pytest.raises(ValueError, match="not supported"):
            svc.generate_caption("/fake.png", "nonexistent_model", {})
        CaptionService._instance = None

    def test_generate_delegates_to_plugin(self, tmp_path):
        from app.core.captioning.caption_service import CaptionService
        CaptionService._instance = None
        svc = CaptionService.get_instance()

        # Create a real image to avoid load failures
        img_path = str(tmp_path / "test.png")
        Image.new("RGB", (32, 32)).save(img_path)

        # Mock a plugin
        mock_plugin = MagicMock()
        mock_plugin.generate.return_value = "a test caption"
        svc.plugins["florence-2"] = mock_plugin

        caption = svc.generate_caption(img_path, "florence-2", {})
        assert caption == "a test caption"
        mock_plugin.load.assert_called_once()
        CaptionService._instance = None

    def test_model_switching_unloads_previous(self, tmp_path):
        from app.core.captioning.caption_service import CaptionService
        CaptionService._instance = None
        svc = CaptionService.get_instance()

        img_path = str(tmp_path / "test.png")
        Image.new("RGB", (32, 32)).save(img_path)

        mock_a = MagicMock()
        mock_a.generate.return_value = "a"
        mock_b = MagicMock()
        mock_b.generate.return_value = "b"
        svc.plugins["florence-2"] = mock_a
        svc.plugins["joycaption"] = mock_b

        svc.generate_caption(img_path, "florence-2", {})
        svc.generate_caption(img_path, "joycaption", {})

        # A should have been unloaded when switching to B
        mock_a.unload.assert_called()
        CaptionService._instance = None


class TestCaptionServiceUnload:
    def test_unload_models_unloads_plugins_and_resets_active_key(self, monkeypatch):
        """Real (non-mocked) unload_models() call, routed through the shared
        gpu_unload helper (P2c / B-CLEAN-9): every plugin unloaded, active key
        reset, CUDA cache released."""
        import torch
        from app.core.captioning.caption_service import CaptionService

        calls: list[str] = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("synchronize"))
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))

        CaptionService._instance = None
        svc = CaptionService.get_instance()
        CaptionService._active_model_key = "florence-2"
        mock_a, mock_b = MagicMock(), MagicMock()
        svc.plugins = {"florence-2": mock_a, "joycaption": mock_b}

        result = CaptionService.unload_models()

        assert result is True
        mock_a.unload.assert_called_once()
        mock_b.unload.assert_called_once()
        assert "synchronize" in calls and "empty_cache" in calls
        assert CaptionService._active_model_key is None
        CaptionService._instance = None

    def test_unload_models_skip_if_batch_active_noops_when_batch_running(
        self, monkeypatch
    ):
        """W5.T10: skip_if_batch_active=True checks task_manager.list() AND
        performs the unload under the SAME lock — a running caption_batch
        task means the plugins are left untouched entirely."""
        from app.core.captioning.caption_service import CaptionService
        from app.core.tasks.task import TaskStatus
        from app.core.tasks.task_manager import task_manager as tm_instance

        fake_task = MagicMock(type="caption_batch", status=TaskStatus.RUNNING)
        monkeypatch.setattr(tm_instance, "list", lambda: [fake_task])

        CaptionService._instance = None
        svc = CaptionService.get_instance()
        CaptionService._active_model_key = "florence-2"
        mock_a = MagicMock()
        svc.plugins = {"florence-2": mock_a}

        result = CaptionService.unload_models(skip_if_batch_active=True)

        assert result is False
        mock_a.unload.assert_not_called()
        assert CaptionService._active_model_key == "florence-2"
        CaptionService._instance = None

    def test_unload_models_skip_if_batch_active_unloads_when_no_batch(
        self, monkeypatch
    ):
        """The mirror case: no active caption_batch -> the unload proceeds
        exactly as the unconditional call would."""
        from app.core.captioning.caption_service import CaptionService
        from app.core.tasks.task_manager import task_manager as tm_instance

        monkeypatch.setattr(tm_instance, "list", lambda: [])

        CaptionService._instance = None
        svc = CaptionService.get_instance()
        CaptionService._active_model_key = "florence-2"
        mock_a = MagicMock()
        svc.plugins = {"florence-2": mock_a}

        result = CaptionService.unload_models(skip_if_batch_active=True)

        assert result is True
        mock_a.unload.assert_called_once()
        assert CaptionService._active_model_key is None
        CaptionService._instance = None


class TestCaptionServiceLoadImage:
    def test_load_rgb(self, tmp_path):
        from app.core.captioning.caption_service import CaptionService
        CaptionService._instance = None
        svc = CaptionService.get_instance()
        img_path = str(tmp_path / "rgba.png")
        Image.new("RGBA", (32, 32)).save(img_path)
        img = svc._load_image(img_path)
        assert img.mode == "RGB"
        CaptionService._instance = None

    def test_load_nonexistent_raises(self):
        from app.core.captioning.caption_service import CaptionService
        CaptionService._instance = None
        svc = CaptionService.get_instance()
        with pytest.raises(ValueError, match="Could not open"):
            svc._load_image("/nonexistent.png")
        CaptionService._instance = None
