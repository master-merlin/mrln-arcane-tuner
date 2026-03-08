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
