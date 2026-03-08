"""
Tests for MaskingService — covers singleton, mask generation, combine, preview, model switching.
"""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock
from PIL import Image


class TestMaskingServiceSingleton:
    def test_get_instance_returns_same(self):
        from app.core.masking.masking_service import MaskingService
        MaskingService._instance = None
        a = MaskingService.get_instance()
        b = MaskingService.get_instance()
        assert a is b
        MaskingService._instance = None

    def test_unload_clears_active_model(self):
        from app.core.masking.masking_service import MaskingService
        MaskingService._instance = None
        svc = MaskingService.get_instance()
        MaskingService._active_model_id = "rembg"
        svc.unload_models()
        assert MaskingService._active_model_id is None
        MaskingService._instance = None


class TestMaskingServiceGenerate:
    def test_unknown_model_raises(self, tmp_path):
        from app.core.masking.masking_service import MaskingService
        MaskingService._instance = None
        svc = MaskingService.get_instance()
        img_path = str(tmp_path / "test.png")
        Image.new("RGB", (32, 32)).save(img_path)
        with pytest.raises(ValueError, match="Unknown masking model"):
            svc.generate_mask(img_path, "nonexistent", {})
        MaskingService._instance = None

    def test_missing_image_raises(self):
        from app.core.masking.masking_service import MaskingService
        MaskingService._instance = None
        svc = MaskingService.get_instance()
        with pytest.raises(FileNotFoundError):
            svc.generate_mask("/nonexistent.png", "rembg", {})
        MaskingService._instance = None

    def test_generate_returns_l_mode(self, tmp_path):
        from app.core.masking.masking_service import MaskingService
        MaskingService._instance = None
        svc = MaskingService.get_instance()

        img_path = str(tmp_path / "test.png")
        Image.new("RGB", (32, 32)).save(img_path)

        mock_plugin = MagicMock()
        mock_plugin.generate.return_value = Image.new("RGB", (32, 32), (255, 255, 255))
        svc.plugins["rembg"] = mock_plugin

        mask = svc.generate_mask(img_path, "rembg", {})
        assert mask.mode == "L"
        MaskingService._instance = None


class TestMaskingServiceCombine:
    def test_combine_png(self, tmp_path):
        from app.core.masking.masking_service import MaskingService
        MaskingService._instance = None
        svc = MaskingService.get_instance()

        img_path = str(tmp_path / "test.png")
        mask_path = str(tmp_path / "mask.png")
        output_path = str(tmp_path / "output.png")

        Image.new("RGB", (32, 32), (255, 0, 0)).save(img_path)
        Image.new("L", (32, 32), 255).save(mask_path)

        svc.combine_mask(img_path, mask_path, output_path, opacity=0.0)
        assert os.path.exists(output_path)

        result = Image.open(output_path)
        assert result.mode == "RGBA"
        MaskingService._instance = None

    def test_combine_jpg(self, tmp_path):
        from app.core.masking.masking_service import MaskingService
        MaskingService._instance = None
        svc = MaskingService.get_instance()

        img_path = str(tmp_path / "test.png")
        mask_path = str(tmp_path / "mask.png")
        output_path = str(tmp_path / "output.jpg")

        Image.new("RGB", (32, 32), (255, 0, 0)).save(img_path)
        Image.new("L", (32, 32), 128).save(mask_path)

        svc.combine_mask(img_path, mask_path, output_path, opacity=0.5)
        assert os.path.exists(output_path)

        result = Image.open(output_path)
        assert result.mode == "RGB"  # JPEG has no alpha
        MaskingService._instance = None


class TestMaskingServicePreview:
    def test_generate_preview_rgba(self, tmp_path):
        from app.core.masking.masking_service import MaskingService
        MaskingService._instance = None
        svc = MaskingService.get_instance()

        img_path = str(tmp_path / "test.png")
        mask_path = str(tmp_path / "mask.png")

        Image.new("RGB", (32, 32), (0, 255, 0)).save(img_path)
        Image.new("L", (32, 32), 255).save(mask_path)  # Full foreground

        preview = svc.generate_preview(img_path, mask_path, opacity=0.0)
        assert preview.mode == "RGBA"
        assert preview.size == (32, 32)

        # Full mask → alpha channel should be 255 everywhere (fully opaque foreground)
        alpha = np.array(preview)[:, :, 3]
        assert alpha.min() == 255
        MaskingService._instance = None

    def test_preview_with_partial_opacity(self, tmp_path):
        from app.core.masking.masking_service import MaskingService
        MaskingService._instance = None
        svc = MaskingService.get_instance()

        img_path = str(tmp_path / "test.png")
        mask_path = str(tmp_path / "mask.png")

        Image.new("RGB", (32, 32)).save(img_path)
        Image.new("L", (32, 32), 0).save(mask_path)  # All background

        preview = svc.generate_preview(img_path, mask_path, opacity=0.5)
        alpha = np.array(preview)[:, :, 3]
        # Background with 0.5 opacity → alpha = (0 + 1.0 * 0.5) * 255 = 127-128
        assert 126 <= alpha[0, 0] <= 128
        MaskingService._instance = None
