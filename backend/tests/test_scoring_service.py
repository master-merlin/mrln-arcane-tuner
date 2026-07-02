"""Tests for ScoringService — singleton + unload contract (mirrors
test_masking_service.py / test_caption_service.py's unload coverage).
"""
from unittest.mock import MagicMock

import pytest


class TestScoringServiceSingleton:
    def test_get_instance_returns_same(self):
        from app.core.scoring.scoring_service import ScoringService
        ScoringService._instance = None
        a = ScoringService.get_instance()
        b = ScoringService.get_instance()
        assert a is b
        ScoringService._instance = None

    def test_reset_instance_clears(self):
        from app.core.scoring.scoring_service import ScoringService
        ScoringService._instance = None
        ScoringService.get_instance()
        ScoringService.reset_instance()
        assert ScoringService._instance is None


class TestScoringServiceGenerate:
    def test_unknown_model_raises(self):
        from app.core.scoring.scoring_service import ScoringService
        ScoringService._instance = None
        svc = ScoringService.get_instance()
        with pytest.raises(ValueError, match="Unknown scoring model"):
            svc.score_image("/fake.png", "nonexistent", {})
        ScoringService._instance = None

    def test_missing_image_raises(self):
        from app.core.scoring.scoring_service import ScoringService
        ScoringService._instance = None
        svc = ScoringService.get_instance()
        with pytest.raises(FileNotFoundError):
            svc.score_image("/nonexistent.png", "hpsv2", {})
        ScoringService._instance = None


class TestScoringServiceUnload:
    def test_unload_models_unloads_plugins_and_resets_active_key(self, monkeypatch):
        """Real (non-mocked) unload_models() call, routed through the shared
        gpu_unload helper (P2c / B-CLEAN-9): every plugin unloaded, active key
        reset, CUDA cache released."""
        import torch
        from app.core.scoring.scoring_service import ScoringService

        calls: list[str] = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("synchronize"))
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))

        ScoringService._instance = None
        svc = ScoringService.get_instance()
        ScoringService._active_model_id = "hpsv2"
        mock_plugin = MagicMock()
        svc.plugins = {"hpsv2": mock_plugin}

        ScoringService.unload_models()

        mock_plugin.unload.assert_called_once()
        assert "synchronize" in calls and "empty_cache" in calls
        assert ScoringService._active_model_id is None
        ScoringService._instance = None
