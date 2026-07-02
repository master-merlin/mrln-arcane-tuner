"""
Scoring Service for AI-powered image quality scoring.

Uses a plugin architecture where each model is implemented as a separate class,
following the same pattern as CaptionService and MaskingService.
"""
from __future__ import annotations

import os

from PIL import Image

from app.core.gpu_unload import unload_gpu_plugins
from app.core.scoring.models import ScoringModel, HPSv2Model


class ScoringService:
    """
    Singleton service for AI-powered image quality scoring.

    Manages model loading, caching, and score generation using a plugin
    architecture.
    """

    _instance: ScoringService | None = None
    _active_model_id: str | None = None

    def __init__(self) -> None:
        self.plugins: dict[str, ScoringModel] = {
            "hpsv2": HPSv2Model(self),
        }

    @classmethod
    def get_instance(cls) -> ScoringService:
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = ScoringService()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls.unload_models()
        cls._instance = None

    @classmethod
    def unload_models(cls) -> None:
        """Unload all models from memory and clear CUDA cache."""
        unload_gpu_plugins(
            cls,
            plugins=cls._instance.plugins if cls._instance else {},
            active_attr="_active_model_id",
            service_label="scoring",
        )

    def score_image(self, image_path: str, model_id: str, params: dict) -> float:
        """
        Score a single image using the specified model.

        Returns a float score (higher = better quality).
        """
        if model_id not in self.plugins:
            raise ValueError(f"Unknown scoring model: {model_id}")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        plugin = self.plugins[model_id]

        # Handle model switching
        if self.__class__._active_model_id and self.__class__._active_model_id != model_id:
            self.unload_models()

        # Load if not already loaded
        if self.__class__._active_model_id != model_id:
            plugin.load(**params)
            self.__class__._active_model_id = model_id

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Could not open image {image_path}: {e}")

        return plugin.score(image, params)
