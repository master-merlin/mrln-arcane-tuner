"""
Scoring Service for AI-powered image quality scoring.

Uses a plugin architecture where each model is implemented as a separate class,
following the same pattern as CaptionService and MaskingService.
"""
from __future__ import annotations

import gc
import os
from typing import Callable

import structlog
import torch
from PIL import Image

from app.core.scoring.models import ScoringModel, HPSv2Model

logger = structlog.get_logger(__name__)


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
        if cls._active_model_id:
            logger.info("unloading_scoring_models", active_model=cls._active_model_id)

        if cls._instance:
            for plugin in cls._instance.plugins.values():
                plugin.unload()

        cls._active_model_id = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        logger.info("all_scoring_models_unloaded")

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

    def score_batch(
        self,
        image_paths: list[str],
        model_id: str,
        params: dict,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, float]:
        """
        Score multiple images using the specified model.

        Args:
            image_paths: List of absolute image paths.
            model_id: Scoring model identifier (e.g. "hpsv2").
            params: Model-specific parameters.
            progress_callback: Optional (current, total, filename) callback.

        Returns:
            Dict mapping filename → score.
        """
        if model_id not in self.plugins:
            raise ValueError(f"Unknown scoring model: {model_id}")

        plugin = self.plugins[model_id]

        # Handle model switching
        if self.__class__._active_model_id and self.__class__._active_model_id != model_id:
            self.unload_models()

        # Load if not already loaded
        if self.__class__._active_model_id != model_id:
            plugin.load(**params)
            self.__class__._active_model_id = model_id

        # Filter to existing files only
        valid_paths = [p for p in image_paths if os.path.exists(p)]

        results: dict[str, float] = {}
        total = len(valid_paths)

        for i, path in enumerate(valid_paths):
            filename = os.path.basename(path)
            try:
                image = Image.open(path).convert("RGB")
                score = plugin.score(image, params)
                results[filename] = score
            except Exception as e:
                logger.warning("scoring_image_failed", file=filename, error=str(e))
                results[filename] = 0.0

            if progress_callback:
                progress_callback(i + 1, total, filename)

        return results
