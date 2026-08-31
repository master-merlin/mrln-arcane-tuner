"""
Scoring Service for AI-powered image quality scoring.

Uses a plugin architecture where each model is implemented as a separate class,
following the same pattern as CaptionService and MaskingService.
"""
from __future__ import annotations

import os
import threading

from PIL import Image

from app.core.gpu_unload import gpu_batch_active, unload_gpu_plugins
from app.core.scoring.models import ScoringModel, HPSv2Model

# Task types that own the scoring GPU plugin for their duration, so a global
# unload must not rip a model out from under them.
#
# `rescan_batch` is the ONLY one: scoring has no batch task of its own — the
# score-batch task was removed and scoring now runs only inside a rescan, whose
# worker owns the model and frees it in its own `finally`
# (rescan_batch.py:47-49, `_unload() → ScoringService.unload_models()`).
# `grep -rl ScoringService backend/app` names no other task worker.
_SCORING_GUARD_TASK_TYPES = ("rescan_batch",)


class ScoringService:
    """
    Singleton service for AI-powered image quality scoring.

    Manages model loading, caching, and score generation using a plugin
    architecture.
    """

    _instance: ScoringService | None = None
    _active_model_id: str | None = None
    # Guards unload_models(skip_if_batch_active=True)'s check-then-act, so a
    # rescan cannot start between "no rescan is running" and the unload.
    # Same shape and same reason as CaptionService._unload_lock.
    _unload_lock = threading.Lock()

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
    def unload_models(cls, *, skip_if_batch_active: bool = False) -> bool:
        """Unload all models from memory and clear CUDA cache.

        ``skip_if_batch_active=True`` (the ``/system/gpu/unload`` route's mode)
        — the check for a pending/running rescan PLUS the unload itself run
        under ``_unload_lock``, atomically, mirroring
        ``CaptionService.unload_models``. Internal callers (``score_image``
        switching models, ``rescan_batch``'s own ``finally`` unload,
        ``reset_instance``) pass the default ``False`` and unload
        unconditionally.

        Returns ``True`` if the unload actually ran, ``False`` if skipped.
        """
        with cls._unload_lock:
            if skip_if_batch_active and gpu_batch_active(_SCORING_GUARD_TASK_TYPES):
                return False

            unload_gpu_plugins(
                cls,
                plugins=cls._instance.plugins if cls._instance else {},
                active_attr="_active_model_id",
                service_label="scoring",
            )
            return True

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
