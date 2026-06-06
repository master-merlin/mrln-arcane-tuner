"""
HPSv2 (Human Preference Score v2) model plugin for image quality scoring.

Uses a fine-tuned CLIP ViT-H/14 backbone trained on 798K human preference
pairs. Scores represent relative quality — higher is better.

Reference: https://github.com/tgxs002/HPSv2
"""
from __future__ import annotations

import gc

import structlog
import torch
from PIL import Image

from app.core.scoring.models.base import ScoringModel

logger = structlog.get_logger(__name__)


class HPSv2Model(ScoringModel):
    """HPSv2 scoring model plugin."""

    def __init__(self, service: "ScoringService") -> None:  # noqa: F821
        super().__init__(service)
        self._loaded = False

    def load(self, **kwargs) -> None:
        """Pre-warm HPSv2 by running a single dummy score.

        The hpsv2 library lazily loads its CLIP model on first call.
        We trigger that here so subsequent score() calls are fast.
        """
        if self._loaded:
            return

        import hpsv2  # noqa: F811 — lazy import to avoid startup cost

        logger.info("hpsv2_loading_model")

        # Create a tiny dummy image to trigger model loading
        dummy = Image.new("RGB", (64, 64), color=(128, 128, 128))
        version = kwargs.get("hps_version", "v2.1")
        try:
            hpsv2.score(dummy, "", hps_version=version)
        except Exception:
            pass  # Some versions may error on tiny images; model is still loaded

        self._loaded = True
        logger.info("hpsv2_model_loaded")

    def score(self, image: Image.Image, params: dict) -> float:
        """Score a single image.

        Args:
            image: PIL Image in RGB mode.
            params: dict with optional keys:
                - prompt (str): caption text for image-text alignment scoring
                - hps_version (str): "v2.0" or "v2.1" (default "v2.1")

        Returns:
            Raw HPSv2 score (float). Higher = better quality/preference.
        """
        import hpsv2

        prompt = params.get("prompt", "")
        version = params.get("hps_version", "v2.1")

        results = hpsv2.score(image, prompt, hps_version=version)
        return float(results[0]) if results else 0.0

    def unload(self) -> None:
        """Unload HPSv2 model and free GPU memory."""
        if not self._loaded:
            return

        logger.info("hpsv2_unloading_model")

        # HPSv2 caches its model in the module scope — clear torch cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        self._loaded = False
        logger.info("hpsv2_model_unloaded")
