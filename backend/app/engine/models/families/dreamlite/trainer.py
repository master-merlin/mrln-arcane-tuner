"""DreamLiteTrainer — family-specific trainer for DreamLite (Base / Mobile).

All shared logic (optimizer, EMA, gradient accumulation, noise offset,
checkpointing, signals, logging) lives in ``GenericTrainingPipeline``.
Loader / driver / sampler / saver are lazy-imported inside ``_setup_family``
so that registry discovery (which merely imports this module) never trips
on a missing later-task module (krea2 Phase-1 convention).
"""

from __future__ import annotations

import structlog

from app.engine.core.pipeline import GenericTrainingPipeline

logger = structlog.get_logger(__name__)


class DreamLiteTrainer(GenericTrainingPipeline):
    """DreamLite LoRA trainer.

    ~0.39B DreamLite **U-Net** (GQA/MQA + qk_norm + depthwise-separable
    convs — NOT a DiT) with a Qwen3-VL text encoder (hidden 2048),
    AutoencoderTiny VAE, and flow-matching noise schedule. Base supports
    CFG with negative prompts; Mobile is CFG-distilled (4 steps).
    """

    def _setup_family(self) -> None:
        """Initialize DreamLite-specific loader and driver.

        All family-specific classes are lazy-imported here so that registry
        discovery (which just imports this module) never trips on missing
        later-task modules.
        """
        from .loader import DreamLiteLoader  # noqa: PLC0415
        from .driver import DreamLiteDriver  # noqa: PLC0415

        self.loader = DreamLiteLoader(self.device)
        self.driver = DreamLiteDriver(self.definition, self.device)
