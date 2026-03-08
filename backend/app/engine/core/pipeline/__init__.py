"""
Generic Training Pipeline — shared base class for all model family trainers.

Implements the canonical training loop once with family-specific hooks for
model-dependent logic.  Families only override the abstract methods below;
everything else (optimizer, EMA, checkpoint, signals, logging, gradient
accumulation, noise offset, etc.) is handled here.

Pipeline ordering follows industry standard (Kohya / OneTrainer / ai-toolkit):
  1. Load weights (family)
  2. Freeze all
  3. Quantize frozen components
  4. Apply PEFT / LoRA (family provides targets)
  5. Gradient checkpointing
  6. Collect trainable params → optimizer → scheduler → scaler
  7. EMA
  8. Checkpoint manager → resume

This module re-exports ``GenericTrainingPipeline`` so existing imports
continue to work unchanged::

    from app.engine.core.pipeline import GenericTrainingPipeline  # ✓
"""

from app.engine.core.pipeline.pipeline_base import PipelineBaseMixin
from app.engine.core.pipeline.pipeline_loading import PipelineLoadingMixin
from app.engine.core.pipeline.pipeline_optimization import PipelineOptimizationMixin
from app.engine.core.pipeline.pipeline_caching import PipelineCachingMixin
from app.engine.core.pipeline.pipeline_data import PipelineDataMixin
from app.engine.core.pipeline.pipeline_train import PipelineTrainMixin


class GenericTrainingPipeline(
    PipelineTrainMixin,
    PipelineDataMixin,
    PipelineCachingMixin,
    PipelineOptimizationMixin,
    PipelineLoadingMixin,
    PipelineBaseMixin,
):
    """Shared base class implementing the canonical LoRA training pipeline.

    Subclasses must implement the family-specific hooks listed below.
    Everything else — optimizer, EMA, checkpointing, gradient accumulation,
    noise offset, signals, logging — is handled by this class.

    This class is assembled from focused mixin modules:

    - :class:`PipelineBaseMixin`         — abstract hooks, setup, component wiring
    - :class:`PipelineLoadingMixin`      — model loading, quantization, offloading
    - :class:`PipelineOptimizationMixin` — PEFT, optimizer, EMA, resume
    - :class:`PipelineCachingMixin`      — TE + latent pre-caching
    - :class:`PipelineDataMixin`         — dataset prep, batching
    - :class:`PipelineTrainMixin`        — main training loop
    """


__all__ = ["GenericTrainingPipeline"]
