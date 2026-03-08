"""Model subsystem — base classes, plugin registry, and training configuration."""

from app.engine.models.base import TrainingPlugin
from app.engine.models.training_plugin import StandardPlugin

__all__ = [
    "TrainingPlugin",
    "StandardPlugin"
]
