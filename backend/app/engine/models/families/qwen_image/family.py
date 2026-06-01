"""Qwen-Image model family registration."""

from app.engine.core.definitions import ModelFamily
from .trainer import QwenImageTrainer


class QwenImageFamily(ModelFamily):
    """Qwen-Image (2512) implementation of the ModelFamily logic provider."""

    family_name = "qwen_image"
    archetype = "latent_diffusion"

    def get_trainer_class(self):
        return QwenImageTrainer
