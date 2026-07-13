"""Lumina-Image-2.0 model family registration."""

from app.engine.core.definitions import ModelFamily
from .trainer import Lumina2Trainer


class Lumina2Family(ModelFamily):
    """Lumina-Image-2.0 implementation of the ModelFamily logic provider."""

    family_name = "lumina2"
    archetype = "latent_diffusion"

    def get_trainer_class(self):
        return Lumina2Trainer
