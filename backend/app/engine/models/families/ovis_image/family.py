"""Ovis-Image model family registration."""

from app.engine.core.definitions import ModelFamily
from .trainer import OvisImageTrainer


class OvisImageFamily(ModelFamily):
    """Ovis-Image implementation of the ModelFamily logic provider."""

    family_name = "ovis_image"
    archetype = "latent_diffusion"

    def get_trainer_class(self):
        return OvisImageTrainer
