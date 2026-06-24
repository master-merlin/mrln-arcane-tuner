"""Krea2 model family registration."""

from app.engine.core.definitions import ModelFamily
from .trainer import Krea2Trainer


class Krea2Family(ModelFamily):
    """Krea-2 (Raw / Turbo) implementation of the ModelFamily logic provider."""

    family_name = "krea2"
    archetype = "latent_diffusion"

    def get_trainer_class(self):
        return Krea2Trainer
