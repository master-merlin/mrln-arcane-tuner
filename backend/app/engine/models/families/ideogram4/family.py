"""Ideogram 4 model family registration."""

from app.engine.core.definitions import ModelFamily
from .trainer import IdeogramV4Trainer


class IdeogramV4Family(ModelFamily):
    """Ideogram 4 implementation of the ModelFamily logic provider."""

    family_name = "ideogram4"
    archetype = "latent_diffusion"

    def get_trainer_class(self):
        return IdeogramV4Trainer
