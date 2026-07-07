"""DreamLite model family registration."""

from app.engine.core.definitions import ModelFamily
from .trainer import DreamLiteTrainer


class DreamLiteFamily(ModelFamily):
    """DreamLite (Base / Mobile) implementation of the ModelFamily logic provider."""

    family_name = "dreamlite"
    archetype = "latent_diffusion"

    def get_trainer_class(self):
        return DreamLiteTrainer
