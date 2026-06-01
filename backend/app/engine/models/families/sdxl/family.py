"""SDXL model family registration."""

from app.engine.core.definitions import ModelFamily
from .trainer import SDXLTrainer


class SDXLFamily(ModelFamily):
    """SDXL implementation of the ModelFamily logic provider."""

    family_name = "sdxl"
    archetype = "latent_diffusion"
    capability_overrides = {"supports_train_te": True}

    def get_trainer_class(self):
        return SDXLTrainer
