"""FLUX.2 model family registration."""

from app.engine.core.definitions import ModelFamily
from .trainer import Flux2Trainer


class Flux2Family(ModelFamily):
    """FLUX.2 (Klein / Dev) implementation of the ModelFamily logic provider."""

    family_name = "flux2"
    archetype = "latent_diffusion"

    def get_trainer_class(self):
        return Flux2Trainer
