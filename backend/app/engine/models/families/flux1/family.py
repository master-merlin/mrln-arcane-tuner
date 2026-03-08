"""FLUX.1 model family registration."""

from app.engine.core.definitions import ModelFamily
from .trainer import Flux1Trainer


class Flux1Family(ModelFamily):
    """FLUX.1 (Dev / Schnell) implementation of the ModelFamily logic provider."""

    family_name = "flux1"

    def get_trainer_class(self):
        return Flux1Trainer
