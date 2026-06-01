"""Z-Image model family registration."""

from app.engine.core.definitions import ModelFamily
from .trainer import ZImageTrainer


class ZImageFamily(ModelFamily):
    """Z-Image Base implementation of the ModelFamily logic provider."""

    family_name = "zimage"
    archetype = "latent_diffusion"

    def get_trainer_class(self):
        return ZImageTrainer
