"""Nucleus-Image model family registration."""

from app.engine.core.definitions import ModelFamily
from .trainer import NucleusImageTrainer


class NucleusImageFamily(ModelFamily):
    """Nucleus-Image implementation of the ModelFamily logic provider."""

    family_name = "nucleus_image"
    archetype = "latent_diffusion"

    def get_trainer_class(self):
        return NucleusImageTrainer
