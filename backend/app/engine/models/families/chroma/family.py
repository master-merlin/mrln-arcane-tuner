"""Chroma model family registration."""

from app.engine.core.definitions import ModelFamily
from .trainer import ChromaTrainer


class ChromaFamily(ModelFamily):
    """Chroma (lodestones' FLUX.1-schnell-derived, T5-only, real-CFG DiT)
    implementation of the ModelFamily logic provider.

    Two definitions: ``chroma1-base`` (primary training checkpoint) and
    ``chroma1-hd`` (higher-fidelity variant).
    """

    family_name = "chroma"
    archetype = "latent_diffusion"

    def get_trainer_class(self):
        return ChromaTrainer
