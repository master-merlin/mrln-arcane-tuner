"""Microsoft Lens model family registration."""

from app.engine.core.definitions import ModelFamily
from .trainer import MicrosoftLensTrainer


class MicrosoftLensFamily(ModelFamily):
    """Microsoft Lens implementation of the ModelFamily logic provider."""

    family_name = "microsoft_lens"
    archetype = "latent_diffusion"

    def get_trainer_class(self):
        return MicrosoftLensTrainer
