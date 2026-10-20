"""MOSS-Image model family registration (LANE-165)."""

from app.engine.core.definitions import ModelFamily


class MossImageFamily(ModelFamily):
    """MOSS-Image (Base and SFT) implementation of the ModelFamily logic provider."""

    family_name = "moss_image"
    archetype = "latent_diffusion"

    def get_trainer_class(self):
        # Imported at call time: the trainer pulls in torch-heavy modules, and a family module must
        # never raise at import (ARCHITECTURE D1).
        from .trainer import MossImageTrainer

        return MossImageTrainer
