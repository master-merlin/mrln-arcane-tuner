"""FLUX.3 model family registration (LANE-144)."""

from app.engine.core.definitions import ModelFamily


class Flux3Family(ModelFamily):
    """FLUX.3 Image Klein (Base) implementation of the ModelFamily logic provider."""

    family_name = "flux3"
    archetype = "latent_diffusion"

    def get_trainer_class(self):
        # Imported at call time: the trainer pulls in torch-heavy modules, and a family module must
        # never raise at import (ARCHITECTURE D1).
        from .trainer import Flux3Trainer

        return Flux3Trainer
