"""FLUX.1 model family registration."""

from app.engine.core.definitions import ModelFamily
from .trainer import Flux1Trainer


class Flux1Family(ModelFamily):
    """FLUX.1 (Dev / Schnell / Kontext) implementation of the ModelFamily logic provider."""

    family_name = "flux1"
    archetype = "latent_diffusion"

    def get_trainer_class(self):
        # Kontext (control_inputs > 0) conditions on a clean control image —
        # dispatch to the image-edit trainer. Standard FLUX.1 (Dev/Schnell)
        # keeps the proven base trainer untouched.
        if int(getattr(self.definition, "control_inputs", 0) or 0) > 0:
            from .trainer_kontext import Flux1KontextTrainer
            return Flux1KontextTrainer
        return Flux1Trainer
