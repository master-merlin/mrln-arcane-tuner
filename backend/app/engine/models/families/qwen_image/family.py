"""Qwen-Image model family registration."""

from app.engine.core.definitions import ModelFamily
from .trainer import QwenImageTrainer


class QwenImageFamily(ModelFamily):
    """Qwen-Image (2512) implementation of the ModelFamily logic provider."""

    family_name = "qwen_image"
    archetype = "latent_diffusion"

    def get_trainer_class(self):
        # Image-edit definitions (control_inputs > 0) train on paired control
        # images — dispatch to the Edit subclass (mirrors flux1 Kontext).
        if int(getattr(self.definition, "control_inputs", 0) or 0) > 0:
            from .trainer_edit import QwenImageEditTrainer
            return QwenImageEditTrainer
        return QwenImageTrainer
