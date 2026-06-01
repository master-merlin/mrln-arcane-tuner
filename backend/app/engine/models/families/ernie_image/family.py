"""ERNIE-Image model family registration."""

# Import for side-effect: registers ``ministral3`` config/model aliases so
# transformers 4.x can parse the text_encoder/config.json shipped by Baidu
# (which was exported from a transformers 5.2.0 dev build).
from . import _compat  # noqa: F401

from app.engine.core.definitions import ModelFamily
from .trainer import ErnieImageTrainer


class ErnieImageFamily(ModelFamily):
    """Baidu ERNIE-Image implementation of the ModelFamily logic provider."""

    family_name = "ernie_image"
    archetype = "latent_diffusion"

    def get_trainer_class(self):
        return ErnieImageTrainer
