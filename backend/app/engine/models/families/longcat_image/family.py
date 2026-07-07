"""LongCat-Image model family registration."""

from app.engine.core.definitions import ModelFamily


class LongCatImageFamily(ModelFamily):
    """LongCat-Image implementation of the ModelFamily logic provider.

    ~11.9B Flux-like DiT (19 double + 38 single blocks) by Meituan with a
    Qwen2.5-VL-7B text encoder and a standard 16-channel AutoencoderKL.
    """

    family_name = "longcat_image"
    archetype = "latent_diffusion"

    def get_trainer_class(self):
        # Lazy import: keeps family discovery lightweight.
        from .trainer import LongCatImageTrainer

        return LongCatImageTrainer
