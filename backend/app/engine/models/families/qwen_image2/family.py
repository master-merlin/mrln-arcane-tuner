"""Qwen-Image 2 model family registration (Qwen-Image-2.1 onwards).

A family of its own, not a ``qwen_image`` definition: the transformer
(``QwenImage21Transformer2DModel``, single-stream, patch 1), the VAE
(``AutoencoderKLQwenImage21``, z_dim 64, 4 channels) and the text encoder
(Qwen3-VL) all differ from Qwen-Image 2512 (LANE-132 spec; evidence
``.agent/workdir/lane-104/evidence.md`` §1).
"""

from app.engine.core.definitions import ModelFamily


class QwenImage2Family(ModelFamily):
    """Qwen-Image 2.x implementation of the ModelFamily logic provider."""

    family_name = "qwen_image2"
    archetype = "latent_diffusion"

    def get_trainer_class(self):
        # Imported on use, not at module level: registry discovery imports
        # this module at startup and must never raise (ARCHITECTURE D1).
        from .trainer import QwenImage2Trainer

        return QwenImage2Trainer
