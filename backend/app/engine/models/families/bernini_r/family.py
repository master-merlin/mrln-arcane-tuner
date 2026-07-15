"""Bernini-R model family registration.

Bernini-R is a renderer-only video-EDIT DiT built from stock Wan components
(UMT5-XXL text encoder + stock-key ``WanTransformer3DModel`` + Wan2.1 VAE), so
it maps to the ``latent_diffusion`` archetype and is flagged a video family. Its
conditioning is data-side (token-concat + ``source_id`` rope — see
``vendor/transformer_forward.py``); there is no CLIP image encoder (unlike Wan
I2V), so ``has_image_encoder`` stays False.
"""

from app.engine.core.definitions import ModelFamily


class BerniniRFamily(ModelFamily):
    """Bernini-R (renderer-only video edit) logic provider."""

    family_name = "bernini_r"
    archetype = "latent_diffusion"

    # Merged into the capability descriptor by ``resolve_capabilities``.
    #  - is_video: batches are 5D [B, C, F, H, W] video clips.
    # UMT5 text encoder is frozen (latent_diffusion defaults supports_train_te
    # False), so no override needed there.
    capability_overrides = {
        "is_video": True,
        "native_fps": 16,
    }

    def get_trainer_class(self):
        """Bernini-R trainer (Task BR3). Imported lazily so family discovery /
        the BR2 driver + forward tests never require BR3 code.
        """
        from .trainer import BerniniRTrainer

        return BerniniRTrainer
