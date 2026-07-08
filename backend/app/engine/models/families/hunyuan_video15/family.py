"""HunyuanVideo 1.5 model family registration.

HunyuanVideo 1.5 is a flow-match video MMDiT (54 dual-stream blocks) with an
external dual text encoder (Qwen2.5-VL 7B + ByT5 glyph channel) and a temporal
16x/4x VAE — the ``latent_diffusion`` archetype. Two 480p definitions ride the
family: T2V and I2V (the I2V checkpoint adds a Siglip image encoder, gated in
the loader by the definition's ``mode``).
"""

from app.engine.core.definitions import ModelFamily


class HunyuanVideo15Family(ModelFamily):
    """HunyuanVideo 1.5 (480p T2V / I2V) logic provider."""

    family_name = "hunyuan_video15"
    archetype = "latent_diffusion"

    # Merged into the capability descriptor by ``resolve_capabilities``.
    #  - is_video: batches are 5D [B, C, F, H, W] clips.
    #  - has_image_encoder: the Siglip vision encoder exists for I2V
    #    definitions (the loader only wires it when ``mode == i2v`` — the
    #    wan21 per-definition mode pattern).
    #
    # NOTE: ``supports_train_te`` is NOT overridden — the ``latent_diffusion``
    # archetype already defaults it to False (frozen dual TE), and a project
    # invariant reserves that override for SDXL.
    capability_overrides = {
        "is_video": True,
        "has_image_encoder": True,
        "native_fps": 24,
    }

    def get_trainer_class(self):
        """T2V and I2V share one trainer — the driver branches on ``mode``."""
        from .trainer import Hv15Trainer

        return Hv15Trainer
