"""WAN 2.1 model family registration.

WAN 2.1 is a flow-match video DiT with an external UMT5-XXL text encoder and a
temporal Wan-VAE — closest to the ``latent_diffusion`` archetype. The
``capability_overrides`` flag it as a video family (so the UI/data path treats
batches as 5D clips) and, for I2V definitions, declare an image encoder.
"""

from app.engine.core.definitions import ModelFamily


class Wan21Family(ModelFamily):
    """WAN 2.1 (T2V 1.3B/14B, I2V 14B) logic provider."""

    family_name = "wan21"
    archetype = "latent_diffusion"

    # Merged into the capability descriptor by ``resolve_capabilities``.
    #  - is_video: batches are 5D [B, C, F, H, W] video clips.
    #  - has_image_encoder: a CLIP-vision encoder is present for I2V
    #    definitions (the loader only wires it when ``mode == i2v``).
    #
    # NOTE: ``supports_train_te`` is intentionally NOT overridden here — the
    # ``latent_diffusion`` archetype already defaults it to ``False``, so WAN
    # (frozen UMT5 text encoder) inherits the correct value without a redundant
    # override (which a project invariant test reserves for SDXL).
    capability_overrides = {
        "is_video": True,
        "has_image_encoder": True,
        "native_fps": 16,
    }

    def get_trainer_class(self):
        """Dispatch T2V vs I2V by the definition's ``mode`` architecture param.

        I2V definitions condition on the first frame (image encoder + 36-channel
        transformer input); T2V is pure text-to-video. Both share one trainer
        class — the driver branches on ``mode`` internally — so dispatch returns
        the same trainer but is structured to allow a future I2V split.
        """
        from .trainer import Wan21Trainer

        return Wan21Trainer
