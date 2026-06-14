"""WAN 2.2 model family registration (dual-expert MoE).

WAN 2.2 A14B is a flow-match video DiT Mixture-of-Experts: a high-noise expert
(``transformer``) + a low-noise expert (``transformer_2``), routed by the sampled
timestep around a boundary (T2V≈0.875, I2V≈0.9). It reuses the ``latent_diffusion``
archetype (UMT5-XXL external TE + temporal Wan-VAE). The ``capability_overrides``
flag it as a video family and add a ``dual_expert`` flag the Training UI consumes.

Unlike WAN 2.1 I2V, WAN 2.2 I2V has NO CLIP image encoder
(``has_image_encoder`` is False) — I2V conditions on the first-frame latent only.
"""

from app.engine.core.definitions import ModelFamily


class Wan22Family(ModelFamily):
    """WAN 2.2 (T2V-A14B / I2V-A14B) dual-expert logic provider."""

    family_name = "wan22"
    archetype = "latent_diffusion"

    # Merged into the capability descriptor by ``core/archetypes.py``.
    #  - is_video:    batches are 5D [B, C, F, H, W] video clips.
    #  - dual_expert: MoE high/low transformers trained in one run (two LoRAs).
    #  - has_image_encoder: False — WAN 2.2 I2V is first-frame-latent only
    #    (no CLIP-vision conditioning), unlike WAN 2.1.
    #
    # ``supports_train_te`` is intentionally NOT overridden (frozen UMT5) — it
    # is already False via the latent_diffusion archetype default.
    capability_overrides = {
        "is_video": True,
        "dual_expert": True,
        "has_image_encoder": False,
        "native_fps": 16,
    }

    def get_trainer_class(self):
        """Both T2V-A14B and I2V-A14B share the dual-expert trainer.

        The driver branches on ``mode`` (t2v/i2v) for the conditioning path;
        the dual-expert routing is identical for both.
        """
        from .trainer import Wan22Trainer

        return Wan22Trainer
