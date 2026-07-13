"""ACE-Step 1.5 model family registration.

ACE-Step 1.5 is a flow-match music-generation DiT (``AceStepTransformer1DModel``,
diffusers 0.39 NATIVE — no vendoring needed, see the driver module docstring)
with a Qwen3-Embedding-0.6B text encoder, a dedicated condition encoder
(``AceStepConditionEncoder`` — folds text + lyric + timbre embeddings into one
cross-attention sequence), and an ``AutoencoderOobleck`` audio VAE (48kHz
stereo, 25Hz latent rate).

This is the FIRST audio-generation family — no spatial dimension, no video
frames. Capability flags (``is_audio_family``) hide the VIDEO/image-editing
field groups and the resolution/bucketing fields; the new AUDIO group
(``duration_s``/``genre_ratio``) surfaces instead (see ``core/archetypes.py``).
"""

from app.engine.core.definitions import ModelFamily


class AceStep15Family(ModelFamily):
    """ACE-Step 1.5 (text2music) logic provider."""

    family_name = "ace_step15"
    archetype = "latent_diffusion"

    # Merged into the capability descriptor by ``resolve_capabilities``.
    #  - is_audio_family: this family's PRIMARY modality is audio — hides
    #    VIDEO/resolution/augmentation/masking fields, surfaces AUDIO fields.
    #  - supports_spatial_resolution: False — audio has no width/height; the
    #    data pipeline stamps a dummy constant spatial dim per item instead
    #    of running BucketManager (see pipeline_data.py's audio branch).
    #
    # NOTE: ``supports_train_te`` is NOT overridden — the ``latent_diffusion``
    # archetype already defaults it to False (frozen Qwen3 TE + condition
    # encoder), and a project invariant reserves that override for SDXL.
    capability_overrides = {
        "is_audio_family": True,
        "supports_spatial_resolution": False,
    }

    def get_trainer_class(self):
        from .trainer import AceStep15Trainer

        return AceStep15Trainer
