"""LTX 2.3 model family registration."""

from app.engine.core.definitions import ModelFamily

from .trainer import Ltx2Trainer


class Ltx2Family(ModelFamily):
    """LTX 2.3 (Lightricks) joint audio + video implementation.

    A single trainable model covers both T2V and I2V (image-to-video is
    conditioning, not a separate checkpoint).  The capability overrides
    surface this to the Training UI:

    - ``is_video``: clips, not stills — drives video bucketing / fps fields.
    - ``has_audio``: an optional joint audio stream the user can toggle on.

    The Gemma3 text encoder is frozen — ``supports_train_te`` is left at the
    ``latent_diffusion`` archetype default (``False``) rather than overridden,
    so the cross-family "only SDXL overrides supports_train_te" guard holds.
    """

    family_name = "ltx2"
    archetype = "latent_diffusion"

    # Merged into the capability descriptor by ``core/archetypes.py``
    # (free-form dict; archetypes.py is never touched).  ``is_video`` and
    # ``has_audio`` are new flags the video program consumes.  TE training is
    # already False by archetype default, so it is NOT repeated here.
    capability_overrides = {
        "is_video": True,
        "has_audio": True,
    }

    def get_trainer_class(self):
        return Ltx2Trainer
