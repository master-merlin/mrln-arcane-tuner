"""MiniMax-H3-Image model family registration.

A text-to-image family of its own: the MiniMax-H3 transformer without its audio
stream, distilled to 6 sampling steps with one output head per step. It is NOT
the ``minimax_h3`` video family (no audio, no frames, no dual shift); the code
both share lives in ``minimax_h3_shared``.
"""

from app.engine.core.definitions import ModelFamily


class MiniMaxH3ImageFamily(ModelFamily):
    """MiniMax-H3-Image implementation of the ModelFamily logic provider."""

    family_name = "minimax_h3_image"
    archetype = "latent_diffusion"

    capability_overrides = {
        "is_video": False,
        "has_audio": False,
        "dual_expert": False,
        # ``supports_train_te`` stays the archetype's False: the 66.7 GB Qwen3-VL
        # encoder is released before the transformer loads and never trains.
        "te_cache": True,
        "latent_cache": True,
        "supports_block_swap": True,
    }

    def get_trainer_class(self):
        # Imported at call time: a family module must never raise at import.
        from .trainer import MiniMaxH3ImageTrainer

        return MiniMaxH3ImageTrainer
