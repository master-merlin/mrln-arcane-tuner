"""HiDream-O1-Image model family registration."""

from app.engine.core.definitions import ModelFamily


class HiDreamO1Family(ModelFamily):
    """Pixel-space Unified Transformer family — text-to-image LoRA only (v1)."""

    family_name = "hidream_o1"
    archetype = "unified_transformer"
    # trainer.py forward_pass / compute_loss draw their own noise in patch space and never
    # receive the shared loop's offset noise, so noise_offset is hidden (interview Q1.1, LANE-168).
    capability_overrides = {"supports_noise_offset": False}

    def get_trainer_class(self):
        # Import inside the method to avoid loading heavy training deps
        # (peft, transformers) just to register the family.
        from .trainer import HiDreamO1Trainer

        return HiDreamO1Trainer
