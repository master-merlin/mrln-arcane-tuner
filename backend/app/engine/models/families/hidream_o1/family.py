"""HiDream-O1-Image model family registration."""

from app.engine.core.definitions import ModelFamily


class HiDreamO1Family(ModelFamily):
    """Pixel-space Unified Transformer family — text-to-image LoRA only (v1)."""

    family_name = "hidream_o1"
    archetype = "unified_transformer"

    def get_trainer_class(self):
        # Import inside the method to avoid loading heavy training deps
        # (peft, transformers) just to register the family.
        from .trainer import HiDreamO1Trainer

        return HiDreamO1Trainer
