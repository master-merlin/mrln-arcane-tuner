"""microsoft_lens family registration tests."""
from app.engine.models.registry import ModelRegistry


def test_microsoft_lens_family_is_discovered():
    registry = ModelRegistry()
    registry.discover_families()
    fam_cls = registry.get_family_class("microsoft_lens")
    assert fam_cls.family_name == "microsoft_lens"


def test_microsoft_lens_trainer_class_resolves():
    from app.engine.models.families.microsoft_lens.family import (
        MicrosoftLensFamily,
    )
    from app.engine.models.families.microsoft_lens.trainer import (
        MicrosoftLensTrainer,
    )
    assert MicrosoftLensFamily.family_name == "microsoft_lens"
    assert MicrosoftLensFamily.get_trainer_class is not None
