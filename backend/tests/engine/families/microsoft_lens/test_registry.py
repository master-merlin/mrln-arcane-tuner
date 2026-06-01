"""microsoft_lens family registration tests."""
import pytest

from app.engine.models.registry import ModelRegistry


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset registry class-level state before and after each test."""
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._paths = {}
    ModelRegistry._discovered = False
    ModelRegistry._definitions_loaded = False
    yield
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._paths = {}
    ModelRegistry._discovered = False
    ModelRegistry._definitions_loaded = False


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
    instance = object.__new__(MicrosoftLensFamily)
    assert instance.get_trainer_class() is MicrosoftLensTrainer
