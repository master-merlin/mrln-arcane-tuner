"""LTX 2.3 family registration + capability-override tests."""

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


def test_ltx2_family_is_discovered():
    registry = ModelRegistry()
    registry.discover_families()
    assert "ltx2" in registry._families
    fam_cls = registry.get_family_class("ltx2")
    assert fam_cls.family_name == "ltx2"
    assert fam_cls.archetype == "latent_diffusion"


def test_ltx2_capability_overrides_video_and_audio():
    registry = ModelRegistry()
    registry.discover_families()
    fam_cls = registry.get_family_class("ltx2")
    overrides = getattr(fam_cls, "capability_overrides", {})
    # The new video-program flags live directly on the family.
    assert overrides.get("is_video") is True
    assert overrides.get("has_audio") is True
    # TE training is left at the archetype default (False) — not overridden —
    # so the "only SDXL overrides supports_train_te" cross-family guard holds.
    assert "supports_train_te" not in overrides


def test_ltx2_trainer_class_resolves():
    from app.engine.models.families.ltx2.family import Ltx2Family
    from app.engine.models.families.ltx2.trainer import Ltx2Trainer

    instance = object.__new__(Ltx2Family)
    assert instance.get_trainer_class() is Ltx2Trainer
