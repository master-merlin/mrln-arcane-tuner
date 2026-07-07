"""Kandinsky 5.0 family registration tests.

After ``discover_families()`` the ``kandinsky5`` family is registered with the
expected capability overrides (incl. ``is_video``) and resolves its trainer.
"""

import pytest

from app.engine.models.registry import ModelRegistry


@pytest.fixture(autouse=True)
def clean_registry():
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


def test_kandinsky5_family_is_discovered():
    registry = ModelRegistry()
    registry.discover_families()
    fam_cls = registry.get_family_class("kandinsky5")
    assert fam_cls.family_name == "kandinsky5"
    assert fam_cls.archetype == "latent_diffusion"


def test_kandinsky5_capability_overrides():
    registry = ModelRegistry()
    registry.discover_families()
    fam_cls = registry.get_family_class("kandinsky5")
    overrides = fam_cls.capability_overrides
    assert overrides["is_video"] is True
    # No CLIP-vision image encoder — I2V conditions through the latent path.
    assert overrides["has_image_encoder"] is False
    assert overrides["native_fps"] == 24
    # supports_train_te stays the latent_diffusion archetype default (False);
    # a redundant override here is reserved for SDXL by a project invariant.
    assert "supports_train_te" not in overrides


def test_kandinsky5_trainer_class_resolves():
    from app.engine.models.families.kandinsky5.family import Kandinsky5Family
    from app.engine.models.families.kandinsky5.trainer import Kandinsky5Trainer

    fam = object.__new__(Kandinsky5Family)
    assert fam.get_trainer_class() is Kandinsky5Trainer
