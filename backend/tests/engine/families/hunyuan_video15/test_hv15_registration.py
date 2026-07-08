"""hunyuan_video15 registration tests.

After ``discover_families()`` the family is registered with the expected
capability overrides (incl. ``is_video``), and the trainer class resolves for
both modes through one family.
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


def test_hv15_family_is_discovered():
    registry = ModelRegistry()
    registry.discover_families()
    fam_cls = registry.get_family_class("hunyuan_video15")
    assert fam_cls.family_name == "hunyuan_video15"
    assert fam_cls.archetype == "latent_diffusion"


def test_hv15_capability_overrides_include_is_video():
    registry = ModelRegistry()
    registry.discover_families()
    fam_cls = registry.get_family_class("hunyuan_video15")
    overrides = fam_cls.capability_overrides
    assert overrides["is_video"] is True
    assert overrides["has_image_encoder"] is True
    assert overrides["native_fps"] == 24
    # supports_train_te stays the latent_diffusion archetype default (False);
    # a project invariant reserves that override for SDXL.
    assert "supports_train_te" not in overrides


def test_hv15_trainer_class_resolves():
    from app.engine.models.families.hunyuan_video15.family import (
        HunyuanVideo15Family,
    )
    from app.engine.models.families.hunyuan_video15.trainer import Hv15Trainer

    fam = object.__new__(HunyuanVideo15Family)
    assert fam.get_trainer_class() is Hv15Trainer
