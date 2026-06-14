"""WAN 2.2 registration tests.

After ``discover_families()`` the ``wan22`` family is registered with the
expected capability overrides (``is_video`` + ``dual_expert``), while
``wan_shared`` (no ``family.py``) stays unregistered. The trainer class resolves.
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


def test_wan22_family_is_discovered():
    registry = ModelRegistry()
    registry.discover_families()
    fam_cls = registry.get_family_class("wan22")
    assert fam_cls.family_name == "wan22"
    assert fam_cls.archetype == "latent_diffusion"


def test_wan22_capability_overrides_include_dual_expert_and_video():
    registry = ModelRegistry()
    registry.discover_families()
    fam_cls = registry.get_family_class("wan22")
    overrides = fam_cls.capability_overrides
    assert overrides["is_video"] is True
    assert overrides["dual_expert"] is True
    # WAN 2.2 I2V has NO CLIP image encoder (first-frame latent only).
    assert overrides["has_image_encoder"] is False
    # supports_train_te inherited from archetype (frozen UMT5) — not repeated.
    assert "supports_train_te" not in overrides


def test_wan22_trainer_class_resolves():
    from app.engine.models.families.wan22.family import Wan22Family
    from app.engine.models.families.wan22.trainer import Wan22Trainer

    fam = object.__new__(Wan22Family)
    assert fam.get_trainer_class() is Wan22Trainer


def test_wan_shared_still_not_registered():
    registry = ModelRegistry()
    registry.discover_families()
    assert "wan_shared" not in ModelRegistry._families
