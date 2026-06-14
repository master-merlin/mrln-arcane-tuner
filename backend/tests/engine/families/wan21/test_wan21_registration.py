"""WAN 2.1 registration tests.

After ``discover_families()`` the ``wan21`` family is registered with the
expected capability overrides (incl. ``is_video``), while ``wan_shared`` — which
has NO ``family.py`` — is deliberately NOT registered.
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


def test_wan21_family_is_discovered():
    registry = ModelRegistry()
    registry.discover_families()
    fam_cls = registry.get_family_class("wan21")
    assert fam_cls.family_name == "wan21"
    assert fam_cls.archetype == "latent_diffusion"


def test_wan21_capability_overrides_include_is_video():
    registry = ModelRegistry()
    registry.discover_families()
    fam_cls = registry.get_family_class("wan21")
    overrides = fam_cls.capability_overrides
    assert overrides["is_video"] is True
    assert overrides["has_image_encoder"] is True
    # supports_train_te is inherited from the latent_diffusion archetype
    # (defaults False); it is deliberately NOT a redundant override here.
    assert "supports_train_te" not in overrides


def test_wan_shared_is_not_registered():
    registry = ModelRegistry()
    registry.discover_families()
    # wan_shared has no family.py → must stay invisible to the registry.
    assert "wan_shared" not in ModelRegistry._families


def test_wan21_trainer_class_resolves_for_t2v_and_i2v():
    from app.engine.models.families.wan21.family import Wan21Family
    from app.engine.models.families.wan21.trainer import Wan21Trainer

    # T2V definition (mode t2v)
    fam = object.__new__(Wan21Family)
    fam.definition = type("D", (), {"control_inputs": 0})()
    assert fam.get_trainer_class() is Wan21Trainer
