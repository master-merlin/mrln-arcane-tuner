"""WAN 2.2 TI2V-5B registration tests (mirrors wan21/wan22 registration).

Pins: the family is discovered under its OWN family_name (not "wan22" — a
second directory could not safely reuse that name; the registry's
``register_family`` silently overwrites on a duplicate key), capability
overrides mark it video + no-image-encoder + NOT dual_expert (the seam that
auto-hides the MoE-only fields for this family via
``core/archetypes.py``'s field-visibility table), and the trainer resolves.
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


def test_wan22_ti2v_5b_family_is_discovered():
    registry = ModelRegistry()
    registry.discover_families()
    fam_cls = registry.get_family_class("wan22_ti2v_5b")
    assert fam_cls.family_name == "wan22_ti2v_5b"
    assert fam_cls.archetype == "latent_diffusion"


def test_wan22_ti2v_5b_capability_overrides():
    registry = ModelRegistry()
    registry.discover_families()
    fam_cls = registry.get_family_class("wan22_ti2v_5b")
    overrides = fam_cls.capability_overrides
    assert overrides["is_video"] is True
    assert overrides["has_image_encoder"] is False
    # dual_expert is deliberately NOT set — must inherit the archetype's False
    # default so the MoE-only field-visibility rules hide expert_mode/etc.
    assert "dual_expert" not in overrides


def test_wan22_and_wan21_still_register_independently():
    """Adding the new family must not clobber the existing wan21/wan22 entries
    (the registry overwrites on a duplicate family_name — this is the risk a
    THIRD directory reusing "wan22" would have created)."""
    registry = ModelRegistry()
    registry.discover_families()
    assert registry.get_family_class("wan21").family_name == "wan21"
    assert registry.get_family_class("wan22").family_name == "wan22"
    assert registry.get_family_class("wan22_ti2v_5b").family_name == "wan22_ti2v_5b"


def test_wan22_ti2v_5b_trainer_class_resolves():
    from app.engine.models.families.wan22_ti2v_5b.family import Wan22Ti2v5bFamily
    from app.engine.models.families.wan22_ti2v_5b.trainer import Wan22Ti2v5bTrainer

    fam = object.__new__(Wan22Ti2v5bFamily)
    fam.definition = type("D", (), {"control_inputs": 0})()
    assert fam.get_trainer_class() is Wan22Ti2v5bTrainer
