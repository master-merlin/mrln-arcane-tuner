"""Tests that every model family declares a known archetype and that the
hydream_o1 / SDXL-specific assignments are correct."""

from app.engine.core.archetypes import ARCHETYPES
from app.engine.models.registry import registry


def _family_classes():
    registry.discover_families()
    # registry._families is {family_name: class}; return the class values.
    return list(registry._families.values())


def test_every_family_declares_known_archetype():
    fams = _family_classes()
    assert fams, "no families discovered"
    for fam in fams:
        assert getattr(fam, "archetype", None) in ARCHETYPES, (
            f"{fam} missing/unknown archetype"
        )


def test_hidream_is_unified_prx_pixel_is_pixel_and_others_latent():
    by_name = {getattr(f, "family_name", f.__name__): f for f in _family_classes()}
    for name, fam in by_name.items():
        lowered = name.lower()
        if "hidream" in lowered:
            expected = "unified_transformer"
        elif lowered == "prx_pixel":
            expected = "pixel_transformer"
        else:
            expected = "latent_diffusion"
        assert fam.archetype == expected, (name, fam.archetype)


def test_only_sdxl_overrides_train_te():
    for fam in _family_classes():
        ov = getattr(fam, "capability_overrides", {})
        name = getattr(fam, "family_name", fam.__name__).lower()
        if "sdxl" in name:
            assert ov.get("supports_train_te") is True
        else:
            assert "supports_train_te" not in ov
