"""minimax_h3 family registration + capability flags."""

from __future__ import annotations

from app.engine.models.registry import ModelRegistry


def _reload_registry() -> type[ModelRegistry]:
    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry.discover_families()
    return ModelRegistry


def test_family_is_auto_discovered():
    assert "minimax_h3" in _reload_registry()._families


def test_capability_flags_declare_video_and_audio():
    family_cls = _reload_registry()._families["minimax_h3"]
    caps = family_cls.capability_overrides
    assert caps["is_video"] is True
    assert caps["has_audio"] is True
    assert caps["has_image_encoder"] is True
    # H3 is single-stream: there is no second expert to schedule.
    assert caps["dual_expert"] is False
    # supports_train_te is intentionally NOT asserted here: only sdxl may put
    # that key in capability_overrides (test_only_sdxl_overrides_train_te);
    # minimax_h3 relies on the latent_diffusion archetype's False default.
    # The 48 GB Qwen3-VL TE must be cacheable.
    assert caps["te_cache"] is True
