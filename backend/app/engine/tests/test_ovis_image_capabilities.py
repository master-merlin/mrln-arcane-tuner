"""Capability gating + schema surface verification for ovis_image.

Mirrors test_krea2_capabilities.py:
  1. ovis-image-base resolves as a proper latent_diffusion image family
     (has_vae, has_external_te, NOT video/audio/dual_expert).
  2. enrich_schema surfaces "ovis_image" in the model_family enum and
     "ovis-image-base" in the definition_id backend_map.
"""

from __future__ import annotations

from app.engine.models.registry import ModelRegistry
from app.engine.core.archetypes import resolve_capabilities
from app.engine.models.base import BaseTrainingConfig
from app.engine.models.training_plugin import StandardPlugin


# ── Step 1 — Capability gating ───────────────────────────────────────────────


def test_ovis_image_capabilities_latent_diffusion():
    """ovis-image-base must resolve as a latent_diffusion image family."""
    ModelRegistry.initialize()

    defn = next(
        d for d in ModelRegistry._definitions.values() if d.id == "ovis-image-base"
    )
    result = resolve_capabilities(defn)
    caps = result["capabilities"]

    assert caps["has_vae"] is True
    assert caps["has_external_te"] is True

    assert caps.get("is_video", False) is False
    assert caps.get("has_audio", False) is False
    assert caps.get("dual_expert", False) is False

    assert result["archetype"] == "latent_diffusion"

    vis = result["field_visibility"]
    assert vis["cache_latents"]["supported"] is True
    assert vis["cache_text_embeddings"]["supported"] is True
    assert vis["unload_text_encoder"]["supported"] is True

    # No TE training for ovis_image
    assert caps["supports_train_te"] is False

    # Video fields hidden for an image family
    assert vis["num_frames"]["supported"] is False
    assert vis["train_audio"]["supported"] is False


# ── Step 2 — Frontend surface (enrich_schema) ────────────────────────────────


def test_ovis_image_in_enriched_schema():
    """enrich_schema must surface ovis_image + ovis-image-base."""
    ModelRegistry.initialize()

    schema = BaseTrainingConfig.model_json_schema()
    enriched = StandardPlugin().enrich_schema(schema)

    props = enriched.get("properties", {})

    family_enum = props.get("model_family", {}).get("enum", [])
    assert "ovis_image" in family_enum, (
        f"'ovis_image' not found in model_family enum: {family_enum}"
    )

    backend_map = props.get("definition_id", {}).get("backend_map", {})
    assert "ovis_image" in backend_map, (
        f"'ovis_image' not in backend_map keys: {list(backend_map.keys())}"
    )
    assert "ovis-image-base" in backend_map["ovis_image"]

    def_enum = props.get("definition_id", {}).get("enum", [])
    assert "ovis-image-base" in def_enum
