"""Capability gating + schema surface verification for longcat_image.

Tests:
  1. longcat-image-base resolves as a proper latent_diffusion image family
     (has_vae, has_external_te, NOT video/audio/dual_expert/edit).
  2. enrich_schema surfaces "longcat_image" in the model_family enum and
     "longcat-image-base" in the definition_id backend_map.
"""

from __future__ import annotations

from app.engine.models.registry import ModelRegistry
from app.engine.core.archetypes import resolve_capabilities
from app.engine.models.base import BaseTrainingConfig
from app.engine.models.training_plugin import StandardPlugin


def test_longcat_image_capabilities_latent_diffusion():
    """longcat-image-base must resolve as a latent_diffusion image family."""
    ModelRegistry.initialize()

    defn = next(
        d for d in ModelRegistry._definitions.values()
        if d.id == "longcat-image-base"
    )
    result = resolve_capabilities(defn)
    caps = result["capabilities"]

    assert result["archetype"] == "latent_diffusion"
    assert caps["has_vae"] is True
    assert caps["has_external_te"] is True
    assert caps["supports_train_te"] is False

    # NOT a video / audio / dual-expert / edit family
    assert caps.get("is_video", False) is False
    assert caps.get("has_audio", False) is False
    assert caps.get("dual_expert", False) is False
    assert caps.get("control_inputs", 0) == 0, "text-to-image only (no edit mode)"

    # Field visibility: latent/TE caching supported; video fields hidden
    vis = result["field_visibility"]
    assert vis["cache_latents"]["supported"] is True
    assert vis["cache_text_embeddings"]["supported"] is True
    assert vis["unload_text_encoder"]["supported"] is True
    assert vis["num_frames"]["supported"] is False
    assert vis["train_audio"]["supported"] is False


def test_longcat_image_in_enriched_schema():
    """enrich_schema must surface longcat_image + longcat-image-base."""
    ModelRegistry.initialize()

    schema = BaseTrainingConfig.model_json_schema()
    enriched = StandardPlugin().enrich_schema(schema)
    props = enriched.get("properties", {})

    family_enum = props.get("model_family", {}).get("enum", [])
    assert "longcat_image" in family_enum, (
        f"'longcat_image' not in model_family enum: {family_enum}"
    )

    backend_map = props.get("definition_id", {}).get("backend_map", {})
    assert "longcat_image" in backend_map, (
        f"'longcat_image' not in backend_map keys: {list(backend_map.keys())}"
    )
    assert "longcat-image-base" in backend_map["longcat_image"]

    def_enum = props.get("definition_id", {}).get("enum", [])
    assert "longcat-image-base" in def_enum
