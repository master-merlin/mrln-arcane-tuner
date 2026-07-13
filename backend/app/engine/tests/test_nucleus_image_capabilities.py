"""Capability gating + schema surface verification for nucleus_image.

Mirrors test_lumina2_capabilities.py:
  1. nucleus-image resolves as a proper latent_diffusion image family
     (has_vae, has_external_te, NOT video/audio/dual_expert).
  2. enrich_schema surfaces "nucleus_image" in the model_family enum and the
     definition ID in the definition_id backend_map.
"""

from __future__ import annotations

from app.engine.models.registry import ModelRegistry
from app.engine.core.archetypes import resolve_capabilities
from app.engine.models.base import BaseTrainingConfig
from app.engine.models.training_plugin import StandardPlugin


# ── Step 1 — Capability gating ───────────────────────────────────────────────


def test_nucleus_image_capabilities_latent_diffusion():
    """nucleus-image must resolve as a latent_diffusion image family."""
    ModelRegistry.initialize()

    defn = next(
        d for d in ModelRegistry._definitions.values() if d.id == "nucleus-image"
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

    # No TE training for nucleus_image — Qwen3-VL stays frozen
    assert caps["supports_train_te"] is False

    # Video fields hidden for an image family
    assert vis["num_frames"]["supported"] is False
    assert vis["train_audio"]["supported"] is False


# ── Step 2 — Frontend surface (enrich_schema) ────────────────────────────────


def test_nucleus_image_in_enriched_schema():
    """enrich_schema must surface nucleus_image + its definition."""
    ModelRegistry.initialize()

    schema = BaseTrainingConfig.model_json_schema()
    enriched = StandardPlugin().enrich_schema(schema)

    props = enriched.get("properties", {})

    family_enum = props.get("model_family", {}).get("enum", [])
    assert "nucleus_image" in family_enum, (
        f"'nucleus_image' not found in model_family enum: {family_enum}"
    )

    backend_map = props.get("definition_id", {}).get("backend_map", {})
    assert "nucleus_image" in backend_map, (
        f"'nucleus_image' not in backend_map keys: {list(backend_map.keys())}"
    )
    assert "nucleus-image" in backend_map["nucleus_image"]

    def_enum = props.get("definition_id", {}).get("enum", [])
    assert "nucleus-image" in def_enum
