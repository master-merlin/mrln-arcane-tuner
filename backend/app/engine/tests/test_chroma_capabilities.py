"""Capability gating + schema surface verification for chroma.

Mirrors test_ovis_image_capabilities.py:
  1. Both chroma1-base and chroma1-hd resolve as proper latent_diffusion
     image families (has_vae, has_external_te, NOT video/audio/dual_expert).
  2. enrich_schema surfaces "chroma" in the model_family enum and both
     definition IDs in the definition_id backend_map.
"""

from __future__ import annotations

import pytest

from app.engine.models.registry import ModelRegistry
from app.engine.core.archetypes import resolve_capabilities
from app.engine.models.base import BaseTrainingConfig
from app.engine.models.training_plugin import StandardPlugin


# ── Step 1 — Capability gating ───────────────────────────────────────────────


@pytest.mark.parametrize("def_id", ["chroma1-base", "chroma1-hd"])
def test_chroma_capabilities_latent_diffusion(def_id):
    """Both Chroma definitions must resolve as latent_diffusion image families."""
    ModelRegistry.initialize()

    defn = next(
        d for d in ModelRegistry._definitions.values() if d.id == def_id
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

    # No TE training for chroma — T5 stays frozen
    assert caps["supports_train_te"] is False

    # Video fields hidden for an image family
    assert vis["num_frames"]["supported"] is False
    assert vis["train_audio"]["supported"] is False


# ── Step 2 — Frontend surface (enrich_schema) ────────────────────────────────


def test_chroma_in_enriched_schema():
    """enrich_schema must surface chroma + both definitions."""
    ModelRegistry.initialize()

    schema = BaseTrainingConfig.model_json_schema()
    enriched = StandardPlugin().enrich_schema(schema)

    props = enriched.get("properties", {})

    family_enum = props.get("model_family", {}).get("enum", [])
    assert "chroma" in family_enum, (
        f"'chroma' not found in model_family enum: {family_enum}"
    )

    backend_map = props.get("definition_id", {}).get("backend_map", {})
    assert "chroma" in backend_map, (
        f"'chroma' not in backend_map keys: {list(backend_map.keys())}"
    )
    assert "chroma1-base" in backend_map["chroma"]
    assert "chroma1-hd" in backend_map["chroma"]

    def_enum = props.get("definition_id", {}).get("enum", [])
    assert "chroma1-base" in def_enum
    assert "chroma1-hd" in def_enum
