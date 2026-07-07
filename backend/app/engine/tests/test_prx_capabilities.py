"""Capability gating + schema surface verification for prx.

Mirrors test_ovis_image_capabilities.py:
  1. prx-sft resolves as a proper latent_diffusion image family
     (has_vae, has_external_te, NOT video/audio/dual_expert).
  2. enrich_schema surfaces "prx" in the model_family enum and
     "prx-sft" in the definition_id backend_map.
"""

from __future__ import annotations

from app.engine.models.registry import ModelRegistry
from app.engine.core.archetypes import resolve_capabilities
from app.engine.models.base import BaseTrainingConfig
from app.engine.models.training_plugin import StandardPlugin


# ── Step 1 — Capability gating ───────────────────────────────────────────────


def test_prx_capabilities_latent_diffusion():
    """prx-sft must resolve as a latent_diffusion image family."""
    ModelRegistry.initialize()

    defn = next(
        d for d in ModelRegistry._definitions.values() if d.id == "prx-sft"
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

    # No TE training for prx
    assert caps["supports_train_te"] is False

    # Video fields hidden for an image family
    assert vis["num_frames"]["supported"] is False
    assert vis["train_audio"]["supported"] is False


# ── Step 2 — Frontend surface (enrich_schema) ────────────────────────────────


def test_prx_in_enriched_schema():
    """enrich_schema must surface prx + prx-sft."""
    ModelRegistry.initialize()

    schema = BaseTrainingConfig.model_json_schema()
    enriched = StandardPlugin().enrich_schema(schema)

    props = enriched.get("properties", {})

    family_enum = props.get("model_family", {}).get("enum", [])
    assert "prx" in family_enum, (
        f"'prx' not found in model_family enum: {family_enum}"
    )

    backend_map = props.get("definition_id", {}).get("backend_map", {})
    assert "prx" in backend_map, (
        f"'prx' not in backend_map keys: {list(backend_map.keys())}"
    )
    assert "prx-sft" in backend_map["prx"]

    def_enum = props.get("definition_id", {}).get("enum", [])
    assert "prx-sft" in def_enum
