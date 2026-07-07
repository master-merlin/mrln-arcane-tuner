"""Capability gating + schema surface verification for dreamlite.

Mirrors test_ovis_image_capabilities.py:
  1. dreamlite-base AND dreamlite-mobile resolve as proper latent_diffusion
     image families (has_vae, has_external_te, NOT video/audio/dual_expert).
  2. enrich_schema surfaces "dreamlite" in the model_family enum and both
     definitions in the definition_id backend_map.
"""

from __future__ import annotations

import pytest

from app.engine.core.archetypes import resolve_capabilities
from app.engine.models.base import BaseTrainingConfig
from app.engine.models.registry import ModelRegistry
from app.engine.models.training_plugin import StandardPlugin


# ── Step 1 — Capability gating ───────────────────────────────────────────────


@pytest.mark.parametrize("def_id", ["dreamlite-base", "dreamlite-mobile"])
def test_dreamlite_capabilities_latent_diffusion(def_id):
    """Both dreamlite definitions must resolve as latent_diffusion."""
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

    # No TE training for dreamlite
    assert caps["supports_train_te"] is False

    # Video fields hidden for an image family
    assert vis["num_frames"]["supported"] is False
    assert vis["train_audio"]["supported"] is False


# ── Step 2 — Frontend surface (enrich_schema) ────────────────────────────────


def test_dreamlite_in_enriched_schema():
    """enrich_schema must surface dreamlite + both definitions."""
    ModelRegistry.initialize()

    schema = BaseTrainingConfig.model_json_schema()
    enriched = StandardPlugin().enrich_schema(schema)

    props = enriched.get("properties", {})

    family_enum = props.get("model_family", {}).get("enum", [])
    assert "dreamlite" in family_enum, (
        f"'dreamlite' not found in model_family enum: {family_enum}"
    )

    backend_map = props.get("definition_id", {}).get("backend_map", {})
    assert "dreamlite" in backend_map, (
        f"'dreamlite' not in backend_map keys: {list(backend_map.keys())}"
    )
    assert "dreamlite-base" in backend_map["dreamlite"]
    assert "dreamlite-mobile" in backend_map["dreamlite"]

    def_enum = props.get("definition_id", {}).get("enum", [])
    assert "dreamlite-base" in def_enum
    assert "dreamlite-mobile" in def_enum
