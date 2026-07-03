"""Task 4.1 — Capability gating + schema surface verification for krea2.

Tests:
  1. test_krea2_capabilities_latent_diffusion — krea2-raw resolves as a proper
     latent_diffusion image family (has_vae, has_external_te, NOT video/audio/dual_expert).
  2. test_krea2_in_enriched_schema — enrich_schema injects "krea2" into model_family
     enum and both krea2-raw + krea2-turbo into backend_map["krea2"].
"""

from __future__ import annotations

from app.engine.models.registry import ModelRegistry
from app.engine.core.archetypes import resolve_capabilities
from app.engine.models.base import BaseTrainingConfig
from app.engine.models.training_plugin import StandardPlugin


# ── Step 1 — Capability gating ───────────────────────────────────────────────


def test_krea2_capabilities_latent_diffusion():
    """krea2-raw must resolve as a latent_diffusion image family."""
    ModelRegistry.initialize()

    defn = next(
        d for d in ModelRegistry._definitions.values() if d.id == "krea2-raw"
    )
    result = resolve_capabilities(defn)
    caps = result["capabilities"]

    # latent_diffusion image family — has VAE + external TE
    assert caps["has_vae"] is True, f"expected has_vae=True, got {caps['has_vae']!r}"
    assert caps["has_external_te"] is True, (
        f"expected has_external_te=True, got {caps['has_external_te']!r}"
    )

    # NOT a video / audio / dual-expert family
    assert caps.get("is_video", False) is False, (
        f"expected is_video=False, got {caps.get('is_video')!r}"
    )
    assert caps.get("has_audio", False) is False, (
        f"expected has_audio=False, got {caps.get('has_audio')!r}"
    )
    assert caps.get("dual_expert", False) is False, (
        f"expected dual_expert=False, got {caps.get('dual_expert')!r}"
    )

    # Archetype must be latent_diffusion
    assert result["archetype"] == "latent_diffusion", (
        f"expected archetype='latent_diffusion', got {result['archetype']!r}"
    )

    # Field visibility: latent/TE caching must be supported; video fields hidden
    vis = result["field_visibility"]
    assert vis["cache_latents"]["supported"] is True, (
        f"cache_latents should be supported; got {vis['cache_latents']}"
    )
    assert vis["cache_text_embeddings"]["supported"] is True, (
        f"cache_text_embeddings should be supported; got {vis['cache_text_embeddings']}"
    )
    assert vis["unload_text_encoder"]["supported"] is True, (
        f"unload_text_encoder should be supported; got {vis['unload_text_encoder']}"
    )

    # Video fields hidden for an image family
    assert vis["num_frames"]["supported"] is False, (
        f"num_frames should be hidden for image model; got {vis['num_frames']}"
    )
    assert vis["train_audio"]["supported"] is False, (
        f"train_audio should be hidden for image model; got {vis['train_audio']}"
    )


# ── Step 2 — Frontend surface (enrich_schema) ────────────────────────────────


def test_krea2_in_enriched_schema():
    """enrich_schema must surface krea2 in model_family and both definitions
    in definition_id backend_map[krea2]."""
    ModelRegistry.initialize()

    schema = BaseTrainingConfig.model_json_schema()
    enriched = StandardPlugin().enrich_schema(schema)

    props = enriched.get("properties", {})

    # model_family enum must include "krea2"
    family_enum = props.get("model_family", {}).get("enum", [])
    assert "krea2" in family_enum, (
        f"'krea2' not found in model_family enum: {family_enum}"
    )

    # definition_id backend_map must contain a "krea2" key
    backend_map = props.get("definition_id", {}).get("backend_map", {})
    assert "krea2" in backend_map, (
        f"'krea2' not found in definition_id backend_map keys: {list(backend_map.keys())}"
    )

    krea2_defs = backend_map["krea2"]
    assert "krea2-raw" in krea2_defs, (
        f"'krea2-raw' not in backend_map['krea2']: {krea2_defs}"
    )
    assert "krea2-turbo" in krea2_defs, (
        f"'krea2-turbo' not in backend_map['krea2']: {krea2_defs}"
    )

    # Both definitions must also appear in the flat definition_id enum
    def_enum = props.get("definition_id", {}).get("enum", [])
    assert "krea2-raw" in def_enum, (
        f"'krea2-raw' not in definition_id enum: {def_enum}"
    )
    assert "krea2-turbo" in def_enum, (
        f"'krea2-turbo' not in definition_id enum: {def_enum}"
    )
