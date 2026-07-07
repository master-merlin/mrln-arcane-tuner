"""Capability gating + schema surface verification for prx_pixel.

Mirrors test_prx_capabilities.py:
  1. prx-pixel-t2i resolves as a pixel_transformer image family
     (NO VAE, but an EXTERNAL cacheable TE — the axis that distinguishes
     it from both latent_diffusion and unified_transformer).
  2. enrich_schema surfaces "prx_pixel" in the model_family enum and
     "prx-pixel-t2i" in the definition_id backend_map.
"""

from __future__ import annotations

from app.engine.models.registry import ModelRegistry
from app.engine.core.archetypes import resolve_capabilities
from app.engine.models.base import BaseTrainingConfig
from app.engine.models.training_plugin import StandardPlugin


# ── Step 1 — Capability gating ───────────────────────────────────────────────


def test_prx_pixel_capabilities_pixel_transformer():
    """prx-pixel-t2i must resolve as a pixel_transformer image family."""
    ModelRegistry.initialize()

    defn = next(
        d for d in ModelRegistry._definitions.values() if d.id == "prx-pixel-t2i"
    )
    result = resolve_capabilities(defn)
    caps = result["capabilities"]

    # THE distinguishing pair: no VAE, but a real external TE.
    assert caps["has_vae"] is False
    assert caps["has_external_te"] is True
    assert caps["latent_cache"] is False
    assert caps["te_cache"] is True

    assert caps.get("is_video", False) is False
    assert caps.get("has_audio", False) is False
    assert caps.get("dual_expert", False) is False

    assert result["archetype"] == "pixel_transformer"

    vis = result["field_visibility"]
    # Hidden: everything that presumes a VAE/latent space.
    assert vis["cache_latents"]["supported"] is False
    assert vis["cache_latents"].get("reason")
    assert vis["low_vram"]["supported"] is False
    # Kept: the external TE's cache + offload toggles.
    assert vis["cache_text_embeddings"]["supported"] is True
    assert vis["unload_text_encoder"]["supported"] is True

    # No TE training / quantization / block swap for prx_pixel.
    assert caps["supports_train_te"] is False
    assert vis["train_text_encoder"]["supported"] is False
    assert vis["te_quantization"]["supported"] is False
    assert vis["block_swap_config"]["supported"] is False

    # Video fields hidden for an image family
    assert vis["num_frames"]["supported"] is False
    assert vis["train_audio"]["supported"] is False

    # Native 1024 checkpoint default flows through.
    assert result["defaults"]["resolution"] == 1024


# ── Step 2 — Frontend surface (enrich_schema) ────────────────────────────────


def test_prx_pixel_in_enriched_schema():
    """enrich_schema must surface prx_pixel + prx-pixel-t2i."""
    ModelRegistry.initialize()

    schema = BaseTrainingConfig.model_json_schema()
    enriched = StandardPlugin().enrich_schema(schema)

    props = enriched.get("properties", {})

    family_enum = props.get("model_family", {}).get("enum", [])
    assert "prx_pixel" in family_enum, (
        f"'prx_pixel' not found in model_family enum: {family_enum}"
    )

    backend_map = props.get("definition_id", {}).get("backend_map", {})
    assert "prx_pixel" in backend_map, (
        f"'prx_pixel' not in backend_map keys: {list(backend_map.keys())}"
    )
    assert "prx-pixel-t2i" in backend_map["prx_pixel"]

    def_enum = props.get("definition_id", {}).get("enum", [])
    assert "prx-pixel-t2i" in def_enum
