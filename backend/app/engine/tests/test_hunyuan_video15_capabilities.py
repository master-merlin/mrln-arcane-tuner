"""Capability gating + schema surface verification for hunyuan_video15.

Mirrors test_ovis_image_capabilities.py for the video case:
  1. Both hv15 definitions resolve as latent_diffusion VIDEO models
     (is_video, has_image_encoder; NOT audio / dual_expert) with the video
     config fields visible and audio/expert fields hidden.
  2. enrich_schema surfaces "hunyuan_video15" in the model_family enum and
     both definition ids in the definition_id backend_map.
"""

from __future__ import annotations

from app.engine.core.archetypes import resolve_capabilities
from app.engine.models.base import BaseTrainingConfig
from app.engine.models.registry import ModelRegistry
from app.engine.models.training_plugin import StandardPlugin

_IDS = ("hv15-480p-t2v", "hv15-480p-i2v")


# ── Step 1 — Capability gating ───────────────────────────────────────────────


def test_hv15_capabilities_latent_diffusion_video():
    ModelRegistry.initialize()

    for def_id in _IDS:
        defn = next(
            d for d in ModelRegistry._definitions.values() if d.id == def_id
        )
        result = resolve_capabilities(defn)
        caps = result["capabilities"]

        assert result["archetype"] == "latent_diffusion"
        assert caps["has_vae"] is True
        assert caps["has_external_te"] is True
        assert caps["is_video"] is True
        assert caps["has_image_encoder"] is True
        assert caps["native_fps"] == 24
        assert caps.get("has_audio", False) is False
        assert caps.get("dual_expert", False) is False
        assert caps["supports_train_te"] is False

        vis = result["field_visibility"]
        # Video fields visible…
        assert vis["num_frames"]["supported"] is True
        assert vis["target_fps"]["supported"] is True
        assert vis["video_mode"]["supported"] is True
        assert vis["temporal_coverage"]["supported"] is True
        # …audio / dual-expert fields hidden.
        assert vis["train_audio"]["supported"] is False
        assert vis["expert_mode"]["supported"] is False
        # Standard latent-diffusion caching fields stay available.
        assert vis["cache_latents"]["supported"] is True
        assert vis["cache_text_embeddings"]["supported"] is True
        assert vis["unload_text_encoder"]["supported"] is True


# ── Step 2 — Frontend surface (enrich_schema) ────────────────────────────────


def test_hv15_in_enriched_schema():
    ModelRegistry.initialize()

    schema = BaseTrainingConfig.model_json_schema()
    enriched = StandardPlugin().enrich_schema(schema)
    props = enriched.get("properties", {})

    family_enum = props.get("model_family", {}).get("enum", [])
    assert "hunyuan_video15" in family_enum, (
        f"'hunyuan_video15' not in model_family enum: {family_enum}"
    )

    backend_map = props.get("definition_id", {}).get("backend_map", {})
    assert "hunyuan_video15" in backend_map, (
        f"'hunyuan_video15' not in backend_map keys: {list(backend_map.keys())}"
    )
    for def_id in _IDS:
        assert def_id in backend_map["hunyuan_video15"]

    def_enum = props.get("definition_id", {}).get("enum", [])
    for def_id in _IDS:
        assert def_id in def_enum
