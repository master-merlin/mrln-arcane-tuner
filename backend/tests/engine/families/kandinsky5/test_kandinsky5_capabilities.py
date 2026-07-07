"""Capability gating + schema surface verification for kandinsky5.

1. Both definitions resolve as latent_diffusion VIDEO families (is_video,
   video fields visible, no audio / dual_expert / image encoder).
2. enrich_schema surfaces "kandinsky5" in the model_family enum and both
   definitions in the definition_id backend_map.
"""

from __future__ import annotations

import pytest

from app.engine.core.archetypes import resolve_capabilities
from app.engine.models.base import BaseTrainingConfig
from app.engine.models.registry import ModelRegistry
from app.engine.models.training_plugin import StandardPlugin

DEF_IDS = ["k5-t2v-lite-sft-5s", "k5-i2v-pro-sft-5s"]


@pytest.mark.parametrize("def_id", DEF_IDS)
def test_kandinsky5_capabilities_video_latent_diffusion(def_id):
    ModelRegistry.initialize()

    defn = next(d for d in ModelRegistry._definitions.values() if d.id == def_id)
    result = resolve_capabilities(defn)
    caps = result["capabilities"]

    assert result["archetype"] == "latent_diffusion"
    assert caps["has_vae"] is True
    assert caps["has_external_te"] is True

    # Video family — the whole point.
    assert caps["is_video"] is True
    assert caps["native_fps"] == 24
    assert caps.get("has_audio", False) is False
    assert caps.get("dual_expert", False) is False
    assert caps.get("has_image_encoder", True) is False  # latent-path i2v

    # Frozen dual TE — no TE training.
    assert caps["supports_train_te"] is False

    vis = result["field_visibility"]
    assert vis["cache_latents"]["supported"] is True
    assert vis["cache_text_embeddings"]["supported"] is True
    # Video fields VISIBLE (image families hide them).
    assert vis["num_frames"]["supported"] is True
    assert vis["target_fps"]["supported"] is True
    assert vis["video_mode"]["supported"] is True
    assert vis["temporal_coverage"]["supported"] is True
    # Audio + expert routing hidden.
    assert vis["train_audio"]["supported"] is False
    assert vis["expert_mode"]["supported"] is False


def test_kandinsky5_in_enriched_schema():
    ModelRegistry.initialize()

    schema = BaseTrainingConfig.model_json_schema()
    enriched = StandardPlugin().enrich_schema(schema)
    props = enriched.get("properties", {})

    family_enum = props.get("model_family", {}).get("enum", [])
    assert "kandinsky5" in family_enum, (
        f"'kandinsky5' not in model_family enum: {family_enum}"
    )

    backend_map = props.get("definition_id", {}).get("backend_map", {})
    assert "kandinsky5" in backend_map, (
        f"'kandinsky5' not in backend_map keys: {list(backend_map.keys())}"
    )
    for def_id in DEF_IDS:
        assert def_id in backend_map["kandinsky5"]

    def_enum = props.get("definition_id", {}).get("enum", [])
    for def_id in DEF_IDS:
        assert def_id in def_enum
