"""Capability gating + schema surface verification for ace_step15.

Mirrors test_hunyuan_video15_capabilities.py for the AUDIO case:
  1. The definition resolves as a latent_diffusion AUDIO model
     (is_audio_family, NOT is_video / dual_expert / has_image_encoder) with
     the AUDIO config fields visible and VIDEO/resolution/augmentation/
     masking fields hidden.
  2. enrich_schema surfaces "ace_step15" in the model_family enum and the
     definition id in the definition_id backend_map.
"""

from __future__ import annotations

from app.engine.core.archetypes import resolve_capabilities
from app.engine.models.base import BaseTrainingConfig
from app.engine.models.registry import ModelRegistry
from app.engine.models.training_plugin import StandardPlugin

_DEF_ID = "ace-step-1.5"


def _get_definition():
    ModelRegistry.initialize()
    return next(d for d in ModelRegistry._definitions.values() if d.id == _DEF_ID)


# ── Step 1 — Capability gating ───────────────────────────────────────────────


def test_ace_step15_capabilities_latent_diffusion_audio():
    defn = _get_definition()
    result = resolve_capabilities(defn)
    caps = result["capabilities"]

    assert result["archetype"] == "latent_diffusion"
    assert caps["has_vae"] is True
    assert caps["has_external_te"] is True
    assert caps["te_cache"] is True
    assert caps["latent_cache"] is True
    assert caps["is_audio_family"] is True
    assert caps["supports_spatial_resolution"] is False
    # NOT a video family, NOT dual-expert, NOT an image encoder.
    assert caps.get("is_video", False) is False
    assert caps.get("dual_expert", False) is False
    assert caps.get("has_image_encoder", False) is False
    assert caps["supports_train_te"] is False

    vis = result["field_visibility"]
    # AUDIO fields visible…
    assert vis["duration_s"]["supported"] is True
    assert vis["genre_ratio"]["supported"] is True
    # …VIDEO fields hidden (is_video stays False)…
    assert vis["num_frames"]["supported"] is False
    assert vis["target_fps"]["supported"] is False
    assert vis["video_mode"]["supported"] is False
    assert vis["temporal_coverage"]["supported"] is False
    assert vis["train_audio"]["supported"] is False  # video-has-audio-track flag
    assert vis["expert_mode"]["supported"] is False
    # …spatial resolution/bucketing hidden (no spatial dimension)…
    assert vis["resolutions"]["supported"] is False
    assert vis["bucketing_mode"]["supported"] is False
    # …pixel augmentation/masking hidden (nothing to flip/mask in audio)…
    assert vis["h_flip"]["supported"] is False
    assert vis["v_flip"]["supported"] is False
    assert vis["masking_enabled"]["supported"] is False
    # Standard latent-diffusion caching fields stay available.
    assert vis["cache_latents"]["supported"] is True
    assert vis["cache_text_embeddings"]["supported"] is True
    assert vis["unload_text_encoder"]["supported"] is True


def test_ace_step15_not_edit_model():
    defn = _get_definition()
    result = resolve_capabilities(defn)
    caps = result["capabilities"]
    assert caps["control_inputs"] == 0
    assert caps["is_edit"] is False


# ── Step 2 — Frontend surface (enrich_schema) ────────────────────────────────


def test_ace_step15_in_enriched_schema():
    ModelRegistry.initialize()

    schema = BaseTrainingConfig.model_json_schema()
    enriched = StandardPlugin().enrich_schema(schema)
    props = enriched.get("properties", {})

    family_enum = props.get("model_family", {}).get("enum", [])
    assert "ace_step15" in family_enum, (
        f"'ace_step15' not in model_family enum: {family_enum}"
    )

    backend_map = props.get("definition_id", {}).get("backend_map", {})
    assert "ace_step15" in backend_map, (
        f"'ace_step15' not in backend_map keys: {list(backend_map.keys())}"
    )
    assert _DEF_ID in backend_map["ace_step15"]

    def_enum = props.get("definition_id", {}).get("enum", [])
    assert _DEF_ID in def_enum


# ── SamplePromptConfig / BaseTrainingConfig schema fields ───────────────────


def test_sample_prompt_config_has_audio_seams():
    from app.engine.models.base import SamplePromptConfig

    fields = SamplePromptConfig.model_fields
    assert "lyrics" in fields
    assert "duration_s" in fields
    assert fields["lyrics"].default is None
    assert fields["duration_s"].default is None


def test_base_training_config_has_audio_group_defaults():
    fields = BaseTrainingConfig.model_fields
    assert fields["duration_s"].default == 30.0
    assert fields["genre_ratio"].default == 0.15
    assert fields["duration_s"].json_schema_extra["group"] == "AUDIO"
    assert fields["genre_ratio"].json_schema_extra["group"] == "AUDIO"
