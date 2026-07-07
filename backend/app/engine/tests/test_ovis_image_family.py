"""Tests for the ovis_image family (diffusers-0.39-native Ovis-Image).

TDD order (mirrors test_krea2_family.py):
  Task 1: family registration + definition loading
  Task 2: loader manifest (component specs + dtype policy)
  Task 3: driver (LoRA targets, forward_pass timestep scale, compute_target,
          encode_text replicating OvisImagePipeline.encode_prompt)
  Task 4: trainer override trio (encode_text tuple, _update_primary_model
          driver sync, transformer property) + TE disk-cache layout
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.engine.core.definitions import ModelDefinition


@pytest.fixture(autouse=True)
def _restore_model_registry():
    """Snapshot + restore ``ModelRegistry`` class state around every test.

    Registration tests mutate the registry's class-level discovery caches
    inline (resetting ``_discovered`` / ``_families`` / ``_definitions`` to
    force a re-scan). Left unrestored those mutations leak into later tests
    in the session (same pattern as test_krea2_family.py).
    """
    from app.engine.models.registry import ModelRegistry

    saved = {
        "_families": dict(ModelRegistry._families),
        "_definitions": dict(ModelRegistry._definitions),
        "_paths": dict(ModelRegistry._paths),
        "_discovered": ModelRegistry._discovered,
        "_definitions_loaded": ModelRegistry._definitions_loaded,
    }
    try:
        yield
    finally:
        ModelRegistry._families = saved["_families"]
        ModelRegistry._definitions = saved["_definitions"]
        ModelRegistry._paths = saved["_paths"]
        ModelRegistry._discovered = saved["_discovered"]
        ModelRegistry._definitions_loaded = saved["_definitions_loaded"]


def _make_ovis_definition(**kwargs) -> MagicMock:
    """Build a mock Ovis-Image ModelDefinition for loader/driver tests."""
    definition = MagicMock(spec=ModelDefinition)
    definition.family = "ovis_image"
    definition.id = kwargs.get("id", "ovis-image-test")
    definition.components = {}
    definition.architecture_params = kwargs.get("architecture_params", {})
    definition.lora_targetable_modules = kwargs.get("lora_targetable_modules", [])
    return definition


# ── Task 1: Family Registration ──────────────────────────────────────────────


def test_family_registered():
    """ovis_image family must appear in ModelRegistry with the correct archetype."""
    from app.engine.models.registry import ModelRegistry

    # Reset discovery state so this test is hermetic
    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry.discover_families()

    fam = ModelRegistry._families.get("ovis_image")
    assert fam is not None, "ovis_image family not registered"
    assert fam.archetype == "latent_diffusion", (
        f"expected archetype='latent_diffusion', got {fam.archetype!r}"
    )


def test_definition_loaded():
    """ovis-image-base definition must load from its YAML file."""
    from app.engine.models.registry import ModelRegistry

    # Full reset so definitions are re-scanned
    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._definitions_loaded = False
    ModelRegistry.initialize()

    fam_defs = {
        d.id: d
        for d in ModelRegistry._definitions.values()
        if d.family == "ovis_image"
    }
    assert "ovis-image-base" in fam_defs, (
        f"missing ovis-image-base definition; found: {set(fam_defs)}"
    )

    base = fam_defs["ovis-image-base"]
    # Canonical checkpoint repo (verified fact — see plan)
    assert base.components["repo"].path == "huggingface:AIDC-AI/Ovis-Image-7B", (
        f"wrong repo path: {base.components['repo'].path!r}"
    )
    # Standard T2I — no paired control inputs
    assert base.control_inputs == 0

    # Verified transformer config facts (from the checkpoint's config.json,
    # identical to the diffusers 0.39 OvisImageTransformer2DModel defaults).
    arch = base.architecture_params
    assert arch.get("transformer.num_layers") == 6
    assert arch.get("transformer.num_single_layers") == 27
    assert arch.get("transformer.num_attention_heads") == 24
    assert arch.get("transformer.attention_head_dim") == 128
    assert arch.get("transformer.joint_attention_dim") == 2048
    assert arch.get("transformer.in_channels") == 64
    assert arch.get("transformer.patch_size") == 1
    # Latent space is 16-channel (packed 2x2 -> 64 transformer channels)
    assert arch.get("vae.latent_channels") == 16
    # Scheduler facts from the checkpoint's scheduler_config.json
    assert arch.get("scheduler.use_dynamic_shifting") is True
    assert arch.get("scheduler.base_shift") == 0.5
    assert arch.get("scheduler.max_shift") == 1.15
    assert arch.get("scheduler.base_image_seq_len") == 256
    assert arch.get("scheduler.max_image_seq_len") == 4096
    # TE prompt-template facts from OvisImagePipeline
    assert arch.get("te.max_sequence_length") == 256
    assert arch.get("te.user_prompt_begin_id") == 28


# ── Task 2: Loader Manifest ──────────────────────────────────────────────────


def test_manifest_components():
    """OvisImageLoader manifest declares all four diffusers-native components."""
    import torch

    from app.engine.models.families.ovis_image.loader import OvisImageLoader

    loader = OvisImageLoader(torch.device("cpu"))
    definition = _make_ovis_definition()
    specs = loader.get_component_manifest(definition)

    keys = {s.key for s in specs}
    assert {"tokenizer", "text_encoder", "vae", "unet"} <= keys, (
        f"missing required manifest keys; got {keys}"
    )

    spec_map = {s.key: s for s in specs}

    # Tokenizer: AutoTokenizer (fast Qwen2TokenizerFast resolves via tokenizer.json)
    assert "AutoTokenizer" in spec_map["tokenizer"].hf_class, (
        f"tokenizer hf_class wrong: {spec_map['tokenizer'].hf_class}"
    )
    assert spec_map["tokenizer"].is_torch_model is False

    # Text encoder: plain transformers.Qwen3Model (text-only, hidden 2048)
    assert spec_map["text_encoder"].hf_class == "transformers.Qwen3Model", (
        f"text_encoder hf_class wrong: {spec_map['text_encoder'].hf_class}"
    )
    assert spec_map["text_encoder"].subfolder == "text_encoder"

    # VAE: standard diffusers AutoencoderKL
    assert spec_map["vae"].hf_class == "diffusers.AutoencoderKL", (
        f"vae hf_class wrong: {spec_map['vae'].hf_class}"
    )

    # Transformer mapped to "unet" (repo convention), diffusers-native class
    assert "OvisImageTransformer2DModel" in spec_map["unet"].hf_class, (
        f"unet hf_class wrong: {spec_map['unet'].hf_class}"
    )
    assert spec_map["unet"].subfolder == "transformer"


def test_loader_dtype_policy_matches_zimage():
    """Dtype policy is identical to zimage's.

    Neither loader overrides the generic ``_resolve_dtype`` (the effective
    bf16 loading dtype comes from ``driver.resolve_loading_dtype()``, passed
    explicitly as ``torch_dtype``), and no component carries a
    ``dtype_override``.
    """
    import torch

    from app.engine.core.pipeline.loader_base import GenericComponentLoader
    from app.engine.models.families.ovis_image.loader import OvisImageLoader
    from app.engine.models.families.zimage.loader import ZImageLoader

    assert (
        OvisImageLoader._resolve_dtype is GenericComponentLoader._resolve_dtype
    ), "OvisImageLoader must inherit the generic dtype policy (like zimage)"
    assert ZImageLoader._resolve_dtype is GenericComponentLoader._resolve_dtype

    loader = OvisImageLoader(torch.device("cpu"))
    definition = _make_ovis_definition()
    for spec in loader.get_component_manifest(definition):
        assert spec.dtype_override is None, (
            f"{spec.key} must not force a dtype override"
        )
