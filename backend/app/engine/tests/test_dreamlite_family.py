"""Tests for the dreamlite family (ByteDance DreamLite, diffusers-0.39-native).

TDD order (mirrors test_ovis_image_family.py / test_krea2_family.py):
  Task 1: family registration + BOTH definitions (base + mobile) loading
  Task 2: loader manifest (component specs + revision plumbing + dtype policy)
  Task 4: driver (LoRA targets vs a tiny DreamLiteUNetModel, forward_pass
          UNET signature contract, raw-timestep contract, compute_target)
  Task 5: trainer override trio (encode_text tuple, _update_primary_model
          driver sync, transformer property) + TE disk-cache layout

Verified checkpoint facts (unet/config.json, revision="diffusers"):
  block_out_channels (256, 512, 896); attention_head_dim (4, 8, 14) →
  head_dim 64 at EVERY level; num_kv_heads=1 (MQA) → to_k/to_v
  out_features = 64; transformer_layers_per_block (1, 2, 4);
  cross_attention_dim 2304; encoder_hid_dim_type "text_proj_rms" (2048→2304);
  addition_embed_type "time" (time_ids [w, h]); ff_mult 3; qk_norm rms_norm.
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


def _make_dreamlite_definition(**kwargs) -> MagicMock:
    """Build a mock DreamLite ModelDefinition for loader/driver tests."""
    definition = MagicMock(spec=ModelDefinition)
    definition.family = "dreamlite"
    definition.id = kwargs.get("id", "dreamlite-test")
    definition.components = {}
    definition.architecture_params = kwargs.get("architecture_params", {})
    definition.lora_targetable_modules = kwargs.get(
        "lora_targetable_modules", [],
    )
    return definition


# ── Task 1: Family Registration ──────────────────────────────────────────────


def test_family_registered():
    """dreamlite family must appear in ModelRegistry with the correct archetype."""
    from app.engine.models.registry import ModelRegistry

    # Reset discovery state so this test is hermetic
    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry.discover_families()

    fam = ModelRegistry._families.get("dreamlite")
    assert fam is not None, "dreamlite family not registered"
    assert fam.archetype == "latent_diffusion", (
        f"expected archetype='latent_diffusion', got {fam.archetype!r}"
    )


def test_both_definitions_loaded():
    """dreamlite-base AND dreamlite-mobile definitions must load from YAML."""
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
        if d.family == "dreamlite"
    }
    assert "dreamlite-base" in fam_defs, (
        f"missing dreamlite-base definition; found: {set(fam_defs)}"
    )
    assert "dreamlite-mobile" in fam_defs, (
        f"missing dreamlite-mobile definition; found: {set(fam_defs)}"
    )

    base = fam_defs["dreamlite-base"]
    mobile = fam_defs["dreamlite-mobile"]

    # Canonical checkpoints — BOTH need the "diffusers" revision (@revision
    # suffix understood by ModelPathResolver since the dreamlite family).
    assert base.components["repo"].path == (
        "huggingface:carlofkl/DreamLite-base@diffusers"
    ), f"wrong base repo path: {base.components['repo'].path!r}"
    assert mobile.components["repo"].path == (
        "huggingface:carlofkl/DreamLite-mobile@diffusers"
    ), f"wrong mobile repo path: {mobile.components['repo'].path!r}"

    # Standard T2I — the diptych "[Edit]" mode is out of scope.
    assert base.control_inputs == 0
    assert mobile.control_inputs == 0

    # CFG split: base is the CFG (30-step) checkpoint, mobile is the
    # CFG-distilled (4-step) checkpoint — the krea2 Raw/Turbo convention.
    assert base.defaults.get("is_distilled") is False
    assert mobile.defaults.get("is_distilled") is True
    assert base.defaults.get("guidance_scale") == 3.5
    assert mobile.defaults.get("guidance_scale") == 0
    assert mobile.defaults.get("num_inference_steps") == 4

    # Verified unet config facts (checkpoint unet/config.json).
    for defn in (base, mobile):
        arch = defn.architecture_params
        assert arch.get("unet._class_name") == "DreamLiteUNetModel"
        assert arch.get("unet.block_out_channels") == [256, 512, 896]
        assert arch.get("unet.attention_head_dim") == [4, 8, 14]
        assert arch.get("unet.transformer_layers_per_block") == [1, 2, 4]
        assert arch.get("unet.layers_per_block") == 2
        assert arch.get("unet.cross_attention_dim") == 2304
        assert arch.get("unet.encoder_hid_dim") == 2048
        assert arch.get("unet.encoder_hid_dim_type") == "text_proj_rms"
        assert arch.get("unet.addition_embed_type") == "time"
        assert arch.get("unet.num_kv_heads") == 1
        assert arch.get("unet.qk_norm") == "rms_norm"
        assert arch.get("unet.ff_mult") == 3
        assert arch.get("unet.in_channels") == 4
        assert arch.get("unet.sample_size") == 128
        # VAE: AutoencoderTiny (taesdxl) — 4 latent channels, 8× spatial.
        assert arch.get("vae._class_name") == "AutoencoderTiny"
        assert arch.get("vae.latent_channels") == 4
        assert arch.get("vae.vae_scale_factor") == 8
        assert arch.get("vae.scaling_factor") == 1.0
        assert arch.get("vae.shift_factor") == 0.0
        # Scheduler facts (checkpoint scheduler_config.json).
        assert arch.get("scheduler.use_dynamic_shifting") is True
        assert arch.get("scheduler.base_shift") == 0.5
        assert arch.get("scheduler.max_shift") == 1.15
        assert arch.get("scheduler.base_image_seq_len") == 256
        assert arch.get("scheduler.max_image_seq_len") == 4096
        # DreamLitePipeline prompt-template contract.
        assert arch.get("te.max_sequence_length") == 200
        assert arch.get("te.drop_idx") == 34


# ── Task 2: Loader Manifest ──────────────────────────────────────────────────


def test_manifest_components():
    """DreamLiteLoader manifest declares all four diffusers-native components.

    The checkpoint stores its primary model under ``unet/`` (it IS a U-Net —
    ``model_index.json`` names the component "unet"), unlike the DiT
    families' ``transformer/`` subfolder. All four components are
    diffusers-0.39 / transformers-4.57-native: the TE config is saved by
    transformers 4.57.3 (verified), so no krea2-style rope translation is
    needed and the plain manifest path suffices.
    """
    import torch  # noqa: PLC0415

    from app.engine.models.families.dreamlite.loader import DreamLiteLoader

    loader = DreamLiteLoader(torch.device("cpu"))
    definition = _make_dreamlite_definition()
    specs = loader.get_component_manifest(definition)

    keys = {s.key for s in specs}
    assert {"tokenizer", "text_encoder", "vae", "unet"} <= keys, (
        f"missing required manifest keys; got {keys}"
    )

    spec_map = {s.key: s for s in specs}

    # Tokenizer: AutoTokenizer (fast Qwen2TokenizerFast via tokenizer.json)
    assert "AutoTokenizer" in spec_map["tokenizer"].hf_class
    assert spec_map["tokenizer"].is_torch_model is False

    # Text encoder: Qwen3-VL base model (no LM head; hidden_states[-1] is
    # identical to the pipeline's Qwen3VLForConditionalGeneration tap).
    assert spec_map["text_encoder"].hf_class == "transformers.Qwen3VLModel"
    assert spec_map["text_encoder"].subfolder == "text_encoder"

    # VAE: AutoencoderTiny (taesdxl — NO latent_dist, encode returns .latents)
    assert spec_map["vae"].hf_class == "diffusers.AutoencoderTiny"
    assert spec_map["vae"].subfolder == "vae"

    # Primary model: DreamLiteUNetModel under the checkpoint's unet/ subfolder
    assert spec_map["unet"].hf_class == "diffusers.DreamLiteUNetModel"
    assert spec_map["unet"].subfolder == "unet"


def test_loader_dtype_policy_is_generic():
    """No dtype overrides — bf16 comes from driver.resolve_loading_dtype."""
    import torch  # noqa: PLC0415

    from app.engine.core.pipeline.loader_base import GenericComponentLoader
    from app.engine.models.families.dreamlite.loader import DreamLiteLoader

    assert (
        DreamLiteLoader._resolve_dtype is GenericComponentLoader._resolve_dtype
    ), "DreamLiteLoader must inherit the generic dtype policy"

    loader = DreamLiteLoader(torch.device("cpu"))
    definition = _make_dreamlite_definition()
    for spec in loader.get_component_manifest(definition):
        assert spec.dtype_override is None, (
            f"{spec.key} must not force a dtype override"
        )


def test_base_and_mobile_architecture_params_identical():
    """Portability requirement: base and mobile share ONE architecture.

    Verified against the hub: unet/vae/text_encoder/scheduler configs are
    byte-identical between DreamLite-base and DreamLite-mobile (only the
    weights differ). The YAMLs must therefore carry identical
    ``architecture_params`` so a LoRA trained on one definition loads onto
    the other.
    """
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._definitions_loaded = False
    ModelRegistry.initialize()

    fam_defs = {
        d.id: d
        for d in ModelRegistry._definitions.values()
        if d.family == "dreamlite"
    }
    base = fam_defs["dreamlite-base"]
    mobile = fam_defs["dreamlite-mobile"]
    assert base.architecture_params == mobile.architecture_params, (
        "architecture_params must be byte-identical across base and mobile"
    )
