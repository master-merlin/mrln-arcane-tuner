"""Tests for the prx_pixel family (pixel-space Photoroom PRXPixel, NO VAE).

TDD order (mirrors test_prx_family.py):
  Task 3: family registration + definition loading
  Task 4: loader manifest (NO VAE spec; Qwen3VLTextModel TE)
  Task 5: driver — x0-objective contract (scaled noise, clean-pixel target,
          normalized-t forward, prx_shared LoRA targets on the pixel
          config variant: in_channels=3 + bottleneck img_in +
          resolution_embeds=True)
  Task 6: trainer override trio (encode_text tuple, _update_primary_model
          driver sync, transformer property) + pixel passthrough wiring +
          TE disk-cache layout
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
    in the session (same pattern as test_prx_family.py).
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


def _make_pixel_definition(**kwargs) -> MagicMock:
    """Build a mock prx_pixel ModelDefinition for loader/driver tests."""
    definition = MagicMock(spec=ModelDefinition)
    definition.family = "prx_pixel"
    definition.id = kwargs.get("id", "prx-pixel-test")
    definition.components = {}
    definition.architecture_params = kwargs.get("architecture_params", {})
    definition.lora_targetable_modules = kwargs.get("lora_targetable_modules", [])
    definition.defaults = kwargs.get("defaults", {})
    return definition


# ── Task 3: Family Registration ──────────────────────────────────────────────


def test_family_registered():
    """prx_pixel family must register with the pixel_transformer archetype."""
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry.discover_families()

    fam = ModelRegistry._families.get("prx_pixel")
    assert fam is not None, "prx_pixel family not registered"
    assert fam.archetype == "pixel_transformer", (
        f"expected archetype='pixel_transformer', got {fam.archetype!r}"
    )


def test_definition_loaded():
    """prx-pixel-t2i definition must load from its YAML file."""
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._definitions_loaded = False
    ModelRegistry.initialize()

    fam_defs = {
        d.id: d
        for d in ModelRegistry._definitions.values()
        if d.family == "prx_pixel"
    }
    assert "prx-pixel-t2i" in fam_defs, (
        f"missing prx-pixel-t2i definition; found: {set(fam_defs)}"
    )

    base = fam_defs["prx-pixel-t2i"]
    # Canonical checkpoint repo (verified model_index.json)
    assert base.components["repo"].path == "huggingface:Photoroom/prxpixel-t2i", (
        f"wrong repo path: {base.components['repo'].path!r}"
    )
    # Standard T2I — no paired control inputs
    assert base.control_inputs == 0
    # Native 1024 default resolution (default_sample_size=1024)
    assert base.defaults.get("resolution") == 1024
    # Pipeline __call__ defaults
    assert base.defaults.get("guidance_scale") == 4.0
    assert base.defaults.get("num_inference_steps") == 28

    # Verified transformer config facts (checkpoint transformer/config.json,
    # fetched 2026-07-08 — the PIXEL variant differs from class defaults).
    arch = base.architecture_params
    assert arch.get("transformer.in_channels") == 3, "pixel space: RGB in"
    assert arch.get("transformer.patch_size") == 16
    assert arch.get("transformer.hidden_size") == 3584
    assert arch.get("transformer.depth") == 24
    assert arch.get("transformer.num_heads") == 28
    assert arch.get("transformer.context_in_dim") == 2048
    assert arch.get("transformer.bottleneck_size") == 768
    assert arch.get("transformer.resolution_embeds") is True
    assert arch.get("transformer.time_factor") == 1000.0
    # Pipeline registered config (model_index.json)
    assert arch.get("pipeline.noise_scale") == 2.0
    assert arch.get("pipeline.prompt_max_tokens") == 256
    # NO VAE — a vae.* section would resurrect latent-space assumptions.
    assert not any(k.startswith("vae.") for k in arch), (
        "prx_pixel is pixel-space — no vae.* architecture params allowed"
    )
    # Scheduler facts (checkpoint scheduler_config.json): static shift 3.0.
    assert arch.get("scheduler.num_train_timesteps") == 1000
    assert arch.get("scheduler.shift") == 3.0
    # TE facts (checkpoint text_encoder/config.json — Qwen3VLTextModel)
    assert arch.get("te.type") == "qwen3_vl_text"
    assert arch.get("te.hidden_size") == 2048
    assert arch.get("te.num_hidden_layers") == 28
    # prompt_max_tokens drives tokenization, NOT tokenizer.model_max_length
    # (the Qwen tokenizer's own model_max_length is far larger than 256).
    assert arch.get("te.max_length") == 256


# ── Task 4: Loader Manifest ──────────────────────────────────────────────────


def test_manifest_components_no_vae():
    """PRXPixelLoader manifest declares tokenizer + TE + transformer and
    NOTHING else — a vae spec would make GenericComponentLoader try to
    download/load a VAE that does not exist in the checkpoint."""
    import torch

    from app.engine.models.families.prx_pixel.loader import PRXPixelLoader

    loader = PRXPixelLoader(torch.device("cpu"))
    definition = _make_pixel_definition()
    specs = loader.get_component_manifest(definition)

    keys = {s.key for s in specs}
    assert keys == {"tokenizer", "text_encoder", "unet"}, (
        f"manifest must be exactly tokenizer/text_encoder/unet, got {keys}"
    )

    spec_map = {s.key: s for s in specs}

    # Tokenizer: AutoTokenizer (Qwen2TokenizerFast resolves via tokenizer.json
    # — mirrors what PRXPixelPipeline.from_pretrained materializes).
    assert "AutoTokenizer" in spec_map["tokenizer"].hf_class, (
        f"tokenizer hf_class wrong: {spec_map['tokenizer'].hf_class}"
    )
    assert spec_map["tokenizer"].is_torch_model is False

    # Text encoder: the checkpoint's model_index.json declares
    # ["transformers", "Qwen3VLTextModel"] — top-level transformers export.
    assert spec_map["text_encoder"].hf_class == "transformers.Qwen3VLTextModel", (
        f"text_encoder hf_class wrong: {spec_map['text_encoder'].hf_class}"
    )
    assert spec_map["text_encoder"].subfolder == "text_encoder"

    # Transformer mapped to "unet" (repo convention), diffusers-native class
    assert "PRXTransformer2DModel" in spec_map["unet"].hf_class, (
        f"unet hf_class wrong: {spec_map['unet'].hf_class}"
    )
    assert spec_map["unet"].subfolder == "transformer"


def test_manifest_classes_are_importable():
    """Every hf_class in the manifest resolves through the generic loader's
    importlib seam (Qwen3VLTextModel must be a real top-level export)."""
    import torch

    from app.engine.core.pipeline.loader_base import GenericComponentLoader
    from app.engine.models.families.prx_pixel.loader import PRXPixelLoader

    loader = PRXPixelLoader(torch.device("cpu"))
    for spec in loader.get_component_manifest(_make_pixel_definition()):
        cls = GenericComponentLoader._import_class(spec.hf_class)
        assert cls is not None, f"{spec.hf_class} not importable"


def test_loader_dtype_policy_is_generic():
    """Dtype policy inherits the generic path (bf16 via driver), no per-spec
    overrides — identical policy to the latent prx sibling."""
    import torch

    from app.engine.core.pipeline.loader_base import GenericComponentLoader
    from app.engine.models.families.prx_pixel.loader import PRXPixelLoader

    assert PRXPixelLoader._resolve_dtype is GenericComponentLoader._resolve_dtype, (
        "PRXPixelLoader must inherit the generic dtype policy"
    )

    loader = PRXPixelLoader(torch.device("cpu"))
    for spec in loader.get_component_manifest(_make_pixel_definition()):
        assert spec.dtype_override is None, (
            f"{spec.key} must not force a dtype override"
        )
