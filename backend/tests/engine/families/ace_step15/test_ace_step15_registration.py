"""ACE-Step 1.5 family registration + loader manifest sanity."""

from __future__ import annotations

from app.engine.models.families.ace_step15.family import AceStep15Family
from app.engine.models.families.ace_step15.loader import AceStep15Loader
from app.engine.models.registry import ModelRegistry


def test_family_registers_under_ace_step15():
    ModelRegistry.initialize()
    assert "ace_step15" in ModelRegistry._families
    assert ModelRegistry._families["ace_step15"] is AceStep15Family


def test_family_archetype_and_capability_overrides():
    assert AceStep15Family.family_name == "ace_step15"
    assert AceStep15Family.archetype == "latent_diffusion"
    assert AceStep15Family.capability_overrides == {
        "is_audio_family": True,
        "supports_spatial_resolution": False,
    }


def test_get_trainer_class_resolves():
    from app.engine.models.families.ace_step15.trainer import AceStep15Trainer

    dummy = object.__new__(AceStep15Family)
    assert dummy.get_trainer_class() is AceStep15Trainer


def test_loader_manifest_keys_and_classes():
    loader = object.__new__(AceStep15Loader)
    manifest = loader.get_component_manifest(definition=None)
    keys = {spec.key for spec in manifest}
    assert keys == {"tokenizer", "text_encoder", "vae", "condition_encoder", "unet"}

    by_key = {spec.key: spec for spec in manifest}
    assert by_key["tokenizer"].hf_class == "transformers.AutoTokenizer"
    assert by_key["tokenizer"].is_torch_model is False
    assert by_key["text_encoder"].hf_class == "transformers.Qwen3Model"
    assert by_key["vae"].hf_class == "diffusers.AutoencoderOobleck"
    assert (
        by_key["condition_encoder"].hf_class
        == "diffusers.pipelines.ace_step.modeling_ace_step.AceStepConditionEncoder"
    )
    assert by_key["unet"].hf_class == "diffusers.AceStepTransformer1DModel"
    assert by_key["unet"].subfolder == "transformer"


def test_loader_hf_classes_are_importable():
    """Every ``hf_class`` string in the manifest must resolve via
    ``GenericComponentLoader._import_class`` — the exact mechanism
    ``_load_single_spec`` uses at real load time."""
    from app.engine.core.pipeline.loader_base import GenericComponentLoader

    loader = object.__new__(AceStep15Loader)
    manifest = loader.get_component_manifest(definition=None)
    for spec in manifest:
        cls = GenericComponentLoader._import_class(spec.hf_class)
        assert cls is not None, spec.hf_class
