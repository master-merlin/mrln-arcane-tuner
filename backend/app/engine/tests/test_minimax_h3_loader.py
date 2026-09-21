"""minimax_h3 component manifest contract.

Asserts the manifest itself — no weights, no network. The point is that the
five components resolve to the right subfolders and the right CLASSES, since
a wrong class path fails only at download time after ~100 GB of traffic.
"""

from __future__ import annotations

from app.engine.models.families.minimax_h3.loader import MiniMaxH3Loader
from app.engine.models.registry import ModelRegistry

def _manifest(def_id: str):
    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    definition = ModelRegistry._definitions[def_id]
    loader = MiniMaxH3Loader.__new__(MiniMaxH3Loader)
    return {spec.key: spec for spec in loader.get_component_manifest(definition)}


def test_manifest_declares_all_five_components():
    specs = _manifest("minimax-h3-t2va")
    assert set(specs) == {"tokenizer", "text_encoder", "vae", "audio_vae", "transformer"}


def test_transformer_uses_the_installed_diffusers_class():
    # diffusers 0.40.0 (the requirements.txt floor) ships H3 natively; the
    # family's vendored fork was retired in plan row 1.8.
    spec = _manifest("minimax-h3-t2va")["transformer"]
    assert spec.hf_class == (
        "diffusers.models.transformers.transformer_minimax_h3.MiniMaxH3Transformer3DModel"
    )
    assert spec.subfolder == "transformer"


def test_ref2va_transformer_points_at_the_ref_checkpoint():
    assert _manifest("minimax-h3-ref2va")["transformer"].subfolder == "transformer_ref"


def test_both_vaes_are_diffusers_classes_and_distinct():
    specs = _manifest("minimax-h3-t2va")
    assert specs["vae"].hf_class == "diffusers.AutoencoderKLMiniMaxH3"
    assert specs["audio_vae"].hf_class == "diffusers.AutoencoderKLMiniMaxH3Audio"
    assert specs["vae"].subfolder == "vae"
    assert specs["audio_vae"].subfolder == "audio_vae"
    assert specs["vae"].hf_class != specs["audio_vae"].hf_class


def test_tokenizer_is_not_moved_to_device():
    # is_torch_model=False — a processor/tokenizer has no .to(device).eval().
    assert _manifest("minimax-h3-t2va")["tokenizer"].is_torch_model is False
