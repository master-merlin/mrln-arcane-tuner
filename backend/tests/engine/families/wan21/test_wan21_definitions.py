"""WAN 2.1 definition-loading tests (parse-only; no weights).

Each YAML loads as a ``ModelDefinition`` with the expected family / mode /
architecture params, and ``resolve_capabilities`` returns ``is_video=True`` with
sane field visibility.
"""

import pytest

from app.engine.core.archetypes import resolve_capabilities
from app.engine.models.registry import ModelRegistry


@pytest.fixture(autouse=True)
def clean_registry():
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._paths = {}
    ModelRegistry._discovered = False
    ModelRegistry._definitions_loaded = False
    yield
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._paths = {}
    ModelRegistry._discovered = False
    ModelRegistry._definitions_loaded = False


@pytest.fixture
def registry():
    r = ModelRegistry()
    r.initialize()
    return r


WAN_IDS = [
    "wan2.1-t2v-1.3b",
    "wan2.1-t2v-14b",
    "wan2.1-i2v-14b-720p",
]


@pytest.mark.parametrize("model_id", WAN_IDS)
def test_definition_loads(registry, model_id):
    defn = registry.get_definition(model_id)
    assert defn is not None, f"{model_id} did not load"
    assert defn.family == "wan21"
    arch = defn.architecture_params
    assert arch["video.frame_rule"] == "4n+1"
    assert arch["video.native_fps"] == 16
    assert arch["video.vae_spatial"] == 8
    assert arch["video.vae_temporal"] == 4
    assert arch["te.max_length"] == 512
    assert arch["mode"] in ("t2v", "i2v")


def test_t2v_definitions_have_t2v_mode(registry):
    for mid in ("wan2.1-t2v-1.3b", "wan2.1-t2v-14b"):
        defn = registry.get_definition(mid)
        assert defn.architecture_params["mode"] == "t2v"
        assert defn.architecture_params["transformer.in_channels"] == 16


def test_i2v_definition_has_i2v_mode_and_image_encoder(registry):
    defn = registry.get_definition("wan2.1-i2v-14b-720p")
    arch = defn.architecture_params
    assert arch["mode"] == "i2v"
    # I2V transformer takes the 36-channel concat input.
    assert arch["transformer.in_channels"] == 36
    assert arch["image_encoder._class_name"] == "CLIPVisionModel"
    assert "image_encoder" in defn.detected_precision
    # i2v LoRA targets include the image cross-attn projections.
    assert "attn2.add_k_proj" in defn.lora_targetable_modules
    assert "attn2.add_v_proj" in defn.lora_targetable_modules


@pytest.mark.parametrize("model_id", WAN_IDS)
def test_resolve_capabilities_marks_video(registry, model_id):
    defn = registry.get_definition(model_id)
    caps = resolve_capabilities(defn)
    assert caps["archetype"] == "latent_diffusion"
    assert caps["capabilities"]["is_video"] is True
    assert caps["capabilities"]["supports_train_te"] is False
    # Text-encoder training must be gated off (no trainable TE).
    fv = caps["field_visibility"]
    assert fv["train_text_encoder"]["supported"] is False
    # VAE-gated fields remain available (WAN has a VAE / external TE).
    assert fv["cache_latents"]["supported"] is True
    assert fv["unload_text_encoder"]["supported"] is True
