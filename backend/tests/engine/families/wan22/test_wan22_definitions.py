"""WAN 2.2 definition-loading tests (parse-only; no weights).

Each YAML loads as a ``ModelDefinition`` with the expected family / mode /
boundary / dual-expert flags, and ``resolve_capabilities`` returns
``is_video=True`` + ``dual_expert=True`` with sane field visibility.
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


WAN22_IDS = ["wan2.2-t2v-a14b", "wan2.2-i2v-a14b"]


@pytest.mark.parametrize("model_id", WAN22_IDS)
def test_definition_loads(registry, model_id):
    defn = registry.get_definition(model_id)
    assert defn is not None, f"{model_id} did not load"
    assert defn.family == "wan22"
    arch = defn.architecture_params
    assert arch["video.frame_rule"] == "4n+1"
    assert arch["video.native_fps"] == 16
    assert arch["te.max_length"] == 512
    assert arch["mode"] in ("t2v", "i2v")
    assert arch["dual_expert"] is True
    # Both experts declared (high = transformer, low = transformer_2).
    assert arch["transformer._class_name"] == "WanTransformer3DModel"
    assert arch["transformer_2._class_name"] == "WanTransformer3DModel"
    assert "unet_low" in defn.detected_precision


def test_t2v_boundary_and_channels(registry):
    defn = registry.get_definition("wan2.2-t2v-a14b")
    arch = defn.architecture_params
    assert arch["mode"] == "t2v"
    assert arch["moe.boundary_ratio"] == 0.875
    assert arch["transformer.in_channels"] == 16


def test_i2v_boundary_channels_and_no_clip(registry):
    defn = registry.get_definition("wan2.2-i2v-a14b")
    arch = defn.architecture_params
    assert arch["mode"] == "i2v"
    assert arch["moe.boundary_ratio"] == 0.9
    # I2V transformer takes the 36-channel concat input (first-frame latent).
    assert arch["transformer.in_channels"] == 36
    # WAN 2.2 I2V has NO CLIP image encoder.
    assert arch["i2v.has_clip_image_encoder"] is False
    assert "image_encoder._class_name" not in arch
    # i2v LoRA targets include the image cross-attn projections.
    assert "attn2.add_k_proj" in defn.lora_targetable_modules
    assert "attn2.add_v_proj" in defn.lora_targetable_modules


@pytest.mark.parametrize("model_id", WAN22_IDS)
def test_resolve_capabilities_marks_video_and_dual_expert(registry, model_id):
    defn = registry.get_definition(model_id)
    caps = resolve_capabilities(defn)
    assert caps["archetype"] == "latent_diffusion"
    assert caps["capabilities"]["is_video"] is True
    assert caps["capabilities"]["dual_expert"] is True
    assert caps["capabilities"]["supports_train_te"] is False
    # WAN 2.2 has no CLIP image encoder.
    assert caps["capabilities"]["has_image_encoder"] is False
    fv = caps["field_visibility"]
    assert fv["train_text_encoder"]["supported"] is False
    assert fv["cache_latents"]["supported"] is True
