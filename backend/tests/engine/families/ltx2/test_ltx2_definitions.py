"""LTX 2.3 definition-loading + capability-surfacing tests."""

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


def test_ltx2_definition_loads():
    registry = ModelRegistry()
    registry.initialize()
    defn = registry.get_definition("ltx2-3-base")
    assert defn is not None
    assert defn.family == "ltx2"
    assert "LTX-2" in defn.components["repo"].path


def test_ltx2_definition_video_architecture_params():
    registry = ModelRegistry()
    registry.initialize()
    defn = registry.get_definition("ltx2-3-base")
    arch = defn.architecture_params
    assert arch["video.frame_rule"] == "8n+1"
    assert arch["video.vae_spatial"] == 32
    assert arch["video.vae_temporal"] == 8
    assert arch["video.divisibility"] == 32
    assert arch["mode"] == "both"
    assert arch["has_audio"] is True


def test_ltx2_definition_has_lora_fallback_targets():
    registry = ModelRegistry()
    registry.initialize()
    defn = registry.get_definition("ltx2-3-base")
    targets = defn.lora_targetable_modules
    # Video stream present.
    assert "attn1.to_q" in targets
    assert "ff.net.0.proj" in targets
    # Audio sub-stream present as a fallback (gated at runtime by the driver).
    assert "audio_attn1.to_q" in targets


def test_ltx2_resolve_capabilities_surfaces_video_and_audio():
    registry = ModelRegistry()
    registry.initialize()
    defn = registry.get_definition("ltx2-3-base")
    caps = resolve_capabilities(defn)["capabilities"]
    assert caps["is_video"] is True
    assert caps["has_audio"] is True
    assert caps["supports_train_te"] is False
