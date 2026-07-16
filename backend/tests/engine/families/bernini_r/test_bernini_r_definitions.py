"""Bernini-R definition-loading tests (parse-only; no weights).

The ``bernini-r-1.3b`` YAML loads as a ``ModelDefinition`` with the v2v sampling
defaults, the Wan2.1-1.3B transformer geometry, the pinned v2v ``flow_shift``,
and the SD3-``mode`` training pins (``mode_scale`` / ``timestep_shift``), and
``resolve_capabilities`` marks it a video family.
"""

import pytest

from app.engine.core.archetypes import resolve_capabilities
from app.engine.models.registry import ModelRegistry

MODEL_ID = "bernini-r-1.3b"


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


def test_definition_loads(registry):
    defn = registry.get_definition(MODEL_ID)
    assert defn is not None, f"{MODEL_ID} did not load"
    assert defn.family == "bernini_r"
    assert defn.control_inputs == 1


def test_sampling_defaults(registry):
    d = registry.get_definition(MODEL_ID).defaults
    assert d["num_inference_steps"] == 40
    assert d["guidance_scale"] == 4.0
    assert d["num_frames"] == 81
    assert d["fps"] == 16
    assert d["height"] == 480
    assert d["width"] == 832


def test_training_mode_pins(registry):
    """The SD3-mode timestep recipe: mode_scale 1.29 + timestep_shift 5.0 must
    load through the registry (else a generic mode_scale default would win).
    """
    d = registry.get_definition(MODEL_ID).defaults
    assert d["mode_scale"] == 1.29
    assert d["timestep_shift"] == 5.0


def test_transformer_geometry(registry):
    arch = registry.get_definition(MODEL_ID).architecture_params
    assert arch["transformer.num_layers"] == 30
    assert arch["transformer.num_attention_heads"] == 12
    assert arch["transformer.attention_head_dim"] == 128
    assert arch["transformer.hidden_size"] == 1536
    assert arch["transformer.ffn_dim"] == 8960
    assert arch["transformer.patch_size"] == [1, 2, 2]
    assert arch["transformer.in_channels"] == 16
    assert arch["transformer.out_channels"] == 16


def test_flow_shift_pinned(registry):
    """scheduler.flow_shift = 5.0 (v2v) and it is enrichment-pinned so a real
    load (repo ships 3.0) cannot clobber it.
    """
    defn = registry.get_definition(MODEL_ID)
    assert defn.architecture_params["scheduler.flow_shift"] == 5.0
    assert "scheduler.flow_shift" in defn.enrich_pinned_keys


def test_video_contract(registry):
    arch = registry.get_definition(MODEL_ID).architecture_params
    assert arch["video.frame_rule"] == "4n+1"
    assert arch["video.native_fps"] == 16
    assert arch["video.vae_spatial"] == 8
    assert arch["video.vae_temporal"] == 4


def test_resolve_capabilities_marks_video(registry):
    defn = registry.get_definition(MODEL_ID)
    caps = resolve_capabilities(defn)
    assert caps["archetype"] == "latent_diffusion"
    assert caps["capabilities"]["is_video"] is True
    assert caps["capabilities"]["supports_train_te"] is False
