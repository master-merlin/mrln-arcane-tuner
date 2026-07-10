"""hunyuan_video15 definition-loading tests (parse-only; no weights).

Both 480p YAMLs load as ``ModelDefinition`` with the verified hub facts
(transformer 65→32 channels, 54 layers; VAE 16x/4x with the scalar
1.03682 scaling factor; scheduler static shift 5.0), the CORRECT hub repo ids
(``HunyuanVideo-1.5-Diffusers-480p_{t2v,i2v}``), and ``resolve_capabilities``
marks the family as video. The video contract derives ``model_shift_fixed``
5.0 from the definition.
"""

import pytest

from app.engine.core.archetypes import resolve_capabilities
from app.engine.core.video_contract import resolve_video_profile, validate_video_config
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


HV15_IDS = ["hv15-480p-t2v", "hv15-480p-i2v"]

# The plan's original repo ids (…/HunyuanVideo-1.5-480p_t2v) do NOT exist on
# the hub — the diffusers-format repos carry a ``-Diffusers-`` segment. Pin
# the verified ids so a silent regression to the wrong repo can't land.
HV15_REPOS = {
    "hv15-480p-t2v": (
        "huggingface:hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
    ),
    "hv15-480p-i2v": (
        "huggingface:hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v"
    ),
}


@pytest.mark.parametrize("model_id", HV15_IDS)
def test_definition_loads_with_verified_facts(registry, model_id):
    defn = registry.get_definition(model_id)
    assert defn is not None, f"{model_id} did not load"
    assert defn.family == "hunyuan_video15"
    assert defn.components["repo"].path == HV15_REPOS[model_id]

    arch = defn.architecture_params
    # 65-channel input contract: latents(32) + cond(32) + mask(1).
    assert arch["transformer.in_channels"] == 65
    assert arch["transformer.out_channels"] == 32
    assert arch["transformer.num_layers"] == 54
    assert arch["transformer.text_embed_dim"] == 3584
    assert arch["transformer.text_embed_2_dim"] == 1472
    assert arch["transformer.image_embed_dim"] == 1152
    assert arch["transformer.use_meanflow"] is False
    # VAE: spatial 16x, temporal 4x, scalar scaling factor.
    assert arch["vae.scaling_factor"] == 1.03682
    assert arch["vae.latent_channels"] == 32
    assert arch["video.frame_rule"] == "4n+1"
    assert arch["video.native_fps"] == 24
    assert arch["video.vae_spatial"] == 16
    assert arch["video.vae_temporal"] == 4
    # Scheduler: static shift 5.0 (verified hub scheduler_config.json).
    assert arch["scheduler.flow_shift"] == 5.0
    assert arch["scheduler.use_dynamic_shifting"] is False
    # Dual TE facts.
    assert arch["te.max_length"] == 1000
    assert arch["te.crop_start"] == 108
    assert arch["te2.max_length"] == 256
    assert arch["mode"] in ("t2v", "i2v")


def test_t2v_definition_has_t2v_mode(registry):
    defn = registry.get_definition("hv15-480p-t2v")
    assert defn.architecture_params["mode"] == "t2v"
    assert "image_encoder" not in defn.detected_precision


def test_i2v_definition_has_i2v_mode_and_image_encoder(registry):
    defn = registry.get_definition("hv15-480p-i2v")
    arch = defn.architecture_params
    assert arch["mode"] == "i2v"
    assert arch["image_encoder._class_name"] == "SiglipVisionModel"
    assert arch["image_encoder.num_semantic_tokens"] == 729
    assert "image_encoder" in defn.detected_precision


@pytest.mark.parametrize("model_id", HV15_IDS)
def test_resolve_capabilities_marks_video(registry, model_id):
    defn = registry.get_definition(model_id)
    caps = resolve_capabilities(defn)
    assert caps["archetype"] == "latent_diffusion"
    assert caps["capabilities"]["is_video"] is True
    assert caps["capabilities"]["has_image_encoder"] is True
    assert caps["capabilities"]["supports_train_te"] is False
    fv = caps["field_visibility"]
    assert fv["train_text_encoder"]["supported"] is False
    assert fv["num_frames"]["supported"] is True
    assert fv["train_audio"]["supported"] is False  # no audio modality
    assert fv["cache_latents"]["supported"] is True
    assert fv["unload_text_encoder"]["supported"] is True


@pytest.mark.parametrize("model_id", HV15_IDS)
def test_video_profile_and_derived_model_shift(registry, model_id):
    defn = registry.get_definition(model_id)
    profile = resolve_video_profile(defn)
    assert profile.is_video is True
    assert profile.frame_rule == "4n+1"
    assert profile.native_fps == 24
    assert profile.vae_spatial == 16
    assert profile.vae_temporal == 4
    assert profile.divisibility == 16
    # 4n+1 frame rule behaves.
    assert profile.frame_ok(1) and profile.frame_ok(17) and profile.frame_ok(121)
    assert not profile.frame_ok(16)

    # The video contract folds the static flow shift into the config.
    report = validate_video_config(defn, {"num_frames": 17})
    assert report.ok, report.errors
    assert report.derived["frame_rule"] == "4n+1"
    assert report.derived["model_shift_fixed"] == 5.0


def test_i2v_mode_supported_only_on_i2v_definition(registry):
    t2v = registry.get_definition("hv15-480p-t2v")
    i2v = registry.get_definition("hv15-480p-i2v")
    assert not resolve_video_profile(t2v).supports_i2v()
    assert resolve_video_profile(i2v).supports_i2v()

    bad = validate_video_config(t2v, {"num_frames": 17, "video_mode": "i2v"})
    assert not bad.ok
    ok = validate_video_config(i2v, {"num_frames": 17, "video_mode": "i2v"})
    assert ok.ok, ok.errors


@pytest.mark.parametrize("def_id", HV15_IDS)
def test_definitions_ship_curated_lora_target_list(registry, def_id):
    """Both 480p definitions MUST ship the curated 648-path target list.

    They previously shipped ``lora_targetable_modules: []`` — but an EMPTY
    list is exactly as exposed as a missing one: the enrichment guard is
    ``if not defn.lora_targetable_modules``, so at first real model load
    ``registry.enrich_definition`` fills it with the introspector's
    EXHAUSTIVE Linear catalog (token refiner, embedders, proj_out, ...) and
    the driver prefers a non-empty definition list over
    ``hv15_lora_target_paths`` — silently widening the tested surface
    (dreamlite 2026-07-08 precedent).

    The shipped list equals the driver's own full-path expansion for the
    checkpoint depth: ``hv15_lora_target_paths(54)`` → 54×12 = 648 paths
    (full ``transformer_blocks.{i}.*`` so the token refiner's look-alike
    ``attn.to_q``/``ff.net.*`` modules are never wrapped).
    """
    import torch

    from app.engine.models.families.hunyuan_video15.driver import (
        Hv15Driver,
        hv15_lora_target_paths,
    )

    defn = registry.get_definition(def_id)
    num_layers = defn.architecture_params["transformer.num_layers"]
    expected = set(hv15_lora_target_paths(num_layers))
    assert len(expected) == 648

    shipped = set(defn.lora_targetable_modules or [])
    assert shipped, (
        f"{def_id}: YAML must ship the curated LoRA target list "
        "(an empty [] is auto-filled by enrich_definition)"
    )
    assert shipped == expected, (
        f"{def_id}: shipped list diverges from hv15_lora_target_paths "
        f"(+{len(shipped - expected)} extra, -{len(expected - shipped)} missing)"
    )

    # The driver returns the shipped list verbatim.
    drv = Hv15Driver(defn, torch.device("cpu"))
    assert set(drv.get_lora_targets()) == expected
