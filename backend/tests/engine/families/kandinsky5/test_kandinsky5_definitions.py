"""Kandinsky 5.0 definition YAML tests.

Both definitions must load, declare the verified checkpoint facts (16 latent
channels, HunyuanVideo VAE, 4n+1 frame rule at 24 fps, divisibility 16) and
project into a correct :class:`VideoProfile` (incl. the flow-shift 5.0 that
``model_shift`` timestep sampling consumes via ``model_shift_fixed``).
"""

import pytest

from app.engine.core.video_contract import (
    resolve_video_profile,
    validate_video_config,
)
from app.engine.models.registry import ModelRegistry

T2V_ID = "k5-t2v-lite-sft-5s"
I2V_ID = "k5-i2v-pro-sft-5s"


@pytest.fixture()
def registry():
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._paths = {}
    ModelRegistry._discovered = False
    ModelRegistry._definitions_loaded = False
    r = ModelRegistry()
    r.discover_families()
    r.load_definitions("app/engine/models/definitions")
    yield r
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._paths = {}
    ModelRegistry._discovered = False
    ModelRegistry._definitions_loaded = False


def test_both_definitions_load(registry):
    for def_id in (T2V_ID, I2V_ID):
        defn = registry.get_definition(def_id)
        assert defn is not None, f"{def_id} not registered"
        assert defn.family == "kandinsky5"


def test_t2v_definition_facts(registry):
    defn = registry.get_definition(T2V_ID)
    arch = defn.architecture_params
    assert arch["mode"] == "t2v"
    assert arch["transformer.in_visual_dim"] == 16
    assert arch["transformer.model_dim"] == 1792
    assert arch["transformer.num_visual_blocks"] == 32
    # Verified on the hub: even the T2V Lite sft checkpoint is visual_cond=True
    # (the T2V pipeline concats a ZERO cond + mask for it).
    assert arch["transformer.visual_cond"] is True
    assert arch["vae._class_name"] == "AutoencoderKLHunyuanVideo"
    assert arch["vae.latent_channels"] == 16
    assert arch["vae.scaling_factor"] == pytest.approx(0.476986)
    assert "Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers" in defn.components["repo"].path


def test_i2v_definition_facts(registry):
    defn = registry.get_definition(I2V_ID)
    arch = defn.architecture_params
    assert arch["mode"] == "i2v"
    assert arch["transformer.model_dim"] == 4096
    assert arch["transformer.num_visual_blocks"] == 60
    assert arch["transformer.visual_cond"] is True
    assert arch["transformer.attention_type"] == "nabla"
    assert "Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers" in defn.components["repo"].path


@pytest.mark.parametrize("def_id", [T2V_ID, I2V_ID])
def test_video_profile(registry, def_id):
    profile = resolve_video_profile(registry.get_definition(def_id))
    assert profile.is_video is True
    assert profile.frame_rule == "4n+1"
    assert profile.native_fps == 24.0
    assert profile.vae_spatial == 8
    assert profile.vae_temporal == 4
    assert profile.divisibility == 16
    assert profile.has_audio is False
    assert profile.has_image_encoder is False
    # 4n+1 frame rule: the 121-frame default is valid, 120 is not.
    assert profile.frame_ok(121)
    assert profile.frame_ok(17)
    assert not profile.frame_ok(120)


def test_i2v_supported_only_on_i2v_definition(registry):
    assert resolve_video_profile(registry.get_definition(I2V_ID)).supports_i2v()
    assert not resolve_video_profile(registry.get_definition(T2V_ID)).supports_i2v()


def test_model_shift_fixed_derived_from_flow_shift(registry):
    """validate_video_config folds shift 5.0 into model_shift_fixed so
    ``timestep_sampling: model_shift`` reproduces the inference schedule."""
    report = validate_video_config(
        registry.get_definition(T2V_ID), {"num_frames": 121}
    )
    assert report.ok, report.errors
    assert report.derived.get("model_shift_fixed") == 5.0
    assert report.derived.get("frame_rule") == "4n+1"
    assert report.derived.get("video_divisibility") == 16


def test_bad_frame_count_rejected(registry):
    report = validate_video_config(
        registry.get_definition(T2V_ID), {"num_frames": 120}
    )
    assert not report.ok
    assert any("frame rule" in e for e in report.errors)


def test_i2v_video_mode_rejected_on_t2v_definition(registry):
    report = validate_video_config(
        registry.get_definition(T2V_ID), {"num_frames": 121, "video_mode": "i2v"}
    )
    assert not report.ok


@pytest.mark.parametrize(("def_id", "blocks"), [(T2V_ID, 32), (I2V_ID, 60)])
def test_definitions_ship_curated_lora_target_list(registry, def_id, blocks):
    """Both definitions MUST ship the curated fully-indexed target list.

    dreamlite precedent (2026-07-08 GPU-UAT crash): a definition without a
    non-empty ``lora_targetable_modules`` gets the field auto-filled at first
    real model load by ``registry.enrich_definition`` with the introspector's
    EXHAUSTIVE Linear catalog (text_transformer_blocks, time embedder, ...) —
    and for kandinsky5 those harvested full paths would then be re-expanded
    per block into paths matching NOTHING, so PEFT would wrap zero modules.

    The shipped list equals the driver's own expansion of
    ``K5_LORA_TARGET_SUFFIXES`` over the checkpoint's visual block count
    (Lite 32×10=320, Pro 60×10=600 — VISUAL blocks only), and the driver
    returns it verbatim (no re-expansion).
    """
    import torch

    from app.engine.models.families.kandinsky5.driver import (
        K5_LORA_TARGET_SUFFIXES,
        Kandinsky5Driver,
    )

    defn = registry.get_definition(def_id)
    expected = {
        f"visual_transformer_blocks.{i}.{suffix}"
        for i in range(blocks)
        for suffix in K5_LORA_TARGET_SUFFIXES
    }
    shipped = set(defn.lora_targetable_modules or [])
    assert shipped, f"{def_id}: YAML must ship the curated LoRA target list"
    assert shipped == expected, (
        f"{def_id}: shipped list diverges from the driver expansion "
        f"(+{len(shipped - expected)} extra, -{len(expected - shipped)} missing)"
    )

    drv = Kandinsky5Driver(defn, torch.device("cpu"))
    targets = drv.get_lora_targets()
    assert set(targets) == expected
    assert len(targets) == blocks * len(K5_LORA_TARGET_SUFFIXES)
