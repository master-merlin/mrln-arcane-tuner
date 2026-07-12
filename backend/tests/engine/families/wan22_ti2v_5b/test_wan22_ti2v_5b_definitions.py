"""WAN 2.2 TI2V-5B definition-loading + video-contract profile tests.

The YAML loads as a ``ModelDefinition`` with the recon'd architecture params
(30-layer/3072-hidden dense transformer, 48ch in/out, the NEW high-compression
VAE: z_dim 48 / spatial 16 / temporal 4), ``mode: both`` (a single checkpoint
serves T2V + I2V, chosen per-run via ``video_mode`` — the ltx2 precedent), and
``resolve_capabilities``/``resolve_video_profile`` agree.
"""

from __future__ import annotations

import pytest

from app.engine.core.archetypes import resolve_capabilities
from app.engine.core.video_contract import resolve_video_profile
from app.engine.models.registry import ModelRegistry, registry


@pytest.fixture(scope="module", autouse=True)
def _loaded_registry():
    registry.discover_families()
    registry.load_definitions("app/engine/models/definitions")
    return registry


MODEL_ID = "wan2.2-ti2v-5b"


def _defn():
    d = ModelRegistry.get_definition(MODEL_ID)
    assert d is not None, f"{MODEL_ID} did not load"
    return d


def test_definition_loads_with_expected_arch():
    defn = _defn()
    assert defn.family == "wan22_ti2v_5b"
    arch = defn.architecture_params
    assert arch["mode"] == "both"
    assert arch["dual_expert"] is False
    assert arch["transformer._class_name"] == "WanTransformer3DModel"
    # Dense 5B: 48-channel in/out (== z_dim, no 36-channel mask+cond concat).
    assert arch["transformer.in_channels"] == 48
    assert arch["transformer.out_channels"] == 48
    assert arch["transformer.num_layers"] == 30
    assert arch["transformer.num_attention_heads"] == 24
    assert arch["transformer.attention_head_dim"] == 128
    assert arch["transformer.image_dim"] is None
    # No second expert / MoE fields at all (unlike wan22 A14B).
    assert "transformer_2._class_name" not in arch
    assert "moe.boundary_ratio" not in arch


def test_definition_has_no_unet_low_precision_entry():
    """Dense load: unlike wan22 A14B, no second-expert precision key."""
    defn = _defn()
    assert "unet_low" not in defn.detected_precision
    assert defn.detected_precision["unet"] == "torch.bfloat16"
    assert defn.detected_precision["vae"] == "torch.float32"


def test_vae_uses_the_new_high_compression_config():
    arch = _defn().architecture_params
    assert arch["vae.z_dim"] == 48
    assert arch["video.vae_spatial"] == 16  # NOT wan21/wan22's 8
    assert arch["video.vae_temporal"] == 4  # unchanged
    assert arch["video.frame_rule"] == "4n+1"
    assert arch["video.native_fps"] == 24  # NOT wan21/wan22's 16
    assert len(arch["vae.latents_mean"]) == 48
    assert len(arch["vae.latents_std"]) == 48


def test_lora_targets_have_no_image_cross_attn():
    """No CLIP image encoder, no added_kv_proj_dim → no add_k_proj/add_v_proj."""
    defn = _defn()
    assert "attn2.add_k_proj" not in defn.lora_targetable_modules
    assert "attn2.add_v_proj" not in defn.lora_targetable_modules
    assert "attn1.to_q" in defn.lora_targetable_modules
    assert "ffn.net.0.proj" in defn.lora_targetable_modules


def test_resolve_capabilities_marks_video_not_dual_expert():
    caps = resolve_capabilities(_defn())
    assert caps["archetype"] == "latent_diffusion"
    assert caps["capabilities"]["is_video"] is True
    assert caps["capabilities"]["dual_expert"] is False
    assert caps["capabilities"]["has_image_encoder"] is False
    fv = caps["field_visibility"]
    # MoE-only fields hidden — the whole point of NOT setting dual_expert=True.
    assert fv["expert_mode"]["supported"] is False
    assert fv["expert_swap_mode"]["supported"] is False
    assert fv["expert_switch_interval"]["supported"] is False
    # Video fields ARE visible (is_video=True).
    assert fv["video_mode"]["supported"] is True
    assert fv["num_frames"]["supported"] is True
    assert fv["first_frame_conditioning_probability"]["supported"] is True


def test_video_profile_both_mode_supports_i2v():
    p = resolve_video_profile(_defn())
    assert p.is_video is True
    assert p.dual_expert is False
    assert p.has_audio is False
    assert p.frame_rule == "4n+1"
    assert p.native_fps == 24.0
    assert p.vae_spatial == 16 and p.vae_temporal == 4
    assert p.mode == "both"
    assert p.supports_i2v() is True
