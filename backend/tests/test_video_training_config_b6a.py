"""Phase B6a: video training config fields + capability gating + VRAM frames.

Covers three slices of the video-LoRA program:

1. ``BaseTrainingConfig`` gains its optional VIDEO fields with the stated
   defaults, and an image-family config (no video fields set) still validates.
2. ``resolve_capabilities`` field-visibility gating shows/hides the video
   fields per family capability (image hides all; wan21 shows frames only;
   ltx2 shows frames + audio; wan22 shows frames + expert routing).
3. The VRAM estimator's activation term scales up with ``num_frames`` for a
   video definition, while an image definition's estimate is unchanged.
"""

from __future__ import annotations

import math

import pytest

from app.engine.core.archetypes import resolve_capabilities
from app.engine.models.base import BaseTrainingConfig
from app.engine.models.registry import registry
from app.engine.utils.vram_estimator import VRAMEstimator


@pytest.fixture(scope="module", autouse=True)
def _loaded_registry():
    registry.discover_families()
    registry.load_definitions("app/engine/models/definitions")
    return registry


# Real registry definition IDs (verified from the YAML files).
_IMAGE_DEF = "sdxl_base_1.0"
_WAN21_DEF = "wan2.1-t2v-1.3b"
_LTX2_DEF = "ltx2-3-base"
_WAN22_DEF = "wan2.2-t2v-a14b"


# ── 1. Config fields + defaults ──────────────────────────────────────────


def _minimal_config(**overrides) -> BaseTrainingConfig:
    base = {"datasets": [{"dataset_name": "demo"}]}
    base.update(overrides)
    return BaseTrainingConfig.model_validate(base)


def test_video_fields_have_stated_defaults():
    cfg = _minimal_config()
    assert cfg.num_frames == 81
    assert cfg.target_fps == 0
    assert cfg.video_mode == "t2v"
    assert cfg.i2v_image_dropout == pytest.approx(0.1)
    assert cfg.train_audio is False
    assert cfg.audio_loss_weight == pytest.approx(1.0)
    assert cfg.expert_swap_mode == "auto"
    assert cfg.expert_switch_interval == 1


def test_all_video_fields_exist():
    expected = {
        "num_frames",
        "target_fps",
        "video_mode",
        "i2v_image_dropout",
        "train_audio",
        "audio_loss_weight",
        "expert_swap_mode",
        "expert_switch_interval",
    }
    assert expected <= set(BaseTrainingConfig.model_fields)


def test_video_fields_declare_video_group():
    fields = BaseTrainingConfig.model_fields
    for name in (
        "num_frames",
        "target_fps",
        "video_mode",
        "i2v_image_dropout",
        "train_audio",
        "audio_loss_weight",
        "expert_swap_mode",
        "expert_switch_interval",
    ):
        extra = fields[name].json_schema_extra or {}
        assert extra.get("group") == "VIDEO", name


def test_image_config_without_video_fields_still_validates():
    # An image-family config that never touches a video field must validate
    # and inherit the safe video defaults (image families never render them).
    cfg = _minimal_config(model_family="sdxl", definition_id=_IMAGE_DEF)
    assert cfg.model_family == "sdxl"
    assert cfg.num_frames == 81  # default present, harmless for images
    assert cfg.train_audio is False


def test_video_fields_serialize_into_schema_with_group():
    # The schema the frontend renders must carry the VIDEO group so the
    # schema-driven Training UI can place the fields.
    schema = BaseTrainingConfig.model_json_schema()
    props = schema["properties"]
    assert props["num_frames"]["group"] == "VIDEO"
    assert props["train_audio"]["group"] == "VIDEO"
    assert props["expert_swap_mode"]["group"] == "VIDEO"


# ── 2. Field-visibility gating per family ────────────────────────────────


def _visibility(def_id: str) -> dict:
    defn = registry.get_definition(def_id)
    assert defn is not None, f"definition {def_id} not in registry"
    return resolve_capabilities(defn)["field_visibility"]


def _shown(vis: dict, field: str) -> bool:
    return vis[field]["supported"]


def test_image_family_hides_all_video_fields():
    vis = _visibility(_IMAGE_DEF)
    for field in (
        "num_frames",
        "target_fps",
        "video_mode",
        "i2v_image_dropout",
        "train_audio",
        "audio_loss_weight",
        "expert_swap_mode",
        "expert_switch_interval",
    ):
        assert _shown(vis, field) is False, field


def test_wan21_shows_frames_hides_audio_and_experts():
    vis = _visibility(_WAN21_DEF)
    assert _shown(vis, "num_frames") is True
    assert _shown(vis, "target_fps") is True
    assert _shown(vis, "video_mode") is True
    # WAN 2.1 has no audio modality and is single-transformer.
    assert _shown(vis, "train_audio") is False
    assert _shown(vis, "audio_loss_weight") is False
    assert _shown(vis, "expert_swap_mode") is False
    assert _shown(vis, "expert_switch_interval") is False


def test_ltx2_shows_frames_and_audio_hides_experts():
    vis = _visibility(_LTX2_DEF)
    assert _shown(vis, "num_frames") is True
    assert _shown(vis, "train_audio") is True
    assert _shown(vis, "audio_loss_weight") is True
    # Single-transformer → no expert routing.
    assert _shown(vis, "expert_swap_mode") is False
    assert _shown(vis, "expert_switch_interval") is False


def test_wan22_shows_frames_and_experts_hides_audio():
    vis = _visibility(_WAN22_DEF)
    assert _shown(vis, "num_frames") is True
    assert _shown(vis, "expert_swap_mode") is True
    assert _shown(vis, "expert_switch_interval") is True
    # WAN 2.2 has no audio modality.
    assert _shown(vis, "train_audio") is False
    assert _shown(vis, "audio_loss_weight") is False


def test_hidden_video_fields_carry_a_reason():
    vis = _visibility(_IMAGE_DEF)
    assert "no video frames" in vis["num_frames"]["reason"]
    assert "audio" in vis["train_audio"]["reason"]
    assert "expert" in vis["expert_swap_mode"]["reason"]


# ── 3. VRAM scales with frames; image estimate unchanged ─────────────────


def test_vram_activations_scale_with_num_frames():
    defn = registry.get_definition(_WAN21_DEF)
    assert defn is not None

    few = VRAMEstimator.estimate(defn, {"quantization": "none", "num_frames": 5})
    many = VRAMEstimator.estimate(defn, {"quantization": "none", "num_frames": 81})

    # More frames → more latent timesteps → more activation memory.
    assert many.activations_mb > few.activations_mb
    # The training peak (which includes activations) rises too.
    assert many.training_peak_mb > few.training_peak_mb
    assert math.isfinite(many.peak_mb)


def test_image_vram_unchanged_by_video_path():
    defn = registry.get_definition(_IMAGE_DEF)
    assert defn is not None

    # An image family must produce an IDENTICAL estimate regardless of any
    # num_frames value in the config — the video path must collapse to a no-op
    # (latent_frames=1, no expert term).
    no_frames = VRAMEstimator.estimate(defn, {"quantization": "none"}).to_dict()
    with_frames = VRAMEstimator.estimate(
        defn, {"quantization": "none", "num_frames": 81}
    ).to_dict()
    assert no_frames == with_frames

    # And the activation row is a real, finite, positive number (the video
    # branch didn't zero it out).
    assert no_frames["activations_mb"] > 0


def test_wan22_resident_costs_more_weights_than_swap():
    defn = registry.get_definition(_WAN22_DEF)
    assert defn is not None

    swap = VRAMEstimator.estimate(
        defn, {"quantization": "none", "expert_swap_mode": "swap"}
    )
    resident = VRAMEstimator.estimate(
        defn, {"quantization": "none", "expert_swap_mode": "resident"}
    )
    # Resident keeps BOTH experts on the GPU → ~2× the weight term of swap.
    assert resident.model_weights_mb > swap.model_weights_mb
    assert resident.model_weights_mb == pytest.approx(
        swap.model_weights_mb * 2, rel=1e-6
    )
