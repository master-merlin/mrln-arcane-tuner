"""Tests for the video-training contract — model-derived profile + run-config
validation (prevent-by-construction, else reject hard).

Uses the real registry + shipped definitions (mirrors test_resolve_capabilities).
"""

from __future__ import annotations

import pytest

from app.engine.models.registry import registry
from app.engine.core.video_contract import (
    frame_predicate,
    resolve_video_profile,
    validate_video_config,
)
from app.engine.components.bucketing import BucketManager


@pytest.fixture(scope="module", autouse=True)
def _loaded_registry():
    registry.discover_families()
    registry.load_definitions("app/engine/models/definitions")
    return registry


# Real shipped definition ids.
LTX2 = "ltx2-3-base"
WAN21_T2V = "wan2.1-t2v-1.3b"
WAN21_I2V = "wan2.1-i2v-14b-720p"
WAN22_T2V = "wan2.2-t2v-a14b"
SDXL = "sdxl_base_1.0"


def _defn(model_id):
    d = registry.get_definition(model_id)
    assert d is not None, f"missing definition {model_id}"
    return d


# ── frame_predicate / Nn+1 parsing ───────────────────────────────────────


def test_frame_predicate_4n1():
    p = frame_predicate("4n+1")
    assert p(1) and p(5) and p(9) and p(81)
    assert not p(4) and not p(8) and not p(80)


def test_frame_predicate_8n1():
    p = frame_predicate("8n+1")
    assert p(1) and p(9) and p(17)
    assert not p(5) and not p(8) and not p(80)


def test_frame_predicate_none_is_unconstrained():
    p = frame_predicate(None)
    assert p(1) and p(7) and p(13)


def test_frame_ladder_generalizes_to_any_nn1():
    # Future family: 6n+1 must produce [1, 7, 13, ...] without code edits.
    assert BucketManager.frame_ladder(25, "6n+1") == [1, 7, 13, 19, 25]
    assert BucketManager._parse_frame_step("6n+1") == 6
    assert BucketManager._parse_frame_step("nonsense") is None


# ── resolve_video_profile ─────────────────────────────────────────────────


def test_profile_ltx2():
    p = resolve_video_profile(_defn(LTX2))
    assert p.is_video and p.has_audio
    assert p.frame_rule == "8n+1"
    assert p.native_fps == 24.0
    assert p.vae_spatial == 32 and p.vae_temporal == 8
    assert p.mode == "both" and p.supports_i2v()
    assert p.dual_expert is False


def test_profile_wan21_t2v():
    p = resolve_video_profile(_defn(WAN21_T2V))
    assert p.is_video and not p.has_audio
    assert p.frame_rule == "4n+1" and p.native_fps == 16.0
    assert p.mode == "t2v" and not p.supports_i2v()
    assert p.dual_expert is False


def test_profile_wan22_is_dual_expert():
    p = resolve_video_profile(_defn(WAN22_T2V))
    assert p.is_video and p.dual_expert and not p.has_audio


def test_profile_image_model_not_video():
    p = resolve_video_profile(_defn(SDXL))
    assert p.is_video is False


# ── validate_video_config: derive + reject ────────────────────────────────


def test_video_model_derives_frame_rule():
    r = validate_video_config(_defn(WAN21_T2V), {"num_frames": 81})
    assert r.ok
    assert r.derived["frame_rule"] == "4n+1"
    assert r.derived["video_native_fps"] == 16.0


def test_reject_audio_on_non_audio_model():
    r = validate_video_config(_defn(WAN21_T2V), {"train_audio": True})
    assert not r.ok
    assert any("audio" in e.lower() for e in r.errors)


def test_allow_audio_on_ltx2():
    r = validate_video_config(_defn(LTX2), {"train_audio": True, "num_frames": 25})
    assert r.ok


def test_reject_i2v_on_t2v_only_model():
    r = validate_video_config(_defn(WAN21_T2V), {"video_mode": "i2v"})
    assert not r.ok
    assert any("image-to-video" in e.lower() for e in r.errors)


def test_allow_i2v_on_capable_models():
    assert validate_video_config(_defn(WAN21_I2V), {"video_mode": "i2v"}).ok
    assert validate_video_config(_defn(LTX2), {"video_mode": "i2v"}).ok


def test_reject_bad_frame_count():
    assert not validate_video_config(_defn(WAN21_T2V), {"num_frames": 80}).ok  # 4n+1
    assert not validate_video_config(_defn(LTX2), {"num_frames": 80}).ok  # 8n+1


def test_accept_valid_frame_count_and_single_image():
    assert validate_video_config(_defn(WAN21_T2V), {"num_frames": 81}).ok
    assert validate_video_config(_defn(LTX2), {"num_frames": 81}).ok
    # Single still (F=1) is a valid, intentional path on every video family.
    for mid in (WAN21_T2V, WAN22_T2V, LTX2):
        assert validate_video_config(_defn(mid), {"num_frames": 1}).ok


def test_target_fps_native_or_match_ok_mismatch_rejected():
    assert validate_video_config(_defn(WAN21_T2V), {"target_fps": 0}).ok
    assert validate_video_config(_defn(WAN21_T2V), {"target_fps": 16}).ok
    assert not validate_video_config(_defn(WAN21_T2V), {"target_fps": 30}).ok


def test_image_model_rejects_audio_but_no_derive():
    r = validate_video_config(_defn(SDXL), {})
    assert r.ok and r.derived == {}
    assert not validate_video_config(_defn(SDXL), {"train_audio": True}).ok
