"""Phase 1 temporal-sampling config schema + capability gate + contract validation (no GPU)."""

from __future__ import annotations

from app.engine.core.archetypes import build_field_visibility
from app.engine.core.video_contract import validate_video_config
from app.engine.models.base import BaseTrainingConfig


def test_phase1_video_fields_exist_with_defaults():
    schema = BaseTrainingConfig.model_json_schema()["properties"]
    # Axis A
    assert schema["temporal_coverage"]["default"] == "first"
    assert set(schema["temporal_coverage"]["enum"]) == {"first", "tiled", "sliding"}
    assert schema["window_overlap"]["default"] == 0.0
    assert schema["max_windows"]["default"] == 10
    # Axis B
    assert schema["frame_stride"]["default"] == 1
    # Forward-compat (declared, inert in Phase 1)
    assert schema["still_resolutions"]["default"] == []
    assert schema["radc_seqlen_influence"]["default"] == 0.0


def test_phase1_video_fields_carry_group_and_depends_on():
    schema = BaseTrainingConfig.model_json_schema()["properties"]
    for key in (
        "temporal_coverage",
        "window_overlap",
        "max_windows",
        "frame_stride",
        "still_resolutions",
        "radc_seqlen_influence",
    ):
        assert schema[key]["group"] == "VIDEO"
    assert schema["window_overlap"]["depends_on"] == "temporal_coverage:tiled"
    assert schema["max_windows"]["depends_on"] == "temporal_coverage:tiled"
    assert schema["radc_seqlen_influence"]["depends_on"] == "timestep_sampling:radc"


def test_new_video_knobs_hidden_for_image_models():
    # An image archetype has is_video False → field_visibility must report the
    # new temporal knobs as unsupported so the frontend hides them.
    vis = build_field_visibility({"is_video": False})
    for key in (
        "temporal_coverage",
        "window_overlap",
        "max_windows",
        "frame_stride",
        "still_resolutions",
        "radc_seqlen_influence",
    ):
        assert key in vis, f"{key} missing from field_visibility"
        assert vis[key]["supported"] is False, f"{key} should be hidden for image models"


def test_new_video_knobs_shown_for_video_models():
    vis = build_field_visibility({"is_video": True})
    for key in (
        "temporal_coverage",
        "window_overlap",
        "max_windows",
        "frame_stride",
        "still_resolutions",
        "radc_seqlen_influence",
    ):
        assert vis[key]["supported"] is True


class _Defn:
    """Minimal definition stub: LTX-2 (8n+1, 24fps) video profile."""

    id = "ltx2/ltx2_3"
    control_inputs = 0
    architecture_params = {
        "video.frame_rule": "8n+1",
        "video.frame_rate": 24.0,
        "video.vae_temporal": 8,
        "video.vae_spatial": 32,
        "mode": "t2v",
        "video.divisibility": 32,
    }


def _defn(monkeypatch):
    # resolve_video_profile imports resolve_capabilities locally from
    # app.engine.core.archetypes (not a module-level name in video_contract),
    # so patch it at the source module. Force is_video=True here.
    import app.engine.core.archetypes as arch

    monkeypatch.setattr(
        arch,
        "resolve_capabilities",
        lambda d: {
            "capabilities": {
                "is_video": True,
                "has_audio": False,
                "dual_expert": False,
                "has_image_encoder": False,
            }
        },
    )
    return _Defn()


def test_valid_tiled_config_passes(monkeypatch):
    rep = validate_video_config(
        _defn(monkeypatch),
        {
            "temporal_coverage": "tiled",
            "window_overlap": 0.5,
            "max_windows": 4,
            "frame_stride": 1,
            "num_frames": 25,
        },
    )
    assert rep.ok, rep.errors


def test_bad_temporal_coverage_rejected(monkeypatch):
    rep = validate_video_config(
        _defn(monkeypatch), {"temporal_coverage": "bogus", "num_frames": 25}
    )
    assert not rep.ok
    assert any("temporal_coverage" in e for e in rep.errors)


def test_window_overlap_out_of_range_rejected(monkeypatch):
    rep = validate_video_config(
        _defn(monkeypatch),
        {"temporal_coverage": "tiled", "window_overlap": 1.5, "num_frames": 25},
    )
    assert not rep.ok
    assert any("window_overlap" in e for e in rep.errors)


def test_max_windows_below_one_rejected(monkeypatch):
    rep = validate_video_config(
        _defn(monkeypatch),
        {"temporal_coverage": "tiled", "max_windows": 0, "num_frames": 25},
    )
    assert not rep.ok
    assert any("max_windows" in e for e in rep.errors)


def test_frame_stride_below_one_rejected(monkeypatch):
    rep = validate_video_config(
        _defn(monkeypatch), {"frame_stride": 0, "num_frames": 25}
    )
    assert not rep.ok
    assert any("frame_stride" in e for e in rep.errors)


def test_stride_keeps_frame_count_valid_against_ladder(monkeypatch):
    # 25 frames is 8n+1 (8*3+1). frame_stride only changes the SAMPLED rate,
    # not the frame COUNT, so 25 stays valid regardless of stride. target_fps=0
    # (native) so the native-fps tolerance check is satisfied.
    rep = validate_video_config(
        _defn(monkeypatch),
        {"frame_stride": 2, "num_frames": 25, "target_fps": 0.0},
    )
    assert rep.ok, rep.errors


def test_nonnative_target_fps_rejected(monkeypatch):
    # punch-list #1(b): a user-set target_fps != native (24) is rejected by the
    # existing native-tolerance check; stride is the only fps lever.
    rep = validate_video_config(
        _defn(monkeypatch), {"num_frames": 25, "target_fps": 12.0}
    )
    assert not rep.ok
    assert any("target_fps" in e for e in rep.errors)


def test_stride_with_manual_target_fps_rejected(monkeypatch):
    # punch-list #1(a): frame_stride>1 combined with a manually-set target_fps
    # is rejected with a precedence message so stride never compounds with an
    # override. (target_fps=24 is native-valid on its own, so this proves the
    # combination — not the native check — is what rejects it.)
    rep = validate_video_config(
        _defn(monkeypatch),
        {"frame_stride": 2, "num_frames": 25, "target_fps": 24.0},
    )
    assert not rep.ok
    assert any("frame_stride" in e and "target_fps" in e for e in rep.errors)


def test_radc_seqlen_influence_rejected_without_radc(monkeypatch):
    rep = validate_video_config(
        _defn(monkeypatch),
        {
            "radc_seqlen_influence": 0.3,
            "timestep_sampling": "uniform",
            "num_frames": 25,
        },
    )
    assert not rep.ok
    assert any("radc_seqlen_influence" in e for e in rep.errors)


def test_sliding_coverage_accepted(monkeypatch):
    # Forward-compat (Phase 2): the contract must ACCEPT 'sliding' (validated,
    # not errored) so Phase 2 wires only behavior, not config/validation/UI.
    rep = validate_video_config(
        _defn(monkeypatch),
        {"temporal_coverage": "sliding", "num_frames": 25},
    )
    assert rep.ok, rep.errors


def test_radc_seqlen_influence_accepted_with_radc(monkeypatch):
    # Positive branch: radc_seqlen_influence > 0 is valid WHEN the radc sampler
    # is selected (mirror of test_radc_seqlen_influence_rejected_without_radc).
    rep = validate_video_config(
        _defn(monkeypatch),
        {
            "radc_seqlen_influence": 0.3,
            "timestep_sampling": "radc",
            "num_frames": 25,
        },
    )
    assert rep.ok, rep.errors


def test_sliding_max_clip_seconds_schema_default_and_metadata():
    schema = BaseTrainingConfig.model_json_schema()["properties"]
    assert schema["sliding_max_clip_seconds"]["default"] == 0.0
    assert schema["sliding_max_clip_seconds"]["group"] == "VIDEO"
    assert schema["sliding_max_clip_seconds"]["depends_on"] == "temporal_coverage:sliding"


def test_sliding_max_clip_seconds_hidden_for_image_models():
    vis = build_field_visibility({"is_video": False})
    assert vis["sliding_max_clip_seconds"]["supported"] is False


def test_sliding_max_clip_seconds_shown_for_video_models():
    vis = build_field_visibility({"is_video": True})
    assert vis["sliding_max_clip_seconds"]["supported"] is True


def test_negative_sliding_max_clip_seconds_rejected(monkeypatch):
    rep = validate_video_config(
        _defn(monkeypatch),
        {"temporal_coverage": "sliding", "sliding_max_clip_seconds": -3, "num_frames": 25},
    )
    assert not rep.ok
    assert any("sliding_max_clip_seconds" in e for e in rep.errors)


def test_zero_sliding_max_clip_seconds_accepted(monkeypatch):
    rep = validate_video_config(
        _defn(monkeypatch),
        {"temporal_coverage": "sliding", "sliding_max_clip_seconds": 0, "num_frames": 25},
    )
    assert rep.ok, rep.errors


class _DefnLtxSched(_Defn):
    architecture_params = {
        **_Defn.architecture_params,
        "scheduler.use_dynamic_shifting": True,
        "scheduler.base_shift": 0.95,
        "scheduler.max_shift": 2.05,
        "scheduler.base_image_seq_len": 1024,
        "scheduler.max_image_seq_len": 4096,
    }


class _DefnWanSched(_Defn):
    architecture_params = {
        **_Defn.architecture_params,
        "scheduler.flow_shift": 3.0,
    }


def _patch_video(monkeypatch):
    import app.engine.core.archetypes as arch
    monkeypatch.setattr(arch, "resolve_capabilities", lambda d: {
        "capabilities": {"is_video": True, "has_audio": False,
                         "dual_expert": False, "has_image_encoder": False}})


def test_contract_injects_ltx_dynamic_shift(monkeypatch):
    _patch_video(monkeypatch)
    rep = validate_video_config(_DefnLtxSched(), {"num_frames": 25})
    assert rep.ok, rep.errors
    assert rep.derived["model_shift_base_shift"] == 0.95
    assert rep.derived["model_shift_max_shift"] == 2.05
    assert rep.derived["model_shift_base_seq"] == 1024
    assert rep.derived["model_shift_max_seq"] == 4096
    assert "model_shift_fixed" not in rep.derived


def test_contract_injects_wan_fixed_shift(monkeypatch):
    _patch_video(monkeypatch)
    rep = validate_video_config(_DefnWanSched(), {"num_frames": 25})
    assert rep.ok, rep.errors
    assert rep.derived["model_shift_fixed"] == 3.0
    assert "model_shift_base_shift" not in rep.derived
