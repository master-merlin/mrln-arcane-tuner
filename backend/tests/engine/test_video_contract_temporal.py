"""Phase 1 temporal-sampling config schema + capability gate + contract validation (no GPU)."""

from __future__ import annotations

from app.engine.core.archetypes import build_field_visibility
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
    for key in ("temporal_coverage", "frame_stride"):
        assert vis[key]["supported"] is True
