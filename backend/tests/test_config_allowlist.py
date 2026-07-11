"""Tests for the backend capability allowlist — the single server-side choke
point that silently drops top-level config keys a family's descriptor gates OFF.

Descriptor SSOT: ``resolve_capabilities(defn)["field_visibility"]`` (the exact
map the Training UI reads). The allowlist only ever pops a key that appears in
that map AND is marked unsupported; every other key (runtime/system, unknown
vendor, ungated schema) is left untouched. Uses the real registry + shipped
definitions (mirrors test_resolve_capabilities / test_video_contract).
"""

from __future__ import annotations

import pytest

from app.engine.models.registry import registry
from app.engine.core.config_allowlist import (
    apply_capability_allowlist,
    compute_disallowed_keys,
    EXEMPT_KEYS,
)


@pytest.fixture(scope="module", autouse=True)
def _loaded_registry():
    registry.discover_families()
    registry.load_definitions("app/engine/models/definitions")
    return registry


# Real shipped definition ids.
IMAGE = "flux1-dev"          # is_video False, has_audio False, non-edit
EDIT = "flux1-kontext-dev"   # control_inputs > 0 → is_edit True
VIDEO = "wan2.1-t2v-1.3b"    # is_video True


def _defn(model_id):
    d = registry.get_definition(model_id)
    assert d is not None, f"missing definition {model_id}"
    return d


# ── Image family: video/audio knobs gated off are dropped ──────────────────


def test_image_family_drops_video_fields():
    cfg = {"num_frames": 81, "target_fps": 24, "learning_rate": 1e-4}
    dropped = apply_capability_allowlist(cfg, _defn(IMAGE))
    assert set(dropped) == {"num_frames", "target_fps"}
    assert "num_frames" not in cfg and "target_fps" not in cfg
    # Ungated field survives.
    assert cfg["learning_rate"] == 1e-4


def test_image_family_drops_train_audio():
    cfg = {"train_audio": False, "audio_loss_weight": 0.5}
    dropped = apply_capability_allowlist(cfg, _defn(IMAGE))
    assert set(dropped) == {"train_audio", "audio_loss_weight"}
    assert cfg == {}


def test_image_family_drops_control_resolution_but_keeps_flips():
    # Non-edit image model: augmentation supported (flips kept), control_resolution
    # gated off (is_edit False).
    cfg = {"h_flip": True, "v_flip": True, "control_resolution": 512}
    dropped = apply_capability_allowlist(cfg, _defn(IMAGE))
    assert dropped == ["control_resolution"]
    assert cfg == {"h_flip": True, "v_flip": True}


# ── Video family: the same keys are SUPPORTED and pass through ─────────────


def test_video_family_keeps_video_fields():
    cfg = {"num_frames": 81, "target_fps": 24, "temporal_coverage": "tiled"}
    dropped = apply_capability_allowlist(cfg, _defn(VIDEO))
    assert dropped == []
    assert cfg == {"num_frames": 81, "target_fps": 24, "temporal_coverage": "tiled"}


# ── Edit family: augmentation + masked variants gated off (top-level) ──────


def test_edit_family_drops_flips_and_masking():
    cfg = {"h_flip": True, "v_flip": True, "masking_enabled": True,
           "control_resolution": 768}
    dropped = apply_capability_allowlist(cfg, _defn(EDIT))
    # Flips + masking dropped; control_resolution SUPPORTED on an edit model.
    assert set(dropped) == {"h_flip", "v_flip", "masking_enabled"}
    assert cfg == {"control_resolution": 768}


# ── Nested descriptor semantics: flat field-name map, NO path syntax ───────


def test_nested_dataset_masking_is_untouched():
    """The descriptor (_FIELD_RULES) is a FLAT field-name→gate map with no
    path syntax, and ``datasets`` is exempt — so the backend allowlist never
    reaches into ``datasets[].masking_enabled`` (that nested edit-masking gate
    is owned by validate_edit_config at run start)."""
    cfg = {
        "datasets": [
            {"dataset_name": "a", "masking_enabled": True, "h_flip": True},
        ],
    }
    dropped = apply_capability_allowlist(cfg, _defn(EDIT))
    assert dropped == []
    assert cfg["datasets"][0]["masking_enabled"] is True
    assert cfg["datasets"][0]["h_flip"] is True


# ── Exempt runtime/system keys always survive ──────────────────────────────


def test_exempt_keys_survive_even_on_edit_model():
    cfg = {
        "job_id": "abc",
        "definition_id": EDIT,
        "project_id": "p1",
        "lora_name": "my_lora",
        "output_dir": "outputs",
        "resume_from_checkpoint": "outputs/run/ckpt-100",
        "use_cached_latents": True,
        "use_cached_embeddings": True,
        "datasets": [],
    }
    before = dict(cfg)
    dropped = apply_capability_allowlist(cfg, _defn(EDIT))
    assert dropped == []
    assert cfg == before


def test_exempt_keys_are_a_superset_of_injected_keys():
    # Guards against a future rename desyncing the exemption list.
    for k in ("job_id", "definition_id", "resume_from_checkpoint",
              "use_cached_latents", "use_cached_embeddings", "datasets"):
        assert k in EXEMPT_KEYS


# ── Unknown / vendor / forward-compat keys survive ─────────────────────────


def test_unknown_keys_survive():
    cfg = {"num_frames": 81, "my_vendor_flag": 7, "future_field": "x"}
    dropped = apply_capability_allowlist(cfg, _defn(IMAGE))
    assert dropped == ["num_frames"]
    assert cfg == {"my_vendor_flag": 7, "future_field": "x"}


# ── Purity / return contract ───────────────────────────────────────────────


def test_compute_does_not_mutate():
    cfg = {"num_frames": 81}
    keys = compute_disallowed_keys(cfg, _defn(IMAGE))
    assert keys == ["num_frames"]
    # compute_* is read-only.
    assert cfg == {"num_frames": 81}


def test_empty_config_yields_no_drops():
    cfg: dict = {}
    assert apply_capability_allowlist(cfg, _defn(IMAGE)) == []
    assert cfg == {}
