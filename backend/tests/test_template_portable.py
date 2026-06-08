"""Unit tests for the pure template portable module."""

import io
import json
import zipfile

import pytest

from app.core.portable.envelope import ManifestError
from app.core.template import portable


def _train_row():
    return {
        "id": "train_1", "project_id": "p1", "name": "Anime LoRA",
        "definition_id": "flux2-klein-base-4b",
        "config": {"definition_id": "flux2-klein-base-4b", "lora_name": "anime"},
        "created_at": 1.0, "updated_at": 2.0, "used_count": 5,
        "is_default": False, "readonly": False, "branched_from": "train_0",
    }


def test_build_training_entry_carries_fields_and_drops_machine_specific():
    e = portable.build_template_entry("training", _train_row(), definition=None)
    assert e["domain"] == "training"
    assert e["name"] == "Anime LoRA"
    assert e["definition_id"] == "flux2-klein-base-4b"
    assert e["config"]["lora_name"] == "anime"
    for dropped in ("id", "project_id", "created_at", "updated_at",
                    "used_count", "is_default", "readonly", "branched_from"):
        assert dropped not in e
    # no definition carried when none provided
    assert "definition" not in e


def test_build_training_entry_embeds_definition_when_provided():
    defn = {"id": "flux2-klein-base-4b", "family": "flux2",
            "components": {"repo": {"path": "huggingface:foo/bar"}}}
    e = portable.build_template_entry("training", _train_row(), definition=defn)
    assert e["definition"]["family"] == "flux2"


def test_build_captioning_entry_carries_model_and_prompt_fields():
    row = {"id": "cap_1", "project_id": "p1", "name": "Cap", "model_id": "qwen3-vl",
           "system_prompt": "Describe", "wildcard": "w", "config": {"max_tokens": 512},
           "created_at": 1.0}
    e = portable.build_template_entry("captioning", row, definition=None)
    assert e["domain"] == "captioning"
    assert e["model_id"] == "qwen3-vl"
    assert e["system_prompt"] == "Describe"
    assert e["wildcard"] == "w"
    assert e["config"]["max_tokens"] == 512
    assert "definition" not in e
    assert "id" not in e


def test_build_masking_entry_carries_model_and_config_only():
    row = {"id": "mask_1", "name": "Mask", "model_id": "sam3",
           "config": {"text_prompt": "subject"}, "created_at": 1.0}
    e = portable.build_template_entry("masking", row, definition=None)
    assert e["domain"] == "masking"
    assert e["model_id"] == "sam3"
    assert e["config"]["text_prompt"] == "subject"
    assert "system_prompt" not in e


def test_build_template_entry_rejects_unknown_domain():
    with pytest.raises(ValueError):
        portable.build_template_entry("bogus", {}, definition=None)


def test_build_template_manifest_wraps_entries_with_header():
    e = portable.build_template_entry("masking",
                                      {"name": "M", "model_id": "sam3", "config": {}}, None)
    m = portable.build_template_manifest([e], app_version="0.4.0-alpha")
    assert m["kind"] == "template"
    assert m["format_version"] == portable.MANIFEST_VERSION
    assert m["app_version"] == "0.4.0-alpha"
    assert m["templates"][0]["model_id"] == "sam3"


def _zip_manifest(manifest: dict) -> zipfile.ZipFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
    buf.seek(0)
    return zipfile.ZipFile(buf)


def test_read_template_manifest_returns_entries():
    m = portable.build_template_manifest(
        [portable.build_template_entry("masking",
            {"name": "M", "model_id": "sam3", "config": {}}, None)],
        app_version="x")
    with _zip_manifest(m) as zf:
        out = portable.read_template_manifest(zf)
    assert out["templates"][0]["domain"] == "masking"


def test_read_template_manifest_rejects_empty_and_bad_domain():
    with _zip_manifest({"format_version": 1, "kind": "template", "templates": []}) as zf:
        with pytest.raises(ManifestError):
            portable.read_template_manifest(zf)
    with _zip_manifest(
        {"format_version": 1, "kind": "template",
         "templates": [{"domain": "nope"}]}) as zf:
        with pytest.raises(ManifestError):
            portable.read_template_manifest(zf)


def test_read_template_manifest_rejects_wrong_kind():
    with _zip_manifest({"format_version": 1, "kind": "dataset", "templates": []}) as zf:
        with pytest.raises(ManifestError):
            portable.read_template_manifest(zf)


def test_is_local_component_path():
    assert portable.is_local_component_path("D:/models/foo") is True
    assert portable.is_local_component_path("/mnt/models/bar") is True
    assert portable.is_local_component_path("huggingface:org/repo") is False
    assert portable.is_local_component_path("https://example.com/x") is False
    assert portable.is_local_component_path("") is False
    assert portable.is_local_component_path(None) is False


def test_scan_local_component_paths_finds_only_local():
    defn = {"components": {
        "repo": {"path": "huggingface:org/repo"},
        "vae": {"path": "D:/models/vae"},
        "te": "C:/local/te",  # shorthand string form
    }}
    found = portable.scan_local_component_paths(defn)
    names = {f["component"] for f in found}
    assert names == {"vae", "te"}
