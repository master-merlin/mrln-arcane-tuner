"""Route tests for template export (single + bundle)."""

import io
import json
import zipfile
from unittest.mock import MagicMock, patch

_CAP_REPO = "app.core.db.repositories.captioning_template_repo.CaptioningTemplateRepository"
_MASK_REPO = "app.core.db.repositories.masking_template_repo.MaskingTemplateRepository"
_TRAIN_REPO = "app.core.db.repositories.training_template_repo.TrainingTemplateRepository"
_REGISTRY = "app.engine.models.registry.registry"


def _manifest_from_response(resp) -> dict:
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        return json.loads(zf.read("manifest.json"))


@patch(_CAP_REPO)
def test_export_single_captioning_template(MockRepo, client):
    MockRepo.return_value.get_by_id.return_value = {
        "id": "cap_1", "project_id": "p1", "name": "My Caption",
        "model_id": "qwen3-vl", "system_prompt": "Describe", "wildcard": "",
        "config": {"max_tokens": 512}, "created_at": 1.0,
    }
    resp = client.get("/api/templates/captioning/cap_1/export")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    manifest = _manifest_from_response(resp)
    assert manifest["kind"] == "template"
    entry = manifest["templates"][0]
    assert entry["domain"] == "captioning"
    assert entry["model_id"] == "qwen3-vl"
    assert entry["name"] == "My Caption"
    assert "id" not in entry and "project_id" not in entry


@patch(_MASK_REPO)
def test_export_non_ascii_template_name_does_not_crash(MockRepo, client):
    # A Unicode name must not crash the latin-1-encoded Content-Disposition header.
    MockRepo.return_value.get_by_id.return_value = {
        "id": "mask_1", "name": "日本語のマスク", "model_id": "sam3",
        "config": {}, "created_at": 1.0,
    }
    resp = client.get("/api/templates/masking/mask_1/export")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    # ASCII-only fallback filename in the header (no crash).
    assert "template.zip" in resp.headers["content-disposition"]
    entry = _manifest_from_response(resp)["templates"][0]
    assert entry["name"] == "日本語のマスク"  # the real name still travels in the manifest


@patch(_REGISTRY)
@patch(_TRAIN_REPO)
def test_export_single_training_template_embeds_definition(MockRepo, mock_registry, client):
    MockRepo.return_value.get_by_id.return_value = {
        "id": "train_1", "name": "Anime LoRA",
        "definition_id": "flux2-klein-base-4b",
        "config": {"definition_id": "flux2-klein-base-4b"}, "created_at": 1.0,
    }
    defn = MagicMock()
    defn.model_dump.return_value = {
        "id": "flux2-klein-base-4b", "family": "flux2",
        "components": {"repo": {"path": "huggingface:foo/bar"}},
    }
    mock_registry.get_definition.return_value = defn
    resp = client.get("/api/templates/training/train_1/export")
    assert resp.status_code == 200
    entry = _manifest_from_response(resp)["templates"][0]
    assert entry["definition_id"] == "flux2-klein-base-4b"
    assert entry["definition"]["family"] == "flux2"


@patch(_REGISTRY)
@patch(_TRAIN_REPO)
def test_export_training_template_omits_definition_when_missing(MockRepo, mock_registry, client):
    MockRepo.return_value.get_by_id.return_value = {
        "id": "train_1", "name": "Orphan", "definition_id": "gone",
        "config": {}, "created_at": 1.0,
    }
    mock_registry.get_definition.return_value = None
    resp = client.get("/api/templates/training/train_1/export")
    assert resp.status_code == 200
    entry = _manifest_from_response(resp)["templates"][0]
    assert "definition" not in entry


@patch(_TRAIN_REPO)
def test_export_single_404_when_missing(MockRepo, client):
    MockRepo.return_value.get_by_id.return_value = None
    resp = client.get("/api/templates/training/nope/export")
    assert resp.status_code == 404


def test_export_unknown_domain_400(client):
    resp = client.get("/api/templates/bogus/x/export")
    assert resp.status_code == 400


@patch(_MASK_REPO)
@patch(_CAP_REPO)
def test_export_bundle_collects_multiple_domains(MockCap, MockMask, client):
    MockCap.return_value.get_by_id.return_value = {
        "id": "cap_1", "name": "C", "model_id": "qwen3-vl",
        "system_prompt": "d", "wildcard": "", "config": {}, "created_at": 1.0,
    }
    MockMask.return_value.get_by_id.return_value = {
        "id": "mask_1", "name": "M", "model_id": "sam3", "config": {}, "created_at": 1.0,
    }
    resp = client.post("/api/templates/export", json={"items": [
        {"domain": "captioning", "id": "cap_1"},
        {"domain": "masking", "id": "mask_1"},
    ]})
    assert resp.status_code == 200
    entries = _manifest_from_response(resp)["templates"]
    domains = {e["domain"] for e in entries}
    assert domains == {"captioning", "masking"}


def test_export_bundle_empty_400(client):
    resp = client.post("/api/templates/export", json={"items": []})
    assert resp.status_code == 400


@patch(_MASK_REPO)
def test_export_template_archive_bytes_returns_zip_bytes(MockRepo, client):
    from app.api.training.template_routes import export_template_archive_bytes

    MockRepo.return_value.get_by_id.return_value = {
        "id": "mask_1", "name": "M", "model_id": "sam3", "config": {}, "created_at": 1.0,
    }
    payload = export_template_archive_bytes("masking", "mask_1")
    assert isinstance(payload, bytes)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest["kind"] == "template"
    assert manifest["templates"][0]["model_id"] == "sam3"


@patch(_MASK_REPO)
def test_export_template_archive_bytes_none_when_missing(MockRepo, client):
    from app.api.training.template_routes import export_template_archive_bytes

    MockRepo.return_value.get_by_id.return_value = None
    assert export_template_archive_bytes("masking", "nope") is None
