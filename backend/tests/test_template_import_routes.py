"""Route tests for template import — plan phase."""

from unittest.mock import MagicMock, patch

from app.core.portable.archive import write_manifest_zip
from app.core.template import portable

_CAP_REPO = "app.core.db.repositories.captioning_template_repo.CaptioningTemplateRepository"
_TRAIN_REPO = "app.core.db.repositories.training_template_repo.TrainingTemplateRepository"
_REGISTRY = "app.engine.models.registry.registry"
_OVERRIDE = "app.engine.utils.model_override_manager.ModelOverrideManager"


def _zip_bytes(entries) -> bytes:
    manifest = portable.build_template_manifest(entries, "test")
    return write_manifest_zip(manifest).getvalue()


def _upload(client, path, zip_bytes, **form):
    return client.post(
        path, files={"file": ("t.template.zip", zip_bytes, "application/zip")}, data=form
    )


@patch(_CAP_REPO)
def test_plan_captioning_available_model(MockRepo, client):
    MockRepo.return_value.list_for_project.return_value = []
    entry = portable.build_template_entry(
        "captioning",
        {"name": "Cap", "model_id": "qwen3-vl", "system_prompt": "d",
         "wildcard": "", "config": {"max_tokens": 256}}, None)
    resp = _upload(client, "/api/templates/import/plan", _zip_bytes([entry]))
    assert resp.status_code == 200
    item = resp.json()["entries"][0]
    assert item["domain"] == "captioning"
    assert item["model_available"] is True
    assert item["blocker"] is False


@patch(_CAP_REPO)
def test_plan_captioning_unavailable_model_is_blocker(MockRepo, client):
    MockRepo.return_value.list_for_project.return_value = []
    entry = portable.build_template_entry(
        "captioning",
        {"name": "Cap", "model_id": "ghost-model", "system_prompt": "d",
         "wildcard": "", "config": {}}, None)
    resp = _upload(client, "/api/templates/import/plan", _zip_bytes([entry]))
    assert resp.status_code == 200
    item = resp.json()["entries"][0]
    assert item["model_available"] is False
    assert item["blocker"] is True


@patch(_OVERRIDE)
@patch(_REGISTRY)
@patch(_TRAIN_REPO)
def test_plan_training_definition_present(MockRepo, mock_registry, mock_override, client):
    MockRepo.return_value.list_for_project.return_value = []
    mock_registry.get_definition.return_value = MagicMock()  # present
    entry = portable.build_template_entry(
        "training", {"name": "T", "definition_id": "flux2-x", "config": {}}, None)
    resp = _upload(client, "/api/templates/import/plan", _zip_bytes([entry]))
    assert resp.status_code == 200
    item = resp.json()["entries"][0]
    assert item["definition_status"] == "present"
    assert item["blocker"] is False


@patch(_OVERRIDE)
@patch(_REGISTRY)
@patch(_TRAIN_REPO)
def test_plan_training_missing_carried_installable_with_hf_substitute(
        MockRepo, mock_registry, mock_override, client):
    MockRepo.return_value.list_for_project.return_value = []
    mock_override.is_offline.return_value = False
    # The target definition is absent...
    # ...but a same-family definition present on the machine has an HF "vae".
    sibling = MagicMock()
    sibling.family = "flux2"
    sibling.components = {"vae": MagicMock(path="huggingface:org/flux2-vae")}
    def _get_def(did):
        return None if did == "flux2-x" else None
    mock_registry.get_definition.side_effect = lambda did: None
    mock_registry.list_models.return_value = ["other"]
    mock_registry.get_definition.side_effect = (
        lambda did: sibling if did == "other" else None)
    carried = {"id": "flux2-x", "family": "flux2", "name": "X",
               "components": {"repo": {"path": "huggingface:o/r"},
                              "vae": {"path": "D:/old/vae"}}}
    entry = portable.build_template_entry(
        "training", {"name": "T", "definition_id": "flux2-x", "config": {}}, carried)
    resp = _upload(client, "/api/templates/import/plan", _zip_bytes([entry]))
    assert resp.status_code == 200
    item = resp.json()["entries"][0]
    assert item["definition_status"] == "installable"
    comps = {c["component"]: c for c in item["local_components"]}
    assert comps["vae"]["hf_substitute"] == "huggingface:org/flux2-vae"
    assert item["blocker"] is False


@patch(_OVERRIDE)
@patch(_REGISTRY)
@patch(_TRAIN_REPO)
def test_plan_training_missing_not_carried_is_blocker(
        MockRepo, mock_registry, mock_override, client):
    MockRepo.return_value.list_for_project.return_value = []
    mock_registry.get_definition.return_value = None
    entry = portable.build_template_entry(
        "training", {"name": "T", "definition_id": "gone", "config": {}}, None)
    resp = _upload(client, "/api/templates/import/plan", _zip_bytes([entry]))
    assert resp.status_code == 200
    item = resp.json()["entries"][0]
    assert item["definition_status"] == "missing"
    assert item["blocker"] is True


def test_plan_rejects_non_template_zip(client):
    bad = write_manifest_zip({"format_version": 1, "kind": "dataset"}).getvalue()
    resp = _upload(client, "/api/templates/import/plan", bad)
    assert resp.status_code == 400
