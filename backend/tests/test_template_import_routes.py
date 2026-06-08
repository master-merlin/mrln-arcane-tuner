"""Route tests for template import — plan phase."""

import json
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


_MASK_REPO = "app.core.db.repositories.masking_template_repo.MaskingTemplateRepository"


def _apply(client, zip_bytes, resolutions, project_id=None):
    form = {"resolutions": json.dumps(resolutions)}
    if project_id is not None:
        form["project_id"] = project_id
    return client.post(
        "/api/templates/import/apply",
        files={"file": ("t.template.zip", zip_bytes, "application/zip")}, data=form)


@patch(_CAP_REPO)
def test_apply_creates_captioning_template(MockRepo, client):
    MockRepo.return_value.create.return_value = {"id": "cap_new", "name": "Cap"}
    entry = portable.build_template_entry(
        "captioning",
        {"name": "Cap", "model_id": "qwen3-vl", "system_prompt": "d",
         "wildcard": "", "config": {"max_tokens": 256}}, None)
    resp = _apply(client, _zip_bytes([entry]), {"entries": {}})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["created"]) == 1
    # The repo was asked to create with the carried fields.
    args = MockRepo.return_value.create.call_args[0][0]
    assert args["model_id"] == "qwen3-vl"
    assert args["system_prompt"] == "d"


@patch(_CAP_REPO)
def test_apply_skips_unavailable_model(MockRepo, client):
    entry = portable.build_template_entry(
        "captioning",
        {"name": "Cap", "model_id": "ghost", "config": {}}, None)
    resp = _apply(client, _zip_bytes([entry]), {"entries": {}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["created"] == []
    assert body["skipped"][0]["reason"]
    MockRepo.return_value.create.assert_not_called()


@patch(_CAP_REPO)
def test_apply_rename_override(MockRepo, client):
    MockRepo.return_value.create.return_value = {"id": "x", "name": "Renamed"}
    entry = portable.build_template_entry(
        "captioning",
        {"name": "Cap", "model_id": "qwen3-vl", "system_prompt": "d",
         "wildcard": "", "config": {}}, None)
    resp = _apply(client, _zip_bytes([entry]),
                  {"entries": {"0": {"action": "create", "name": "Renamed"}}})
    assert resp.status_code == 200
    args = MockRepo.return_value.create.call_args[0][0]
    assert args["name"] == "Renamed"


@patch(_OVERRIDE)
@patch(_REGISTRY)
@patch(_TRAIN_REPO)
def test_apply_training_present_definition_creates(
        MockRepo, mock_registry, mock_override, client):
    mock_registry.get_definition.return_value = MagicMock()  # present
    MockRepo.return_value.create.return_value = {"id": "t_new", "name": "T"}
    entry = portable.build_template_entry(
        "training", {"name": "T", "definition_id": "flux2-x", "config": {"a": 1}}, None)
    resp = _apply(client, _zip_bytes([entry]), {"entries": {}})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["created"]) == 1
    args = MockRepo.return_value.create.call_args[0][0]
    assert args["definition_id"] == "flux2-x"


@patch(_OVERRIDE)
@patch(_REGISTRY)
@patch(_TRAIN_REPO)
def test_apply_training_missing_declined_is_skipped(
        MockRepo, mock_registry, mock_override, client):
    mock_registry.get_definition.return_value = None
    carried = {"id": "flux2-x", "family": "flux2", "name": "X",
               "components": {"repo": {"path": "huggingface:o/r"}}}
    entry = portable.build_template_entry(
        "training", {"name": "T", "definition_id": "flux2-x", "config": {}}, carried)
    # install_definition defaults to False → declined.
    resp = _apply(client, _zip_bytes([entry]), {"entries": {}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["created"] == []
    assert "declin" in body["skipped"][0]["reason"].lower() or \
           "install" in body["skipped"][0]["reason"].lower()
    MockRepo.return_value.create.assert_not_called()


@patch(_OVERRIDE)
@patch(_REGISTRY)
@patch(_TRAIN_REPO)
@patch("app.api.training.template_routes._install_definition")
def test_apply_training_missing_confirmed_installs_and_creates(
        mock_install, MockRepo, mock_registry, mock_override, client):
    mock_registry.get_definition.return_value = None
    mock_override.is_offline.return_value = True  # no HF substitution
    MockRepo.return_value.create.return_value = {"id": "t_new", "name": "T"}
    carried = {"id": "flux2-x", "family": "flux2", "name": "X",
               "components": {"repo": {"path": "huggingface:o/r"}}}
    entry = portable.build_template_entry(
        "training", {"name": "T", "definition_id": "flux2-x", "config": {}}, carried)
    resp = _apply(client, _zip_bytes([entry]),
                  {"entries": {"0": {"action": "create", "install_definition": True}}})
    assert resp.status_code == 200
    body = resp.json()
    assert "flux2-x" in body["installed_definitions"]
    assert len(body["created"]) == 1
    mock_install.assert_called_once()


@patch(_OVERRIDE)
@patch(_REGISTRY)
@patch(_TRAIN_REPO)
def test_apply_rejects_traversal_family_as_skip_not_crash(
        MockRepo, mock_registry, mock_override, client):
    # A crafted ``family`` must not write outside the families dir, and must
    # skip the entry (not 500 the whole request).
    mock_registry.get_definition.return_value = None
    mock_override.is_offline.return_value = True
    carried = {"id": "evil", "family": "..", "name": "X",
               "components": {"repo": {"path": "huggingface:o/r"}}}
    entry = portable.build_template_entry(
        "training", {"name": "Evil", "definition_id": "evil", "config": {}}, carried)
    resp = _apply(client, _zip_bytes([entry]),
                  {"entries": {"0": {"action": "create", "install_definition": True}}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["created"] == []
    assert "family" in body["skipped"][0]["reason"].lower()
    MockRepo.return_value.create.assert_not_called()


@patch(_CAP_REPO)
def test_apply_one_create_failure_does_not_abort_bundle(MockRepo, client):
    # Best-effort: a repo.create exception on one entry skips it, the other
    # entry still imports.
    MockRepo.return_value.create.side_effect = [
        RuntimeError("db boom"),
        {"id": "c2", "name": "Good"},
    ]
    e1 = portable.build_template_entry(
        "captioning", {"name": "Bad", "model_id": "qwen3-vl",
                       "system_prompt": "d", "wildcard": "", "config": {}}, None)
    e2 = portable.build_template_entry(
        "captioning", {"name": "Good", "model_id": "qwen3-vl",
                       "system_prompt": "d", "wildcard": "", "config": {}}, None)
    resp = _apply(client, _zip_bytes([e1, e2]), {"entries": {}})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["created"]) == 1 and body["created"][0]["name"] == "Good"
    assert len(body["skipped"]) == 1 and "db boom" in body["skipped"][0]["reason"]
