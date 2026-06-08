"""Route tests for project import (plan + apply)."""

import json
from types import SimpleNamespace
from unittest.mock import patch

from app.core.portable.archive import write_bundle_zip, write_manifest_zip
from app.core.project import portable as pportable
from app.core.template import portable as tportable

_PROJECTS = "app.api.project_routes._projects"
_PREFS = "app.api.project_routes._prefs"
_DSMGR = "app.core.dataset_manager.dataset_manager"


def _template_archive(domain, name, model_id="qwen3-vl"):
    entry = tportable.build_template_entry(
        domain, {"name": name, "model_id": model_id, "system_prompt": "d",
                 "wildcard": "", "config": {}}, None)
    return write_manifest_zip(tportable.build_template_manifest([entry], "v")).getvalue()


def _project_zip(project, templates=None, datasets=None, entries=None):
    manifest = pportable.build_project_manifest(
        project, project.get("preferences", {}), templates or [], datasets or [], "v")
    return write_bundle_zip(manifest, entries or {}).getvalue()


def _upload(client, path, zip_bytes, **form):
    return client.post(
        path, files={"file": ("p.project.zip", zip_bytes, "application/zip")}, data=form)


@patch(_PROJECTS)
def test_plan_reports_project_name_conflict(MockProjects, client):
    MockProjects.get_by_name.return_value = {"id": "existing"}  # name taken
    zb = _project_zip({"name": "Anime"})
    resp = _upload(client, "/api/projects/import/plan", zb)
    assert resp.status_code == 200
    body = resp.json()
    assert body["project"]["name"] == "Anime"
    assert body["project"]["conflict"] is True


@patch(_DSMGR)
@patch(_PROJECTS)
def test_plan_reports_reference_dataset_presence(MockProjects, mock_dsmgr, client):
    MockProjects.get_by_name.return_value = None
    mock_dsmgr.get_dataset.side_effect = lambda n: SimpleNamespace(id=n) if n == "here" else None
    zb = _project_zip({"name": "P"}, datasets=[
        {"mode": "reference", "name": "here"},
        {"mode": "reference", "name": "gone"}])
    resp = _upload(client, "/api/projects/import/plan", zb)
    assert resp.status_code == 200
    ds = {d["name"]: d for d in resp.json()["datasets"]}
    assert ds["here"]["reference_present"] is True
    assert ds["gone"]["reference_present"] is False


@patch(_DSMGR)
@patch(_PROJECTS)
def test_plan_reports_embed_dataset_name_conflict(MockProjects, mock_dsmgr, client):
    MockProjects.get_by_name.return_value = None
    mock_dsmgr.get_dataset.return_value = SimpleNamespace(id="x")  # name already exists
    zb = _project_zip({"name": "P"},
                      datasets=[{"mode": "embed", "name": "ds", "archive": "datasets/ds.zip"}],
                      entries={"datasets/ds.zip": b"DSZIP"})
    resp = _upload(client, "/api/projects/import/plan", zb)
    assert resp.status_code == 200
    ds = {d["name"]: d for d in resp.json()["datasets"]}
    assert ds["ds"]["mode"] == "embed" and ds["ds"]["embed_conflict"] is True


@patch(_DSMGR)
@patch(_PROJECTS)
def test_plan_aggregates_template_entries(MockProjects, mock_dsmgr, client):
    MockProjects.get_by_name.return_value = None
    with patch("app.api.project_routes._cap_repo_list", create=True):
        pass
    # captioning template plan is exercised via the real plan_template_entries;
    # patch the captioning repo it lists against.
    with patch("app.core.db.repositories.captioning_template_repo.CaptioningTemplateRepository") as MockCap:
        MockCap.return_value.list_for_project.return_value = []
        zb = _project_zip(
            {"name": "P"},
            templates=[{"domain": "captioning", "archive": "templates/c.zip"}],
            entries={"templates/c.zip": _template_archive("captioning", "Cap")})
        resp = _upload(client, "/api/projects/import/plan", zb)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["templates"]) == 1
    assert body["templates"][0]["domain"] == "captioning"


def test_plan_rejects_non_project_zip(client):
    bad = write_manifest_zip({"format_version": 1, "kind": "template"}).getvalue()
    resp = _upload(client, "/api/projects/import/plan", bad)
    assert resp.status_code == 400


def _apply(client, zip_bytes, resolutions=None):
    form = {}
    if resolutions is not None:
        form["resolutions"] = json.dumps(resolutions)
    return client.post(
        "/api/projects/import/apply",
        files={"file": ("p.project.zip", zip_bytes, "application/zip")}, data=form)


@patch(_PREFS)
@patch(_DSMGR)
@patch(_PROJECTS)
def test_apply_creates_project_and_links_references(MockProjects, mock_dsmgr, MockPrefs, client):
    MockProjects.get_by_name.return_value = None
    MockProjects.create.return_value = {"id": "new_p", "name": "P"}
    mock_dsmgr.get_dataset.side_effect = lambda n: SimpleNamespace(id="d_" + n) if n == "here" else None
    zb = _project_zip({"name": "P", "preferences": {"selected_caption_model": "qwen3-vl"}},
                      datasets=[{"mode": "reference", "name": "here"},
                                {"mode": "reference", "name": "gone"}])
    resp = _apply(client, zb)
    assert resp.status_code == 200
    body = resp.json()
    assert body["project_id"] == "new_p"
    assert body["linked_references"] == ["here"]
    assert body["missing_references"] == ["gone"]
    MockProjects.add_dataset.assert_any_call("new_p", "d_here")
    MockPrefs.upsert.assert_called_once()


@patch(_PROJECTS)
def test_apply_project_name_conflict_no_directive_409(MockProjects, client):
    MockProjects.get_by_name.return_value = {"id": "existing"}
    zb = _project_zip({"name": "Dup"})
    resp = _apply(client, zb)
    assert resp.status_code == 409


@patch(_PREFS)
@patch(_DSMGR)
@patch("app.api.dataset.crud_routes._import_from_zip_path")
@patch(_PROJECTS)
def test_apply_embeds_dataset_and_links(MockProjects, mock_import, mock_dsmgr, MockPrefs, client):
    MockProjects.get_by_name.return_value = None
    MockProjects.create.return_value = {"id": "new_p", "name": "P"}
    mock_import.return_value = SimpleNamespace(id="imp_d", name="ds (imported)")
    zb = _project_zip({"name": "P"},
                      datasets=[{"mode": "embed", "name": "ds", "archive": "datasets/ds.zip"}],
                      entries={"datasets/ds.zip": b"DSZIP"})
    resp = _apply(client, zb)
    assert resp.status_code == 200
    assert resp.json()["imported_datasets"] == ["ds (imported)"]
    MockProjects.add_dataset.assert_any_call("new_p", "imp_d")


@patch(_PREFS)
@patch(_DSMGR)
@patch("app.api.dataset.crud_routes._import_from_zip_path")
@patch(_PROJECTS)
def test_apply_rolls_back_on_dataset_import_failure(MockProjects, mock_import, mock_dsmgr, MockPrefs, client):
    MockProjects.get_by_name.return_value = None
    MockProjects.create.return_value = {"id": "new_p", "name": "P"}
    # First embed imports OK, second blows up → must roll everything back.
    good = SimpleNamespace(id="d1", name="one")
    mock_import.side_effect = [good, RuntimeError("extract boom")]
    zb = _project_zip({"name": "P"}, datasets=[
        {"mode": "embed", "name": "one", "archive": "datasets/one.zip"},
        {"mode": "embed", "name": "two", "archive": "datasets/two.zip"}],
        entries={"datasets/one.zip": b"A", "datasets/two.zip": b"B"})
    resp = _apply(client, zb)
    assert resp.status_code == 500
    # rollback: the created project deleted, the one imported dataset deleted
    MockProjects.delete.assert_called_once_with("new_p")
    mock_dsmgr.delete_dataset.assert_any_call("one", delete_files=True)


@patch(_DSMGR)
@patch(_PROJECTS)
def test_user_triggered_rollback_undoes_a_kept_import(MockProjects, mock_dsmgr, client):
    resp = client.post("/api/projects/import/rollback", json={
        "project_id": "p_keep", "imported_datasets": ["ds1"],
        "installed_definitions": []})
    assert resp.status_code == 200
    assert resp.json()["status"] == "rolled_back"
    MockProjects.delete.assert_called_once_with("p_keep")
    mock_dsmgr.delete_dataset.assert_called_once_with("ds1", delete_files=True)


@patch("app.api.project_routes._uninstall_definition")
@patch("app.api.training.template_routes._install_definition")
@patch("app.core.db.repositories.training_template_repo.TrainingTemplateRepository")
@patch("app.engine.utils.model_override_manager.ModelOverrideManager")
@patch("app.engine.models.registry.registry")
@patch(_PREFS)
@patch(_DSMGR)
@patch(_PROJECTS)
def test_apply_rolls_back_installed_definition_on_later_failure(
        MockProjects, mock_dsmgr, MockPrefs, mock_registry, mock_override,
        MockTrain, mock_install, mock_uninstall, client):
    # A training template installs a definition, then a LATER step (prefs upsert)
    # fails → rollback must uninstall the newly-installed definition.
    MockProjects.get_by_name.return_value = None
    MockProjects.create.return_value = {"id": "new_p", "name": "P"}
    MockTrain.return_value.create.return_value = {"id": "t1", "name": "T"}
    mock_registry.get_definition.return_value = None      # definition missing → install
    mock_override.is_offline.return_value = True           # no HF substitution
    MockPrefs.upsert.side_effect = RuntimeError("prefs boom")  # fail after template install

    carried = {"id": "flux2-x", "family": "flux2", "name": "X",
               "components": {"repo": {"path": "huggingface:o/r"}}}
    entry = tportable.build_template_entry(
        "training", {"name": "T", "definition_id": "flux2-x", "config": {}}, carried)
    tbytes = write_manifest_zip(tportable.build_template_manifest([entry], "v")).getvalue()
    zb = _project_zip(
        {"name": "P", "preferences": {"selected_caption_model": "qwen3-vl"}},
        templates=[{"domain": "training", "archive": "templates/t.zip"}],
        entries={"templates/t.zip": tbytes})

    resp = _apply(client, zb, {"templates": {"0": {"install_definition": True}}})
    assert resp.status_code == 500
    mock_install.assert_called_once()                     # definition was installed
    mock_uninstall.assert_called_once_with("flux2-x")     # …then rolled back
    MockProjects.delete.assert_called_once_with("new_p")
