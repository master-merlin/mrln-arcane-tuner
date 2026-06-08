"""Route tests for project import (plan + apply)."""

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
