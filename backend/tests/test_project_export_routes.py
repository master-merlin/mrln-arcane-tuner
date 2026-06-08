"""Route tests for project export."""

import io
import json
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

_PROJECTS = "app.api.project_routes._projects"
_PREFS = "app.api.project_routes._prefs"
_DSMGR = "app.core.dataset_manager.dataset_manager"
_TPL_BYTES = "app.api.training.template_routes.export_template_archive_bytes"


def _manifest(resp) -> dict:
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        return json.loads(zf.read("manifest.json"))


@patch(_PREFS)
@patch(_PROJECTS)
def test_export_project_metadata_and_prefs_only(MockProjects, MockPrefs, client):
    MockProjects.get_by_id.return_value = {
        "id": "p1", "name": "Anime", "description": "d", "color": "#abc",
        "created_at": 1.0, "updated_at": 2.0}
    MockPrefs.get.return_value = {"id": "pr1", "project_id": "p1",
                                  "selected_caption_model": "qwen3-vl"}
    resp = client.post("/api/projects/p1/export", json={"templates": [], "datasets": []})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    m = _manifest(resp)
    assert m["kind"] == "project"
    assert m["project"]["name"] == "Anime"
    assert m["project"]["preferences"]["selected_caption_model"] == "qwen3-vl"
    assert m["templates"] == [] and m["datasets"] == []


@patch(_PREFS)
@patch(_PROJECTS)
def test_export_project_404_when_missing(MockProjects, MockPrefs, client):
    MockProjects.get_by_id.return_value = None
    resp = client.post("/api/projects/nope/export", json={"templates": [], "datasets": []})
    assert resp.status_code == 404


@patch(_TPL_BYTES)
@patch(_PREFS)
@patch(_PROJECTS)
def test_export_project_embeds_template_archive(MockProjects, MockPrefs, mock_tpl, client):
    MockProjects.get_by_id.return_value = {
        "id": "p1", "name": "P", "description": "", "color": "",
        "created_at": 1.0, "updated_at": 2.0}
    MockPrefs.get.return_value = {"id": "pr1", "project_id": "p1"}
    mock_tpl.return_value = b"TEMPLATEZIP"
    resp = client.post("/api/projects/p1/export", json={
        "templates": [{"domain": "training", "id": "t1"}], "datasets": []})
    assert resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = set(zf.namelist())
        m = json.loads(zf.read("manifest.json"))
        tpl_arc = m["templates"][0]["archive"]
        assert zf.read(tpl_arc) == b"TEMPLATEZIP"
    assert any(n.startswith("templates/") for n in names)
    assert m["templates"][0]["domain"] == "training"


@patch(_TPL_BYTES)
@patch(_PREFS)
@patch(_PROJECTS)
def test_export_project_404_when_template_missing(MockProjects, MockPrefs, mock_tpl, client):
    MockProjects.get_by_id.return_value = {
        "id": "p1", "name": "P", "description": "", "color": "",
        "created_at": 1.0, "updated_at": 2.0}
    MockPrefs.get.return_value = {}
    mock_tpl.return_value = None  # template not found
    resp = client.post("/api/projects/p1/export", json={
        "templates": [{"domain": "training", "id": "gone"}], "datasets": []})
    assert resp.status_code == 404


@patch(_DSMGR)
@patch(_PREFS)
@patch(_PROJECTS)
def test_export_project_embed_and_reference_datasets(MockProjects, MockPrefs, mock_dsmgr, client):
    MockProjects.get_by_id.return_value = {
        "id": "p1", "name": "P", "description": "", "color": "",
        "created_at": 1.0, "updated_at": 2.0}
    MockPrefs.get.return_value = {}
    ds = SimpleNamespace(id="d1", name="style", path="/tmp/style")
    mock_dsmgr.get_dataset.return_value = ds
    with patch("app.core.dataset.portable.build_manifest", return_value={"kind": "dataset"}), \
         patch("app.core.dataset.portable.write_export_zip", return_value=io.BytesIO(b"DSZIP")):
        resp = client.post("/api/projects/p1/export", json={
            "templates": [],
            "datasets": [{"name": "style", "mode": "embed"},
                         {"name": "shared", "mode": "reference"},
                         {"name": "skip", "mode": "exclude"}]})
    assert resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        m = json.loads(zf.read("manifest.json"))
        embed = next(d for d in m["datasets"] if d["mode"] == "embed")
        assert zf.read(embed["archive"]) == b"DSZIP"
    modes = {d["mode"]: d for d in m["datasets"]}
    assert set(modes) == {"embed", "reference"}  # exclude omitted
    assert "archive" not in modes["reference"]


@patch(_DSMGR)
@patch(_PREFS)
@patch(_PROJECTS)
def test_export_project_404_when_embed_dataset_missing(MockProjects, MockPrefs, mock_dsmgr, client):
    MockProjects.get_by_id.return_value = {
        "id": "p1", "name": "P", "description": "", "color": "",
        "created_at": 1.0, "updated_at": 2.0}
    MockPrefs.get.return_value = {}
    mock_dsmgr.get_dataset.return_value = None
    resp = client.post("/api/projects/p1/export", json={
        "templates": [], "datasets": [{"name": "gone", "mode": "embed"}]})
    assert resp.status_code == 404
