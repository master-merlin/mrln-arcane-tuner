"""Unit tests for the pure project portable module."""

import io
import json
import zipfile

import pytest

from app.core.portable.envelope import ManifestError
from app.core.project import portable


def test_build_project_manifest_shape():
    project = {"id": "p1", "name": "Anime", "description": "d", "color": "#abc",
               "created_at": 1.0, "updated_at": 2.0}
    prefs = {"id": "pref1", "project_id": "p1", "selected_caption_model": "qwen3-vl",
             "training_selections": {"x": 1}}
    templates = [{"domain": "training", "archive": "templates/t.zip"}]
    datasets = [{"mode": "embed", "name": "ds", "archive": "datasets/ds.zip"},
                {"mode": "reference", "name": "shared"}]
    m = portable.build_project_manifest(project, prefs, templates, datasets, "v")
    assert m["kind"] == "project"
    assert m["format_version"] == portable.MANIFEST_VERSION
    assert m["app_version"] == "v"
    assert m["project"]["name"] == "Anime"
    assert m["project"]["color"] == "#abc"
    # machine-specific project fields dropped
    assert "id" not in m["project"] and "created_at" not in m["project"]
    # preferences carried but id/project_id stripped
    assert m["project"]["preferences"]["selected_caption_model"] == "qwen3-vl"
    assert "id" not in m["project"]["preferences"]
    assert "project_id" not in m["project"]["preferences"]
    assert m["templates"] == templates
    assert m["datasets"] == datasets


def _zip_manifest(manifest: dict) -> zipfile.ZipFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
    buf.seek(0)
    return zipfile.ZipFile(buf)


def test_read_project_manifest_defaults_and_kind():
    project = {"name": "P", "description": "", "color": ""}
    m = portable.build_project_manifest(project, {}, [], [], "v")
    with _zip_manifest(m) as zf:
        out = portable.read_project_manifest(zf)
    assert out["project"]["name"] == "P"
    assert out["templates"] == []
    assert out["datasets"] == []


def test_read_project_manifest_rejects_wrong_kind():
    with _zip_manifest({"format_version": 1, "kind": "template"}) as zf:
        with pytest.raises(ManifestError):
            portable.read_project_manifest(zf)


def test_slugify_ascii_only():
    assert portable.slugify("Anime Style") == "Anime_Style"
    assert portable.slugify("a/b\\c") == "a_b_c"
    assert portable.slugify("日本語") == "project"
    assert portable.slugify("") == "project"
