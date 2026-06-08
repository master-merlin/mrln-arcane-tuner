"""Unit tests for the shared portable-archive envelope and zip helpers."""

import io
import json
import zipfile

import pytest

from app.core.portable.envelope import (
    ManifestError,
    build_manifest_header,
    read_manifest,
)


def test_build_manifest_header_carries_kind_version_and_app_version():
    h = build_manifest_header("template", 1, app_version="0.4.0-alpha")
    assert h["kind"] == "template"
    assert h["format_version"] == 1
    assert h["app_version"] == "0.4.0-alpha"
    assert isinstance(h["exported_at"], float)


def _zip_with_manifest(manifest: dict) -> zipfile.ZipFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
    buf.seek(0)
    return zipfile.ZipFile(buf)


def test_read_manifest_accepts_matching_kind_and_version():
    m = {"format_version": 1, "kind": "template", "extra": 9}
    with _zip_with_manifest(m) as zf:
        out = read_manifest(zf, expected_kind="template", max_version=1)
    assert out["extra"] == 9


def test_read_manifest_rejects_wrong_kind():
    m = {"format_version": 1, "kind": "dataset"}
    with _zip_with_manifest(m) as zf:
        with pytest.raises(ManifestError):
            read_manifest(zf, expected_kind="template", max_version=1)


def test_read_manifest_rejects_future_version():
    m = {"format_version": 2, "kind": "template"}
    with _zip_with_manifest(m) as zf:
        with pytest.raises(ManifestError):
            read_manifest(zf, expected_kind="template", max_version=1)


def test_read_manifest_rejects_missing_manifest():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.jpg", b"img")
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        with pytest.raises(ManifestError):
            read_manifest(zf, expected_kind="template", max_version=1)


def test_read_manifest_rejects_malformed_json():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("manifest.json", b"{not json")
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        with pytest.raises(ManifestError):
            read_manifest(zf, expected_kind="template", max_version=1)


from app.core.portable.archive import safe_extract, write_zip


def test_write_zip_includes_manifest_and_files_and_honors_skip_dirs(tmp_path):
    root = tmp_path / "thing"
    (root / "sub").mkdir(parents=True)
    (root / ".cache").mkdir()
    (root / "a.txt").write_text("a", encoding="utf-8")
    (root / "sub" / "b.bin").write_bytes(b"b")
    (root / ".cache" / "skip.bin").write_bytes(b"NOPE")

    buf = write_zip(root, {"kind": "template", "format_version": 1}, skip_dirs=(".cache",))

    with zipfile.ZipFile(buf) as zf:
        names = set(zf.namelist())
    assert "manifest.json" in names
    assert "a.txt" in names
    assert "sub/b.bin" in names
    assert not any(n.startswith(".cache/") for n in names)


def test_safe_extract_blocks_path_traversal(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../escape.txt", b"evil")
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        with pytest.raises(ManifestError):
            safe_extract(zf, dest)
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_writes_files_and_skips_manifest(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("manifest.json", b"{}")
        zf.writestr("a.jpg", b"img")
        zf.writestr("masks/a.png", b"mask")
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        safe_extract(zf, dest)
    assert (dest / "a.jpg").read_bytes() == b"img"
    assert (dest / "masks" / "a.png").read_bytes() == b"mask"
    assert not (dest / "manifest.json").exists()
