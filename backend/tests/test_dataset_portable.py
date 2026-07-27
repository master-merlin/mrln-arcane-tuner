"""Unit tests for the portable dataset archive module (manifest + safe zip)."""

import io
import json
import zipfile

import pytest

from app.core.dataset.portable import (
    MANIFEST_VERSION,
    ManifestError,
    build_manifest,
    write_export_zip,
    write_export_zip_to_path,
    read_manifest,
    safe_extract,
)
from app.core.dataset_manager import Dataset


def _make_dataset() -> Dataset:
    return Dataset(
        id="abc-123",
        name="Portraits",
        path="/some/where/Portraits",
        created_at=1.0,
        version="1.2.0",
        trigger_word="mrln_style",
        tags=["face", "studio"],
        notes="hand-picked",
        classifier="style",
        harmonization_score=0.9,
        majority_ar=1.0,
        multimedia_count=2,
        media_metadata={
            "a.jpg": {"width": 64, "height": 64, "quality_score": 0.31,
                      "enabled": True, "has_caption": True},
            "b.jpg": {"width": 64, "height": 64, "quality_score": 0.18,
                      "enabled": False, "has_mask": True},
        },
    )


def test_build_manifest_carries_dataset_and_media_but_not_id_or_path():
    m = build_manifest(_make_dataset(), app_version="0.4.0-alpha")
    assert m["format_version"] == MANIFEST_VERSION
    assert m["app_version"] == "0.4.0-alpha"
    assert m["kind"] == "dataset"
    assert m["dataset"]["trigger_word"] == "mrln_style"
    assert m["dataset"]["tags"] == ["face", "studio"]
    assert m["dataset"]["version"] == "1.2.0"
    # machine-specific fields must NOT be carried
    assert "id" not in m["dataset"]
    assert "path" not in m["dataset"]
    # per-image metadata lives under "media", keyed by rel_path
    assert m["media"]["a.jpg"]["quality_score"] == 0.31
    assert m["media"]["b.jpg"]["enabled"] is False
    # computed fields are not serialized
    assert "media_metadata" not in m["dataset"]
    assert "excluded_count" not in m["dataset"]


def test_write_export_zip_includes_manifest_and_files_excludes_caches(tmp_path):
    root = tmp_path / "Portraits"
    (root / "masks").mkdir(parents=True)
    (root / ".cache").mkdir()
    (root / ".thumbnails").mkdir()
    (root / "a.jpg").write_bytes(b"img-a")
    (root / "a.txt").write_text("caption a", encoding="utf-8")
    (root / "masks" / "a.png").write_bytes(b"mask-a")
    (root / ".cache" / "latents.bin").write_bytes(b"NOPE")
    (root / ".thumbnails" / "a.webp").write_bytes(b"NOPE")

    manifest = build_manifest(_make_dataset(), app_version="0.4.0-alpha")
    buf = write_export_zip(root, manifest)

    with zipfile.ZipFile(buf) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "a.jpg" in names
        assert "a.txt" in names
        assert "masks/a.png" in names
        # caches excluded
        assert not any(n.startswith(".cache/") for n in names)
        assert not any(n.startswith(".thumbnails/") for n in names)
        loaded = json.loads(zf.read("manifest.json"))
        assert loaded["dataset"]["name"] == "Portraits"


def test_write_export_zip_to_path_matches_in_memory_and_excludes_caches(tmp_path):
    """W4.T11: the disk-streaming variant (used to embed a dataset into a
    project bundle without RAM-buffering it) must exclude caches exactly like
    the in-memory write_export_zip."""
    root = tmp_path / "Portraits"
    (root / "masks").mkdir(parents=True)
    (root / ".cache").mkdir()
    (root / "a.jpg").write_bytes(b"img-a")
    (root / "masks" / "a.png").write_bytes(b"mask-a")
    (root / ".cache" / "latents.bin").write_bytes(b"NOPE")

    manifest = build_manifest(_make_dataset(), app_version="0.4.0-alpha")
    dest = tmp_path / "out.zip"
    write_export_zip_to_path(dest, root, manifest)

    assert dest.is_file()
    with zipfile.ZipFile(dest) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "a.jpg" in names
        assert "masks/a.png" in names
        assert not any(n.startswith(".cache/") for n in names)
        loaded = json.loads(zf.read("manifest.json"))
    assert loaded["dataset"]["name"] == "Portraits"


def test_read_manifest_rejects_missing_and_future_version(tmp_path):
    # missing manifest.json
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.jpg", b"img")
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        with pytest.raises(ManifestError):
            read_manifest(zf)

    # unsupported future format_version
    buf2 = io.BytesIO()
    with zipfile.ZipFile(buf2, "w") as zf:
        zf.writestr("manifest.json", json.dumps(
            {"format_version": MANIFEST_VERSION + 1, "kind": "dataset",
             "dataset": {}, "media": {}}))
    buf2.seek(0)
    with zipfile.ZipFile(buf2) as zf:
        with pytest.raises(ManifestError):
            read_manifest(zf)


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
    # manifest.json is metadata, not a dataset file — must not land on disk
    assert not (dest / "manifest.json").exists()
