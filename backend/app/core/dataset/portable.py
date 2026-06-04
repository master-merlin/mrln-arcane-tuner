"""Portable dataset archive: manifest (de)serialization + safe zip build/extract.

A portable archive is a ``.zip`` whose first entry is ``manifest.json``
(everything that lives only in SQLite) followed by every on-disk dataset
file except the regenerable ``.cache/`` and ``.thumbnails/`` directories.

This module is pure: no DB access, no FastAPI. It is the single source of
truth for the archive format and is fully unit-testable.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any

from app.core.dataset_manager import Dataset

# Bump only on a breaking change to the manifest shape.
MANIFEST_VERSION = 1

_MANIFEST_NAME = "manifest.json"
_EXCLUDED_DIRS = (".cache", ".thumbnails")
# Dataset model fields that are machine-specific or recomputed — never carried.
# .cache and .thumbnails are excluded from the archive, so cache/scan state
# must reset on the target instance (has_cache, last_scanned_at).
_DROP_DATASET_FIELDS = (
    "id", "path", "media_metadata",
    "excluded_count", "median_quality_score",  # computed_field outputs
    "missing", "updated_at",
    "has_cache", "last_scanned_at",
)


class ManifestError(Exception):
    """Raised when an archive is missing/invalid or unsafe to extract."""


def build_manifest(dataset: Dataset, app_version: str) -> dict[str, Any]:
    """Serialize a ``Dataset`` into a manifest dict.

    Carries every ``datasets``-row field (except machine-specific/computed
    ones) under ``dataset`` and the per-image metadata under ``media``.
    """
    data = dataset.model_dump()
    media = data.pop("media_metadata", {}) or {}
    for field in _DROP_DATASET_FIELDS:
        data.pop(field, None)
    return {
        "format_version": MANIFEST_VERSION,
        "app_version": app_version,
        "exported_at": time.time(),
        "dataset": data,
        "media": media,
    }


def write_export_zip(dataset_root: Path, manifest: dict[str, Any]) -> io.BytesIO:
    """Build the export zip in memory: ``manifest.json`` first, then files.

    Excludes the ``.cache`` and ``.thumbnails`` directories. Mirrors the
    exclusion logic of the legacy ``/download`` endpoint.
    """
    dataset_root = Path(dataset_root)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(_MANIFEST_NAME, json.dumps(manifest, indent=2))
        for file_path in dataset_root.rglob("*"):
            if any(part in _EXCLUDED_DIRS for part in file_path.parts):
                continue
            if file_path.is_file():
                arc_name = file_path.relative_to(dataset_root).as_posix()
                zf.write(file_path, arc_name)
    buf.seek(0)
    return buf


def read_manifest(zf: zipfile.ZipFile) -> dict[str, Any]:
    """Read and validate ``manifest.json`` from an open archive.

    Raises ``ManifestError`` if it is missing, malformed, or declares a
    ``format_version`` newer than this build supports.
    """
    if _MANIFEST_NAME not in zf.namelist():
        raise ManifestError("Archive is missing manifest.json — not an exported dataset.")
    try:
        manifest = json.loads(zf.read(_MANIFEST_NAME))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ManifestError(f"manifest.json is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ManifestError("manifest.json must be a JSON object.")
    version = manifest.get("format_version")
    if not isinstance(version, int) or version > MANIFEST_VERSION:
        raise ManifestError(
            f"Unsupported manifest format_version {version!r}; "
            f"this build supports up to {MANIFEST_VERSION}."
        )
    manifest.setdefault("dataset", {})
    manifest.setdefault("media", {})
    return manifest


def safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract every entry except ``manifest.json`` into *dest*, safely.

    Rejects absolute paths and ``..`` traversal (raises ``ManifestError``).
    """
    dest = Path(dest).resolve()
    for member in zf.infolist():
        name = member.filename
        if name == _MANIFEST_NAME or name.endswith("/"):
            continue
        target = (dest / name).resolve()
        if not target.is_relative_to(dest):
            raise ManifestError(f"Unsafe path in archive: {name!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, open(target, "wb") as out:
            out.write(src.read())
