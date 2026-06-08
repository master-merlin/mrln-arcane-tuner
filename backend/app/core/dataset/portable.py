"""Portable dataset archive: dataset-specific manifest on the shared envelope.

The dataset manifest carries every ``datasets``-row field (except
machine-specific/computed ones) under ``dataset`` and per-image metadata under
``media``. The envelope (``kind``/``format_version``/safety) and the zip
build/extract come from :mod:`app.core.portable`.

Public API (``build_manifest``, ``write_export_zip``, ``read_manifest``,
``safe_extract``, ``MANIFEST_VERSION``, ``ManifestError``) is preserved for
existing callers.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

from app.core.dataset_manager import Dataset
from app.core.portable import envelope as _envelope
from app.core.portable.archive import safe_extract, write_zip
from app.core.portable.envelope import ManifestError, build_manifest_header

# Bump only on a breaking change to the dataset manifest shape.
MANIFEST_VERSION = 1
_KIND = "dataset"

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

__all__ = [
    "MANIFEST_VERSION",
    "ManifestError",
    "build_manifest",
    "write_export_zip",
    "read_manifest",
    "safe_extract",
]


def build_manifest(dataset: Dataset, app_version: str) -> dict[str, Any]:
    """Serialize a ``Dataset`` into a dataset manifest dict (``kind='dataset'``)."""
    data = dataset.model_dump()
    media = data.pop("media_metadata", {}) or {}
    for field in _DROP_DATASET_FIELDS:
        data.pop(field, None)
    manifest = build_manifest_header(_KIND, MANIFEST_VERSION, app_version)
    manifest["dataset"] = data
    manifest["media"] = media
    return manifest


def write_export_zip(dataset_root: Path, manifest: dict[str, Any]) -> io.BytesIO:
    """Build the export zip: ``manifest.json`` first, then files (no caches)."""
    return write_zip(dataset_root, manifest, skip_dirs=_EXCLUDED_DIRS)


def read_manifest(zf: zipfile.ZipFile) -> dict[str, Any]:
    """Read + validate a dataset manifest; default ``dataset``/``media`` keys."""
    manifest = _envelope.read_manifest(
        zf, expected_kind=_KIND, max_version=MANIFEST_VERSION
    )
    manifest.setdefault("dataset", {})
    manifest.setdefault("media", {})
    return manifest
