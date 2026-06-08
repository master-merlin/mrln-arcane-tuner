"""Shared portable-archive envelope: common manifest header + validation.

A portable archive (dataset, template, or project) is a ``.zip`` whose first
entry is ``manifest.json``. Every manifest shares a common header so a dropped
archive is self-describing via its ``kind``. Each kind owns an independent
``format_version``.

Pure: no DB, no FastAPI. Single source of truth for the envelope format.
"""

from __future__ import annotations

import json
import time
import zipfile
from typing import Any

MANIFEST_NAME = "manifest.json"


class ManifestError(Exception):
    """Raised when an archive is missing/invalid, the wrong kind, or unsafe."""


def build_manifest_header(
    kind: str, format_version: int, app_version: str
) -> dict[str, Any]:
    """Build the common manifest header. Callers add their kind-specific keys."""
    return {
        "format_version": format_version,
        "app_version": app_version,
        "exported_at": time.time(),
        "kind": kind,
    }


def read_manifest(
    zf: zipfile.ZipFile, *, expected_kind: str, max_version: int
) -> dict[str, Any]:
    """Read and validate ``manifest.json`` from an open archive.

    Validates, in order: presence, JSON-object shape, ``format_version`` not
    newer than *max_version*, and that ``kind`` equals *expected_kind*. Raises
    ``ManifestError`` on any failure.
    """
    if MANIFEST_NAME not in zf.namelist():
        raise ManifestError(
            "Archive is missing manifest.json — not an exported archive."
        )
    try:
        manifest = json.loads(zf.read(MANIFEST_NAME))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ManifestError(f"manifest.json is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ManifestError("manifest.json must be a JSON object.")
    version = manifest.get("format_version")
    if not isinstance(version, int) or version > max_version:
        raise ManifestError(
            f"Unsupported manifest format_version {version!r}; "
            f"this build supports up to {max_version}."
        )
    kind = manifest.get("kind")
    if kind != expected_kind:
        raise ManifestError(
            f"Archive kind {kind!r} does not match expected {expected_kind!r}."
        )
    return manifest


def peek_manifest(zf: zipfile.ZipFile) -> dict[str, Any]:
    """Read ``manifest.json`` and return its header (``kind`` + versions) WITHOUT
    enforcing a specific kind. Used by the generic import-peek route so the UI
    can route a dropped archive to the right importer.
    """
    if MANIFEST_NAME not in zf.namelist():
        raise ManifestError(
            "Archive is missing manifest.json — not an exported archive."
        )
    try:
        manifest = json.loads(zf.read(MANIFEST_NAME))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ManifestError(f"manifest.json is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict) or not manifest.get("kind"):
        raise ManifestError("manifest.json is missing a 'kind'.")
    return {
        "kind": manifest.get("kind"),
        "format_version": manifest.get("format_version"),
        "app_version": manifest.get("app_version"),
    }
