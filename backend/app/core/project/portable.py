"""Portable project archive: build/parse project manifests (``kind='project'``).

A project archive is a ``.zip`` whose ``manifest.json`` carries project
metadata + preferences and references nested ``templates/<slug>.zip`` and
``datasets/<slug>.zip`` payloads. This module owns the manifest shape and a
filename slug helper — it does NOT touch the DB, the registry, or FastAPI; the
nested payload bytes are produced/consumed by the route layer.
"""

from __future__ import annotations

import zipfile
from typing import Any

from app.core.portable import envelope as _envelope
from app.core.portable.envelope import build_manifest_header

MANIFEST_VERSION = 1
_KIND = "project"

# Project + preference fields that are machine-specific and never carried.
_DROP_PROJECT = ("id", "created_at", "updated_at")
_DROP_PREFS = ("id", "project_id")


def slugify(name: str | None) -> str:
    """ASCII-safe slug for a nested archive filename. Falls back to 'project'."""
    cleaned = "".join(
        ("_" if c in (" ", "/", "\\") else c)
        for c in (name or "")
        if c.isascii() and (c.isalnum() or c in (" ", "/", "\\", "-", "_"))
    ).strip("_")
    return cleaned or "project"


def _clean(d: dict[str, Any], drop: tuple[str, ...]) -> dict[str, Any]:
    return {k: v for k, v in (d or {}).items() if k not in drop}


def build_project_manifest(
    project: dict[str, Any],
    preferences: dict[str, Any],
    templates: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    app_version: str,
) -> dict[str, Any]:
    """Assemble the project manifest. *templates*/*datasets* are reference
    entries (each pointing at a nested archive path or, for referenced
    datasets, just a name)."""
    manifest = build_manifest_header(_KIND, MANIFEST_VERSION, app_version)
    proj = _clean(project, _DROP_PROJECT)
    proj["preferences"] = _clean(preferences, _DROP_PREFS)
    manifest["project"] = proj
    manifest["templates"] = list(templates)
    manifest["datasets"] = list(datasets)
    return manifest


def read_project_manifest(zf: zipfile.ZipFile) -> dict[str, Any]:
    """Read + validate a project manifest; default the structural keys."""
    manifest = _envelope.read_manifest(
        zf, expected_kind=_KIND, max_version=MANIFEST_VERSION
    )
    manifest.setdefault("project", {})
    manifest["project"].setdefault("preferences", {})
    manifest.setdefault("templates", [])
    manifest.setdefault("datasets", [])
    return manifest
