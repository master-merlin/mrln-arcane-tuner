"""Portable template archive: build/parse template manifests (``kind='template'``).

A template archive is a manifest-only ``.zip`` carrying 1..N templates across
the training/captioning/masking domains. Training entries embed their model
definition (a ``ModelDefinition`` dump) so the training recipe travels with the
template. Captioning/masking entries reference a built-in ``model_id`` and carry
no embedded model.

Pure: no DB, no registry, no FastAPI. Single source of truth for the template
archive shape, fully unit-testable.
"""

from __future__ import annotations

import re
import zipfile
from typing import Any

from app.core.portable import envelope as _envelope
from app.core.portable.envelope import ManifestError, build_manifest_header

# Bump only on a breaking change to the template manifest shape.
MANIFEST_VERSION = 1
_KIND = "template"

DOMAINS = ("training", "captioning", "masking")

# Fields carried per domain. Everything else on a template row (id, project_id,
# created_at/updated_at, used_count, last_used_at, branched_from, is_default,
# readonly) is machine-specific and dropped.
_CARRY: dict[str, tuple[str, ...]] = {
    "training": ("name", "definition_id", "config"),
    "captioning": ("name", "model_id", "system_prompt", "wildcard", "config"),
    "masking": ("name", "model_id", "config"),
}

# Matches a URI scheme prefix like ``s3://``, ``https://`` — any remote source.
_URI_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://")
# Non-URI remote shorthands used in component paths.
_REMOTE_SHORTHANDS = ("huggingface:", "hf:")


def build_template_entry(
    domain: str, row: dict[str, Any], definition: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build one carried template entry from a DB row.

    For ``training``, *definition* (a ``ModelDefinition`` dump) is embedded when
    provided so the recipe travels with the template.
    """
    if domain not in DOMAINS:
        raise ValueError(f"Unknown template domain: {domain!r}")
    entry: dict[str, Any] = {"domain": domain}
    for field in _CARRY[domain]:
        entry[field] = row.get(field)
    if domain == "training" and definition is not None:
        entry["definition"] = definition
    return entry


def build_template_manifest(
    entries: list[dict[str, Any]], app_version: str
) -> dict[str, Any]:
    """Wrap carried entries in a template manifest with the common header."""
    manifest = build_manifest_header(_KIND, MANIFEST_VERSION, app_version)
    manifest["templates"] = list(entries)
    return manifest


def read_template_manifest(zf: zipfile.ZipFile) -> dict[str, Any]:
    """Read + validate a template manifest. Requires a non-empty ``templates``
    list whose entries each declare a known ``domain``."""
    manifest = _envelope.read_manifest(
        zf, expected_kind=_KIND, max_version=MANIFEST_VERSION
    )
    templates = manifest.get("templates")
    if not isinstance(templates, list) or not templates:
        raise ManifestError("Template archive contains no templates.")
    for entry in templates:
        if not isinstance(entry, dict) or entry.get("domain") not in DOMAINS:
            raise ManifestError(f"Invalid template entry: {entry!r}")
    manifest["templates"] = templates
    return manifest


def is_local_component_path(path: Any) -> bool:
    """True if a component path points at the local filesystem (not a remote
    source). Remote = a ``scheme://`` URI or the ``huggingface:``/``hf:``
    shorthands."""
    if not isinstance(path, str) or not path:
        return False
    lowered = path.lower()
    if lowered.startswith(_REMOTE_SHORTHANDS):
        return False
    if _URI_SCHEME_RE.match(lowered):
        return False
    return True


def scan_local_component_paths(
    definition: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """Return ``[{component, path}]`` for components with local filesystem paths.

    Accepts both the dict component form (``{"path": ...}``) and the shorthand
    string form.
    """
    out: list[dict[str, str]] = []
    components = (definition or {}).get("components") or {}
    for name, comp in components.items():
        path = comp.get("path") if isinstance(comp, dict) else comp
        if is_local_component_path(path):
            out.append({"component": name, "path": path})
    return out
