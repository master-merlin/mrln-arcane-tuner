"""W1.T5: create_definition must sanitize the client-supplied ``family``
path segment before it is joined onto ``engine/models/families/`` and used
as a write target.

Pre-fix, ``family="../../evil"`` escapes the families directory: the
handler's own ``family_def_dir.mkdir(parents=True, exist_ok=True)`` creates
the traversed directory on disk and a YAML file lands there, with no 400
ever surfacing. The exemplar guard already used for imported templates
(``app/api/training/template_routes.py::_install_definition``) is a
``re.fullmatch(r"[A-Za-z0-9._-]+")`` sanitize check plus a
``validate_path_within`` containment check before any write.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

# Mirrors the real `families_dir` computation in
# app/api/training/definition_routes.py::create_definition.
FAMILIES_DIR = (
    Path(__file__).resolve().parents[1] / "app" / "engine" / "models" / "families"
)


@pytest.fixture()
def cleanup_definition_artifacts():
    """Removes any registry entry / on-disk YAML the test creates, plus any
    directory a traversal write would have escaped to — runs regardless of
    whether the test passes (proving RED) or fails, so a RED run never
    leaves stray files behind in the real source tree.
    """
    created_ids: list[str] = []
    escaped_paths: list[Path] = []
    yield created_ids, escaped_paths

    from app.engine.models.registry import registry

    for def_id in created_ids:
        registry._definitions.pop(def_id, None)
        path = registry._paths.pop(def_id, None)
        if path:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass

    for p in escaped_paths:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)


def test_create_definition_rejects_family_traversal(
    client, cleanup_definition_artifacts
):
    """POST with a traversal ``family`` must 400 and must NOT create any
    directory outside ``engine/models/families/``.
    """
    created_ids, escaped_paths = cleanup_definition_artifacts
    marker_id = "__w1t5_traversal_test"
    created_ids.append(marker_id)

    # families_dir / "../../evil" / "definitions" resolves two levels above
    # families/ (i.e. to engine/evil/definitions) — the traversal target the
    # unfixed handler would mkdir() and write into.
    escaped_dir = FAMILIES_DIR.parent.parent / "evil"
    escaped_paths.append(escaped_dir)

    resp = client.post(
        "/api/models/definitions",
        json={
            "id": marker_id,
            "family": "../../evil",
            "name": "W1T5 Traversal Test",
        },
    )

    assert resp.status_code == 400, (
        f"expected 400 for traversal family, got {resp.status_code}: {resp.text[:300]}"
    )
    assert not escaped_dir.exists(), f"traversal family must not create {escaped_dir}"


def test_create_definition_accepts_valid_family(client, cleanup_definition_artifacts):
    """Sanity check: a well-formed family is unaffected by the new guard."""
    created_ids, _escaped_paths = cleanup_definition_artifacts
    marker_id = "__w1t5_valid_family_test"
    created_ids.append(marker_id)

    resp = client.post(
        "/api/models/definitions",
        json={
            "id": marker_id,
            "family": "sdxl",
            "name": "W1T5 Valid Family Test",
        },
    )

    assert resp.status_code == 200, resp.text
    yaml_path = FAMILIES_DIR / "sdxl" / "definitions" / f"{marker_id}.yaml"
    assert yaml_path.exists()
