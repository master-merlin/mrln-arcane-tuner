"""LANE-45 mechanical guard: no user-facing enumeration may bypass the gate.

Two halves.

1. **Parity.** Every user-facing enumeration surface must offer exactly
   ``registry.list_available_models()`` — the same set, not merely "a set that
   happens to exclude minimax today". A new gated definition, or a new surface
   wired to the raw registry, breaks this immediately.

2. **A source scan.** ``registry._definitions`` / ``registry.list_models()`` are
   the INTERNAL, ungated view. Every read of them inside the layers a user can
   reach (``app/api/**`` and ``TrainingPlugin.enrich_schema``) must be on the
   allowlist below WITH a reason saying why that read is not user-facing
   enumeration. The recurrence this fires on: someone adds a fourth definitions
   route, iterates the raw dict because that is what the neighbouring code did,
   and a definition that cannot train becomes selectable again.

Copies the ``_GPU_SERVICES`` coverage pattern from LANE-42, including the
vacuity control — this has the collect-offenders shape, so it must prove the
scanner is not blind before it reports "no offenders".
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.engine.models.registry import registry

# ── 1. Parity ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def available_ids() -> set[str]:
    registry.initialize()
    ids = set(registry.list_available_models())
    assert ids, "registry offers no definitions at all — parity would be vacuous"
    return ids


def test_models_definitions_route_offers_exactly_the_available_set(
    client: TestClient, available_ids: set[str]
) -> None:
    got = {d["id"] for d in client.get("/api/models/definitions").json()}
    assert got == available_ids, f"drift: {got ^ available_ids}"


def test_caption_context_route_offers_exactly_the_available_set(
    client: TestClient, available_ids: set[str]
) -> None:
    got = {d["id"] for d in client.get("/api/caption-context/definitions").json()}
    assert got == available_ids, f"drift: {got ^ available_ids}"


def test_training_schema_offers_exactly_the_available_set(
    client: TestClient, available_ids: set[str]
) -> None:
    from app.core.plugin_manager import plugin_manager

    if not plugin_manager.get_plugin("standard"):
        plugin_manager.discover_plugins()
    props = client.get("/api/plugins/standard/schema").json()["properties"]
    got = set(props["definition_id"]["enum"])
    assert got == available_ids, f"drift: {got ^ available_ids}"
    # The family enum is derived from the same pass and must agree with it.
    expected_families = {
        registry.get_definition(d).family for d in available_ids
    }
    assert set(props["model_family"]["enum"]) == expected_families


# ── 2. Source scan ───────────────────────────────────────────────────────

_BACKEND = Path(__file__).resolve().parents[1]
_RAW_READ = re.compile(r"registry\._definitions|registry\.list_models\(\)")

_SCANNED: tuple[Path, ...] = (
    _BACKEND / "app" / "api",
    _BACKEND / "app" / "engine" / "models" / "base.py",
)

# "<path relative to backend/> : <line text, stripped>" -> why this raw read is
# NOT user-facing enumeration. A row is evidence, not a waiver: if you cannot
# write the reason, the call site wants `available_definitions()`.
_ALLOWED_RAW_READS: dict[str, str] = {
    "app/api/project_routes.py::registry._definitions.pop(definition_id, None)": (
        "Undo of a project import — unregisters ONE explicitly named id. Must "
        "reach gated definitions too, or an imported-then-gated definition "
        "would be un-uninstallable."
    ),
    "app/api/training/definition_routes.py::del registry._definitions[definition_id]": (
        "DELETE /models/definitions/{id} — operates on ONE explicitly named id, "
        "and deleting a gated definition must stay possible."
    ),
    "app/api/training/definition_routes.py::defn = registry._definitions.get(request.definition_id)": (
        "VRAM estimate for ONE explicitly named id. Deliberately NOT gated: it "
        "is a pure read that returns a truthful number and costs nothing, and "
        "the definition is already unreachable from the picker. Gating it "
        "would conflate 'not offered' with 'not answerable'."
    ),
    "app/api/training/template_routes.py::for did in registry.list_models():": (
        "_find_family_hf_source — internal component-source resolution during "
        "template import, never rendered. Must see gated definitions: a "
        "template importing a gated family still needs its HF component path."
    ),
}


def _scan_raw_reads() -> dict[str, str]:
    """Map 'relpath::line' -> line for every raw registry read in the scanned tree."""
    found: dict[str, str] = {}
    files: list[Path] = []
    for target in _SCANNED:
        files.extend(sorted(target.rglob("*.py")) if target.is_dir() else [target])
    for path in files:
        if "tests" in path.parts:
            continue
        rel = path.relative_to(_BACKEND).as_posix()
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if _RAW_READ.search(stripped) and not stripped.startswith("#"):
                found[f"{rel}::{stripped}"] = stripped
    return found


def test_scanner_is_not_blind() -> None:
    """Vacuity control: a check with the collect-offenders shape passes when it
    scans nothing, so prove it sees the known-good raw reads first."""
    found = _scan_raw_reads()
    assert len(found) >= 4, (
        f"the scan found {len(found)} raw registry read(s), expected at least 4 "
        "— the check is blind, not satisfied"
    )


def test_every_raw_registry_read_in_a_user_reachable_layer_is_justified() -> None:
    found = _scan_raw_reads()
    unjustified = sorted(set(found) - set(_ALLOWED_RAW_READS))
    assert not unjustified, (
        "raw (ungated) registry reads in a user-reachable layer with no stated "
        "reason — use registry.available_definitions() / list_available_models() "
        f"for anything the user chooses from, or add a row with evidence: {unjustified}"
    )


def test_allowlist_has_no_stale_rows() -> None:
    """A row that no longer matches any source line is a comment describing a
    check the code does not make — delete it or fix it."""
    found = _scan_raw_reads()
    stale = sorted(set(_ALLOWED_RAW_READS) - set(found))
    assert not stale, f"allowlist rows matching nothing in the tree: {stale}"


def test_every_allowlist_row_states_a_reason() -> None:
    for key, reason in _ALLOWED_RAW_READS.items():
        assert len(reason) > 40, f"{key} carries no real justification: {reason!r}"
