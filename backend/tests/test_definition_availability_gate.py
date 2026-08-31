"""LANE-45 / DECISION-26 (a): the ``unavailable_reason`` availability gate.

A definition that carries a non-empty ``unavailable_reason`` must not reach any
USER-FACING enumeration, and must be refused at the job seam — while staying
fully present in the internal registry, because the registry-wide coverage
sweeps (VRAM entry per family, LoRA target lists, TE-loading contracts,
``resolve_capabilities``) are the guards that make ungating safe later.

Every enumeration test carries a POSITIVE CONTROL asserting a normal definition
IS present, so a broken enumerator that returns nothing cannot pass.

Origin: job ``5677403c`` was created, queued and started against
``minimax-h3-t2va`` and failed 90s in with "minimax_h3 training lands in PR1;
PR0 ships the scaffold only." (``families/minimax_h3/trainer.py:39``).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.job_manager import job_manager
from app.engine.core.definitions import ModelDefinition
from app.engine.models.registry import registry

# The three PR0 scaffold definitions gated for this release (ECOSYSTEM §6:
# family `minimax_h3`; definitions `minimax-h3-t2va`, `minimax-h3-fl2va`,
# `minimax-h3-ref2va`).
GATED_IDS = frozenset(
    {"minimax-h3-t2va", "minimax-h3-fl2va", "minimax-h3-ref2va"}
)
GATED_FAMILY = "minimax_h3"


@pytest.fixture(scope="module")
def control_id() -> str:
    """A real, shipped, NON-gated definition id — the positive control.

    Derived from the registry rather than hardcoded so the control cannot rot
    when a definition is renamed; asserted non-gated so it can never silently
    become a second gated id and turn the control into a tautology.
    """
    registry.initialize()
    available = sorted(
        did
        for did, defn in registry._definitions.items()
        if not getattr(defn, "unavailable_reason", None)
    )
    assert available, "no available definitions at all — registry did not load"
    control = available[0]
    assert control not in GATED_IDS
    return control


@pytest.fixture(scope="module")
def discovered_plugins() -> None:
    """Plugin discovery runs in the app lifespan, which the TestClient fixture
    does not execute — without this the schema route 404s and the enum test
    would pass vacuously. Setup only: the seam under test (``enrich_schema``
    behind ``GET /api/plugins/{id}/schema``) is the real one."""
    from app.core.plugin_manager import plugin_manager

    if not plugin_manager.get_plugin("standard"):
        plugin_manager.discover_plugins()
    assert plugin_manager.get_plugin("standard") is not None


@pytest.fixture(scope="module")
def control_family(control_id: str) -> str:
    defn = registry.get_definition(control_id)
    assert defn is not None
    return defn.family


# ── The schema key itself ────────────────────────────────────────────────


def test_unavailable_reason_defaults_to_available() -> None:
    """Default must be "visible" so no existing definition changes meaning."""
    defn = ModelDefinition(id="__probe", family="sdxl", name="Probe")
    assert defn.unavailable_reason is None


def test_all_three_minimax_definitions_carry_a_reason() -> None:
    """The gate is declarative and states WHY — an empty gate is not a gate."""
    registry.initialize()
    for did in sorted(GATED_IDS):
        defn = registry.get_definition(did)
        assert defn is not None, f"{did} vanished from the registry entirely"
        reason = defn.unavailable_reason
        assert reason, f"{did} is not gated (unavailable_reason={reason!r})"
        assert len(reason) > 20, f"{did}'s reason is not an honest message: {reason!r}"


def test_the_three_reasons_are_distinct_per_definition() -> None:
    """PER-DEFINITION, not one family-level string.

    The three definitions are gated for different reasons and clear on
    different events: t2va is the base case (the trainer is unbuilt) and clears
    with it; fl2va additionally is not a first/last-frame model at all; ref2va's
    reference path is not merely unbuilt but unanswered (the reference encoder
    is closed-source), so it may stay gated after the other two clear. A single
    shared string cannot express that and would have to be rewritten the moment
    one of them clears — so pin that they stay distinct.
    """
    registry.initialize()
    reasons = {
        did: registry.get_definition(did).unavailable_reason for did in GATED_IDS
    }
    assert len(set(reasons.values())) == 3, (
        "the three gate strings collapsed into a shared family-level message: "
        f"{reasons}"
    )


def test_each_reason_is_honest_about_its_own_definition() -> None:
    """DECISION-26's finding: presenting fl2va as a first/last-frame model is
    worse than not shipping it. ``minimax_h3_fl2va.yaml`` declares
    ``mode: both`` with a PR1 marker and NOTHING behind it — no conditioning-
    frame injection, no dedicated parameters — while ``minimax_h3_t2va.yaml:42``
    honestly declares ``mode: t2v``. The user-visible string must say so.
    """
    registry.initialize()

    fl2va = registry.get_definition("minimax-h3-fl2va").unavailable_reason.lower()
    assert "first" in fl2va and "last" in fl2va, (
        "fl2va's reason must name first/last-frame conditioning as the missing "
        f"thing: {fl2va!r}"
    )
    assert "ignore" in fl2va or "not a first" in fl2va, (
        "fl2va's reason must say the frames would be IGNORED, not merely that "
        f"something is unfinished: {fl2va!r}"
    )

    ref2va = registry.get_definition("minimax-h3-ref2va").unavailable_reason.lower()
    assert "reference" in ref2va, f"ref2va's reason must name it: {ref2va!r}"

    # The base case must NOT claim a conditioning defect it does not have:
    # t2va's `mode: t2v` is accurate, so its only blocker is the trainer.
    t2va = registry.get_definition("minimax-h3-t2va").unavailable_reason.lower()
    assert "first" not in t2va and "reference" not in t2va, (
        f"t2va's reason describes a defect it does not have: {t2va!r}"
    )
    assert "train" in t2va


def test_gates_clear_independently_of_each_other() -> None:
    """Ungating one definition must not ungate its siblings — the PR1 concept
    clears t2va ALONE while fl2va and ref2va may stay gated permanently."""
    registry.initialize()
    t2va = registry.get_definition("minimax-h3-t2va")
    original = t2va.unavailable_reason
    try:
        t2va.unavailable_reason = None
        available = set(registry.list_available_models())
        assert "minimax-h3-t2va" in available, "clearing one gate did not take effect"
        assert "minimax-h3-fl2va" not in available, "clearing t2va leaked fl2va"
        assert "minimax-h3-ref2va" not in available, "clearing t2va leaked ref2va"
    finally:
        t2va.unavailable_reason = original
    assert set(registry.list_available_models()) & GATED_IDS == set()


def test_no_other_shipped_definition_is_gated(control_id: str) -> None:
    """Blast radius: exactly the three, nothing else."""
    registry.initialize()
    gated = {
        did
        for did, defn in registry._definitions.items()
        if getattr(defn, "unavailable_reason", None)
    }
    assert gated == set(GATED_IDS), f"unexpected gating: {gated ^ set(GATED_IDS)}"


# ── Internal registries stay complete (the coverage sweeps must keep passing) ──


def test_registry_internals_still_enumerate_the_gated_family() -> None:
    """``_definitions`` / ``list_models()`` are the INTERNAL view and must keep
    every definition: the registry-wide coverage tables enumerate them to catch
    a family that misses a surface. A gate that empties those tables destroys
    the guard that makes ungating safe."""
    registry.initialize()
    all_ids = set(registry.list_models())
    assert GATED_IDS <= all_ids, f"gate leaked into the registry: {GATED_IDS - all_ids}"
    assert GATED_IDS <= set(registry._definitions)
    # The family class is still registered and constructible.
    assert registry.get_family_class(GATED_FAMILY) is not None


def test_available_accessors_are_the_gated_view(control_id: str) -> None:
    registry.initialize()
    available = set(registry.list_available_models())
    assert not (available & GATED_IDS), "gated ids leaked into list_available_models"
    assert control_id in available, "positive control missing — accessor returns nothing?"
    assert set(registry.available_definitions()) == available
    assert registry.is_definition_available(control_id) is True
    for did in GATED_IDS:
        assert registry.is_definition_available(did) is False


# ── User-facing enumeration surface 1: GET /api/models/definitions ────────


def test_models_definitions_route_hides_gated(
    client: TestClient, control_id: str
) -> None:
    resp = client.get("/api/models/definitions")
    assert resp.status_code == 200
    ids = {d["id"] for d in resp.json()}
    assert control_id in ids, "positive control absent — enumerator returned nothing"
    assert not (ids & GATED_IDS), f"gated definitions leaked: {ids & GATED_IDS}"


# ── User-facing enumeration surface 2: GET /api/caption-context/definitions ──


def test_caption_context_definitions_route_hides_gated(
    client: TestClient, control_id: str
) -> None:
    resp = client.get("/api/caption-context/definitions")
    assert resp.status_code == 200
    ids = {d["id"] for d in resp.json()}
    assert control_id in ids, "positive control absent — enumerator returned nothing"
    assert not (ids & GATED_IDS), f"gated definitions leaked: {ids & GATED_IDS}"


# ── User-facing enumeration surface 3: the training form's schema enums ───


def test_plugin_schema_enums_hide_gated(
    client: TestClient, control_id: str, control_family: str, discovered_plugins: None
) -> None:
    """``TrainingPlugin.enrich_schema`` builds the model picker: the
    ``definition_id`` enum, its ``enum_labels``, its ``backend_map`` (family ->
    definitions) and ``edit_map``, plus the ``model_family`` enum."""
    resp = client.get("/api/plugins/standard/schema")
    assert resp.status_code == 200
    props = resp.json()["properties"]

    defs_prop = props["definition_id"]
    enum = defs_prop["enum"]
    assert control_id in enum, "positive control absent — enricher returned nothing"
    assert not (set(enum) & GATED_IDS), f"gated ids in definition_id enum: {enum}"
    # enum_labels is positional: a filter applied to one list and not the other
    # would mislabel every entry after the first gated one.
    assert len(defs_prop["enum_labels"]) == len(enum)
    assert not (set(defs_prop["edit_map"]) & GATED_IDS)
    backend_map = defs_prop["backend_map"]
    assert GATED_FAMILY not in backend_map, "gated family in the picker's backend_map"
    assert control_family in backend_map
    assert control_id in backend_map[control_family]
    assert defs_prop["default"] not in GATED_IDS

    fam_prop = props["model_family"]
    assert control_family in fam_prop["enum"], "positive control family absent"
    assert GATED_FAMILY not in fam_prop["enum"], "gated family in model_family enum"
    assert len(fam_prop["enum_labels"]) == len(fam_prop["enum"])
    assert fam_prop["default"] != GATED_FAMILY


# ── Belt and braces: the job seam refuses, it is not merely hidden ────────


def test_job_guard_refuses_gated_and_passes_control(control_id: str) -> None:
    """The shared guard: raises for a gated id, silent for a normal one."""
    registry.initialize()
    assert job_manager._require_available_definition(control_id) is None
    assert job_manager._require_available_definition(None) is None
    assert job_manager._require_available_definition("__no_such_definition__") is None
    for did in sorted(GATED_IDS):
        with pytest.raises(ValueError) as exc:
            job_manager._require_available_definition(did)
        assert did in str(exc.value)
        assert "PR1" in str(exc.value) or "cannot be trained" in str(exc.value)


def test_create_job_route_refuses_gated_definition(client: TestClient) -> None:
    """A hidden-but-reachable endpoint is how this comes back: POST /api/jobs
    must answer 400 with an honest message, not queue a job that cannot run."""
    before = len(job_manager._jobs)
    resp = client.post(
        "/api/jobs",
        json={
            "plugin_id": "standard",
            "config": {
                "definition_id": "minimax-h3-t2va",
                "lora_name": "__gate_probe",
            },
        },
    )
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert "minimax-h3-t2va" in detail
    assert "PR1" in detail or "cannot be trained" in detail
    assert len(job_manager._jobs) == before, "a refused job was still registered"


def test_start_job_refuses_gated_definition(control_id: str) -> None:
    """``start_job`` is the seam every auto-start path funnels through
    (``advance_queue``, ``restart_job``, crash recovery), and the one that
    triggers the multi-hundred-GB preflight download."""
    from app.core.job_manager import Job

    gated = Job.create("__no_such_plugin__", {"definition_id": "minimax-h3-t2va"})
    control = Job.create("__no_such_plugin__", {"definition_id": control_id})
    job_manager._jobs[gated.id] = gated
    job_manager._jobs[control.id] = control
    try:
        with pytest.raises(ValueError) as exc:
            job_manager.start_job(gated.id)
        assert "minimax-h3-t2va" in str(exc.value)

        # Positive control: the SAME call for a normal definition gets past the
        # availability guard and dies at the next gate (the bogus plugin),
        # proving the guard is a filter and not a blanket refusal.
        with pytest.raises(ValueError) as exc2:
            job_manager.start_job(control.id)
        assert "__no_such_plugin__" in str(exc2.value)
    finally:
        job_manager._jobs.pop(gated.id, None)
        job_manager._jobs.pop(control.id, None)
