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
PR0 ships the scaffold only." (the PR0 ``families/minimax_h3/trainer.py``).

LANE-92 rows 5.0/5.1: the PR0 world ("all three gated") is replaced by the
EXPLICIT MATRIX below. Every test is parameterised from the matrix's two sets;
nothing else in this file names an id as "the gated one" or "the available
one", so flipping one cell leaves no contradicting literal behind.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.job_manager import job_manager
from app.engine.core.definitions import ModelDefinition
from app.engine.models.registry import registry

H3_FAMILY = "minimax_h3"

# The three FROZEN ids (ECOSYSTEM §6) — a literal, never derived.
H3_ALL_IDS = frozenset({"minimax-h3-t2va", "minimax-h3-fl2va", "minimax-h3-ref2va"})

# The matrix. ``None`` = available; a string = gated, the gist of why (the
# shipped wording lives in the YAML and is pinned by REASON_MUST_MENTION).
EXPECTED_AVAILABILITY: dict[str, str | None] = {
    # available: GATE-0..3 `status=pass`, the LoRA trains and saves (LANE-92 It2-It4)
    "minimax-h3-t2va": None,
    # gated: first/last-frame conditioning rows not yet wired (It6 -> PR1b / LANE-99)
    "minimax-h3-fl2va": "first/last-frame conditioning rows not yet wired (It6)",
    # gated: reference packing path unbuilt + its own fine-tune ungated
    "minimax-h3-ref2va": "reference-block packing path unbuilt; checkpoint ungated",
}
assert set(EXPECTED_AVAILABILITY) == H3_ALL_IDS  # every frozen id, no more, no fewer
GATED_IDS = frozenset(k for k, v in EXPECTED_AVAILABILITY.items() if v)
AVAILABLE_H3_IDS = H3_ALL_IDS - GATED_IDS

# Row 4.4's `Verdict:` line, copied verbatim from
# `_harness/research/minimax-h3-context-ir.md:19` (ONE producer of the fact).
CONTEXT_IR_VERDICT = "NOT REQUIRED"
CONTEXT_IR_VERDICTS = ("REQUIRED", "NOT REQUIRED", "UNDETERMINED")

H3_DEFINITIONS_DIR = (
    Path(__file__).resolve().parents[1]
    / "app/engine/models/families/minimax_h3/definitions"
)


def ref2va_reason_rules(verdict: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(must mention, must not mention) for ref2va's reason under a Context-IR
    verdict. The must-not list pins OUT the two claims the shipped PR0 string
    made and the research report falsified: that a needed component is "not
    publicly available", and (since It2) that training is "not built yet"."""
    must = ("reference", "packing")
    must_not = ("not publicly available", "not built yet either")
    if verdict == "REQUIRED":
        # A required separate component is NAMED; `encoder` is allowed.
        return must + ("H3-Context-IR",), must_not
    if verdict == "UNDETERMINED":
        # The string says the question is open and claims nothing.
        return must + ("not settled",), must_not + ("encoder",)
    # NOT REQUIRED: a packing path over the same transformer, no encoder.
    return must, must_not + ("encoder",)


REASON_MUST_MENTION: dict[str, tuple[str, ...]] = {
    # "ignore": DECISION-26 — the string must say the frames would be IGNORED,
    # not merely that something is unfinished.
    "minimax-h3-fl2va": ("first", "last", "It6", "ignore"),
    "minimax-h3-ref2va": ref2va_reason_rules(CONTEXT_IR_VERDICT)[0],
}
REASON_MUST_NOT_MENTION: dict[str, tuple[str, ...]] = {
    # Training IS built since It2 — the PR0 clause is false for every sibling.
    "minimax-h3-fl2va": ("not built yet either",),
    "minimax-h3-ref2va": ref2va_reason_rules(CONTEXT_IR_VERDICT)[1],
}
# A stale entry for an id that became available fails at collection.
assert set(REASON_MUST_MENTION) == GATED_IDS
assert set(REASON_MUST_NOT_MENTION) <= GATED_IDS


def _flat(text: str) -> str:
    return " ".join(text.lower().split())


def _check_reason(reason: str, must: tuple[str, ...], must_not: tuple[str, ...]) -> None:
    lowered = _flat(reason)
    # The falsified claims first, so a regression to a PR0 string is NAMED.
    for token in must_not:
        assert token.lower() not in lowered, (
            f"the reason still makes the falsified claim {token!r}: {reason!r}"
        )
    for token in must:
        assert token.lower() in lowered, f"the reason omits {token!r}: {reason!r}"


@pytest.fixture(scope="module")
def control_id() -> str:
    """A real, shipped, NON-gated, non-H3 definition id — the positive control.

    Derived from the registry rather than hardcoded so the control cannot rot
    when a definition is renamed; asserted outside the matrix so it can never
    silently become one of the ids under test and turn into a tautology.
    """
    registry.initialize()
    available = sorted(
        did
        for did, defn in registry._definitions.items()
        if not getattr(defn, "unavailable_reason", None) and did not in H3_ALL_IDS
    )
    assert available, "no available definitions at all — registry did not load"
    return available[0]


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
    assert defn.family != H3_FAMILY
    return defn.family


# ── The schema key itself ────────────────────────────────────────────────


def test_unavailable_reason_defaults_to_available() -> None:
    """Default must be "visible" so no existing definition changes meaning."""
    defn = ModelDefinition(id="__probe", family="sdxl", name="Probe")
    assert defn.unavailable_reason is None


def test_the_matrix_is_not_vacuous() -> None:
    """Both arms of every parameterised test below must have a population at
    It5: one available id and two gated ones. (Row 6.3 flips fl2va and edits
    this count in the same change.)"""
    assert len(AVAILABLE_H3_IDS) == 1 and len(GATED_IDS) == 2


@pytest.mark.parametrize("did", sorted(GATED_IDS))
def test_gated_definition_carries_a_reason(did: str) -> None:
    """The gate is declarative and states WHY — an empty gate is not a gate."""
    registry.initialize()
    defn = registry.get_definition(did)
    assert defn is not None, f"{did} vanished from the registry entirely"
    reason = defn.unavailable_reason
    assert reason, f"{did} is not gated (unavailable_reason={reason!r})"
    assert len(reason) > 20, f"{did}'s reason is not an honest message: {reason!r}"


@pytest.mark.parametrize("did", sorted(AVAILABLE_H3_IDS))
def test_available_definition_carries_no_reason(did: str) -> None:
    """The POSITIVE arm: the ungating happened, per available id."""
    registry.initialize()
    defn = registry.get_definition(did)
    assert defn is not None, f"{did} vanished from the registry entirely"
    assert not defn.unavailable_reason, (
        f"{did} is still gated: {defn.unavailable_reason!r}"
    )


def test_the_reasons_are_distinct_per_definition() -> None:
    """PER-DEFINITION, not one family-level string: the gated definitions are
    gated for different reasons and clear on different events (fl2va on its
    first/last-frame conditioning rows, ref2va on its reference-block packing
    path and its own checkpoint's gate). A single shared string cannot express
    that and would have to be rewritten the moment one of them clears."""
    registry.initialize()
    if len(GATED_IDS) < 2:
        pytest.skip("one gated id left — distinctness has no population")
    reasons = {
        did: registry.get_definition(did).unavailable_reason for did in GATED_IDS
    }
    assert len(set(reasons.values())) == len(GATED_IDS), (
        f"the gate strings collapsed into a shared family-level message: {reasons}"
    )


@pytest.mark.parametrize("did", sorted(GATED_IDS))
def test_each_reason_is_honest_about_its_own_definition(did: str) -> None:
    """DECISION-26's finding: presenting fl2va as a first/last-frame model is
    worse than not shipping it, and REQUEST-13's: a false reason is worse than
    none. Honesty is asserted only for definitions that still carry a reason."""
    registry.initialize()
    reason = registry.get_definition(did).unavailable_reason
    assert reason
    _check_reason(reason, REASON_MUST_MENTION[did], REASON_MUST_NOT_MENTION.get(did, ()))


def test_ref2va_reason_is_true_today() -> None:
    """REQUEST-13: ref2va's PR0 reason was FALSE (it blamed an unpublished
    "reference encoder" and an unbuilt trainer). Pin the MEANING of the true
    one: what is missing is the reference-block PACKING path in this app and a
    gate for the ref2va checkpoint — nothing upstream withholds."""
    registry.initialize()
    reason = registry.get_definition("minimax-h3-ref2va").unavailable_reason
    assert reason
    _check_reason(reason, *ref2va_reason_rules(CONTEXT_IR_VERDICT))
    lowered = _flat(reason)
    # WHAT is missing, and where: unbuilt HERE, not unpublished upstream.
    assert "is not built" in lowered, reason
    assert "published" in lowered, "must say the model components ARE published"
    assert "has not been validated by a gate" in lowered, reason
    # The honest alternative is named, and it is the one that trains.
    assert "minimax-h3-t2va" in lowered, reason


def test_context_ir_verdict_is_one_of_three() -> None:
    assert CONTEXT_IR_VERDICT in CONTEXT_IR_VERDICTS


@pytest.mark.parametrize("verdict", CONTEXT_IR_VERDICTS)
def test_ref2va_reason_rules_over_every_verdict(verdict: str) -> None:
    """The rule itself, over all three verdicts — so a reason row 5.0 permits
    can never fail row 5.1, whichever way row 4.4 had come out."""
    must, must_not = ref2va_reason_rules(verdict)
    assert {"reference", "packing"} <= set(must)
    assert {"not publicly available", "not built yet either"} <= set(must_not)
    if verdict == "REQUIRED":
        assert "H3-Context-IR" in must
        assert "encoder" not in must_not
    else:
        assert "H3-Context-IR" not in must
        assert "encoder" in must_not
    assert ("not settled" in must) == (verdict == "UNDETERMINED")


def test_ref2va_yaml_comments_point_at_the_reason_instead_of_claiming() -> None:
    """RULE-21, ONE producer: the gating fact lives in ``unavailable_reason``
    and the research report; the YAML comments point there. The two greps are
    the shapes the four falsified comments had."""
    text = (H3_DEFINITIONS_DIR / "minimax_h3_ref2va.yaml").read_text(encoding="utf-8")
    assert text.count("closed-source") == 0
    assert text.count("H3-Context-IR is") == 0
    assert "minimax-h3-context-ir.md" in text, "the comments no longer point at the report"


@pytest.mark.parametrize("did", sorted(AVAILABLE_H3_IDS))
def test_available_definition_promises_no_conditioning_it_lacks(did: str) -> None:
    """The UI-facing texts of an offered definition (its name, its description
    when it has one) must not promise reference or first/last-frame
    conditioning: only plain text-to-video(+audio) went through the gates."""
    registry.initialize()
    defn = registry.get_definition(did)
    shown = _flat(f"{defn.name} {getattr(defn, 'description', '') or ''}")
    for promise in ("reference", "first", "last frame", "image"):
        assert promise not in shown, f"{did} promises {promise!r}: {shown!r}"
    assert defn.architecture_params.get("mode") == "t2v"


@pytest.mark.parametrize("did", sorted(GATED_IDS))
def test_gates_clear_independently_of_each_other(did: str) -> None:
    """Clearing ONE gate must take effect for that id and change nothing for
    any other: every OTHER gated id stays gated, every available id stays
    available."""
    registry.initialize()
    defn = registry.get_definition(did)
    original = defn.unavailable_reason
    try:
        defn.unavailable_reason = None
        available = set(registry.list_available_models())
        assert did in available, "clearing one gate did not take effect"
        assert not (available & (GATED_IDS - {did})), "clearing one gate leaked a sibling"
        assert AVAILABLE_H3_IDS <= available
    finally:
        defn.unavailable_reason = original
    assert set(registry.list_available_models()) & GATED_IDS == set()


def test_no_other_shipped_definition_is_gated(control_id: str) -> None:
    """Blast radius: exactly the matrix's gated set, nothing else."""
    registry.initialize()
    gated = {
        did
        for did, defn in registry._definitions.items()
        if getattr(defn, "unavailable_reason", None)
    }
    assert gated == set(GATED_IDS), f"unexpected gating: {gated ^ set(GATED_IDS)}"


# ── Internal registries stay complete (the coverage sweeps must keep passing) ──


def test_registry_internals_still_enumerate_the_whole_family() -> None:
    """``_definitions`` / ``list_models()`` are the INTERNAL view and must keep
    every definition: the registry-wide coverage tables enumerate them to catch
    a family that misses a surface. All three FROZEN ids, regardless of the
    matrix — shrinking ``GATED_IDS`` must not delete this coverage."""
    registry.initialize()
    all_ids = set(registry.list_models())
    assert H3_ALL_IDS <= all_ids, f"gate leaked into the registry: {H3_ALL_IDS - all_ids}"
    assert H3_ALL_IDS <= set(registry._definitions)
    # The family class is still registered and constructible.
    assert registry.get_family_class(H3_FAMILY) is not None


def test_available_accessors_are_the_gated_view(control_id: str) -> None:
    registry.initialize()
    available = set(registry.list_available_models())
    assert not (available & GATED_IDS), "gated ids leaked into list_available_models"
    assert AVAILABLE_H3_IDS <= available, (
        f"ungated ids missing: {AVAILABLE_H3_IDS - available}"
    )
    assert control_id in available, "positive control missing — accessor returns nothing?"
    assert set(registry.available_definitions()) == available
    assert registry.is_definition_available(control_id) is True
    for did in GATED_IDS:
        assert registry.is_definition_available(did) is False
    for did in AVAILABLE_H3_IDS:
        assert registry.is_definition_available(did) is True


# ── User-facing enumeration surface 1: GET /api/models/definitions ────────


def _assert_surface(ids: set[str], control_id: str, surface: str) -> None:
    assert control_id in ids, f"{surface}: positive control absent — enumerator returned nothing"
    assert not (ids & GATED_IDS), f"{surface}: gated definitions leaked: {ids & GATED_IDS}"
    assert AVAILABLE_H3_IDS <= ids, (
        f"{surface}: ungated definitions missing: {AVAILABLE_H3_IDS - ids}"
    )


def test_models_definitions_route_serves_exactly_the_available(
    client: TestClient, control_id: str
) -> None:
    registry.initialize()
    resp = client.get("/api/models/definitions")
    assert resp.status_code == 200
    _assert_surface({d["id"] for d in resp.json()}, control_id, "models/definitions")


# ── User-facing enumeration surface 2: GET /api/caption-context/definitions ──


def test_caption_context_definitions_route_serves_exactly_the_available(
    client: TestClient, control_id: str
) -> None:
    registry.initialize()
    resp = client.get("/api/caption-context/definitions")
    assert resp.status_code == 200
    _assert_surface(
        {d["id"] for d in resp.json()}, control_id, "caption-context/definitions"
    )


# ── User-facing enumeration surface 3: the training form's schema enums ───


def test_plugin_schema_enums_serve_exactly_the_available(
    client: TestClient, control_id: str, control_family: str, discovered_plugins: None
) -> None:
    """``TrainingPlugin.enrich_schema`` builds the model picker: the
    ``definition_id`` enum, its ``enum_labels``, its ``backend_map`` (family ->
    definitions) and ``edit_map``, plus the ``model_family`` enum."""
    registry.initialize()
    resp = client.get("/api/plugins/standard/schema")
    assert resp.status_code == 200
    props = resp.json()["properties"]

    defs_prop = props["definition_id"]
    enum = defs_prop["enum"]
    _assert_surface(set(enum), control_id, "definition_id enum")
    # enum_labels is positional: a filter applied to one list and not the other
    # would mislabel every entry after the first gated one.
    assert len(defs_prop["enum_labels"]) == len(enum)
    assert not (set(defs_prop["edit_map"]) & GATED_IDS)
    backend_map = defs_prop["backend_map"]
    # The family reached the picker with EXACTLY the available ids (derived,
    # no ordering claim) — or not at all when none is available.
    h3_in_map = backend_map.get(H3_FAMILY, [])
    assert set(h3_in_map) == AVAILABLE_H3_IDS
    assert len(h3_in_map) == len(AVAILABLE_H3_IDS)
    assert (H3_FAMILY in backend_map) == bool(AVAILABLE_H3_IDS)
    assert control_family in backend_map
    assert control_id in backend_map[control_family]
    assert defs_prop["default"] not in GATED_IDS

    fam_prop = props["model_family"]
    assert control_family in fam_prop["enum"], "positive control family absent"
    assert (H3_FAMILY in fam_prop["enum"]) == bool(AVAILABLE_H3_IDS)
    assert len(fam_prop["enum_labels"]) == len(fam_prop["enum"])


# ── Belt and braces: the job seam refuses, it is not merely hidden ────────


def test_job_guard_passes_control_and_unknown(control_id: str) -> None:
    registry.initialize()
    assert job_manager._require_available_definition(control_id) is None
    assert job_manager._require_available_definition(None) is None
    assert job_manager._require_available_definition("__no_such_definition__") is None


@pytest.mark.parametrize("did", sorted(GATED_IDS))
def test_job_guard_refuses_gated(did: str) -> None:
    registry.initialize()
    with pytest.raises(ValueError) as exc:
        job_manager._require_available_definition(did)
    assert did in str(exc.value)
    assert "cannot be trained" in str(exc.value)


@pytest.mark.parametrize("did", sorted(AVAILABLE_H3_IDS))
def test_job_guard_admits_available(did: str) -> None:
    registry.initialize()
    assert job_manager._require_available_definition(did) is None


@pytest.mark.parametrize("did", sorted(GATED_IDS))
def test_create_job_route_refuses_gated_definition(client: TestClient, did: str) -> None:
    """A hidden-but-reachable endpoint is how this comes back: POST /api/jobs
    must answer 400 with an honest message, not queue a job that cannot run."""
    registry.initialize()
    before = len(job_manager._jobs)
    resp = client.post(
        "/api/jobs",
        json={
            "plugin_id": "standard",
            "config": {"definition_id": did, "lora_name": "__gate_probe"},
        },
    )
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert did in detail
    assert "cannot be trained" in detail
    assert len(job_manager._jobs) == before, "a refused job was still registered"


@pytest.mark.parametrize("did", sorted(AVAILABLE_H3_IDS))
def test_create_job_route_admits_available_definition(client: TestClient, did: str) -> None:
    """Admission, proven without leaving a job behind: the request carries a
    frame count that breaks the family's ``17n+5`` rule, so it is refused by
    the VIDEO CONTRACT — the step right after the availability guard in
    ``create_job``. Whatever fails, fails later than the gate."""
    registry.initialize()
    before = len(job_manager._jobs)
    resp = client.post(
        "/api/jobs",
        json={
            "plugin_id": "standard",
            "config": {
                "definition_id": did,
                "lora_name": "__gate_probe",
                "num_frames": 98,  # 98 % 17 == 13, not 5
            },
        },
    )
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert "cannot be trained" not in detail, detail
    assert "17n+5" in detail, f"not the video contract's refusal: {detail!r}"
    assert len(job_manager._jobs) == before, "the probe left a job behind"


def _start_with_bogus_plugin(did: str) -> pytest.ExceptionInfo:
    from app.core.job_manager import Job

    job = Job.create("__no_such_plugin__", {"definition_id": did})
    job_manager._jobs[job.id] = job
    try:
        with pytest.raises(ValueError) as exc:
            job_manager.start_job(job.id)
    finally:
        job_manager._jobs.pop(job.id, None)
    return exc


@pytest.mark.parametrize("did", sorted(GATED_IDS))
def test_start_job_refuses_gated_definition(did: str, control_id: str) -> None:
    """``start_job`` is the seam every auto-start path funnels through
    (``advance_queue``, ``restart_job``, crash recovery), and the one that
    triggers the multi-hundred-GB preflight download."""
    registry.initialize()
    message = str(_start_with_bogus_plugin(did).value)
    assert did in message and "cannot be trained" in message

    # Positive control: the SAME call for a normal definition gets past the
    # availability guard and dies at the next gate (the bogus plugin),
    # proving the guard is a filter and not a blanket refusal.
    assert "__no_such_plugin__" in str(_start_with_bogus_plugin(control_id).value)


@pytest.mark.parametrize("did", sorted(AVAILABLE_H3_IDS))
def test_start_job_admits_available_definition(did: str) -> None:
    """An available H3 id dies at the bogus plugin exactly like the control."""
    registry.initialize()
    message = str(_start_with_bogus_plugin(did).value)
    assert "cannot be trained" not in message
    assert "__no_such_plugin__" in message
