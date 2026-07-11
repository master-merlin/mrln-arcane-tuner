"""Regression guard: EVERY family definition SHIPS its LoRA target list.

Root cause being guarded against (dreamlite 2026-07-08 GPU-UAT crash):
definitions whose YAML ships NO ``lora_targetable_modules`` — an empty
``[]`` counts, the guard in ``registry.enrich_definition`` is
``if not defn.lora_targetable_modules`` — get the field auto-filled and
PERSISTED at first real model load with ``ModelIntrospector._find_lora_targets``
output: EVERY Linear in the model (time embedders, input/output projections,
text blocks, token refiners, ...). Family drivers prefer a non-empty
definition list over their curated pattern defaults, so after the first load
training silently targets far more modules than the tested surface, breaking
pinned key counts and portability expectations at GPU UAT.

The fix (per family) is to ship the curated list in the YAML; the per-family
portability/definition suites pin the exact contents. THIS test only guards
the shared invariant — non-empty for every definition of EVERY registered
family — so a future definition can't reintroduce the exposure.

W2-C (2026-07-11): extended from the original new-families + flux1 tuples to
the full registered-family set. The ``test_guarded_set_matches_registered``
assertion below fails loudly if a new family is added without being swept into
``ALL_FAMILIES`` — so the guard can never silently fall behind the registry.
"""

from __future__ import annotations

import pytest

from app.engine.models.registry import ModelRegistry

# Every family registered in the engine. Kept as an explicit tuple (rather than
# derived from the registry) so that adding a NEW family forces a conscious
# edit here — ``test_guarded_set_matches_registered`` cross-checks it against
# the live registry and fails if the two drift.
ALL_FAMILIES = (
    "boogu_image",
    "dreamlite",
    "ernie_image",
    "flux1",
    "flux2",
    "hidream_o1",
    "hunyuan_video15",
    "ideogram4",
    "kandinsky5",
    "krea2",
    "longcat_image",
    "ltx2",
    "microsoft_lens",
    "ovis_image",
    "prx",
    "prx_pixel",
    "qwen_image",
    "sdxl",
    "wan21",
    "wan22",
    "zimage",
)


@pytest.fixture()
def registry():
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._paths = {}
    ModelRegistry._discovered = False
    ModelRegistry._definitions_loaded = False
    r = ModelRegistry()
    r.initialize()
    yield r
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._paths = {}
    ModelRegistry._discovered = False
    ModelRegistry._definitions_loaded = False


def test_guarded_set_matches_registered(registry):
    """``ALL_FAMILIES`` must equal the set of families the registry loads.

    This is the anti-forgetting assertion: a new family (its ``family.py``
    discovered, its definition YAML loaded) that is NOT added to
    ``ALL_FAMILIES`` makes this fail — forcing the author to sweep it into the
    LoRA target-list guard below rather than shipping a definition that could
    silently get its target surface clobbered by ``enrich_definition``.
    """
    registered = {d.family for d in ModelRegistry._definitions.values()}
    guarded = set(ALL_FAMILIES)
    assert guarded == registered, (
        "LoRA target-list guard drifted from the registered-family set. "
        f"in ALL_FAMILIES but not registered: {sorted(guarded - registered)}; "
        f"registered but NOT guarded: {sorted(registered - guarded)}. "
        "Add the new family to ALL_FAMILIES and ensure every one of its "
        "definition YAMLs ships a curated non-empty lora_targetable_modules."
    )


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_every_definition_ships_nonempty_lora_target_list(registry, family):
    defs = [d for d in ModelRegistry._definitions.values() if d.family == family]
    assert defs, f"no definitions registered for family {family!r}"
    for defn in defs:
        shipped = getattr(defn, "lora_targetable_modules", None) or []
        assert len(shipped) > 0, (
            f"{defn.id}: lora_targetable_modules is empty — "
            "registry.enrich_definition would auto-fill it with the "
            "introspector's exhaustive Linear catalog at first model load, "
            "overriding the driver's curated targets (dreamlite 2026-07-08 "
            "precedent). Ship the curated list in the definition YAML."
        )
