"""Regression guard: every new-family definition SHIPS its LoRA target list.

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
the shared invariant — non-empty for every definition of the seven new
families — so a future definition can't reintroduce the exposure.
"""

from __future__ import annotations

import pytest

from app.engine.models.registry import ModelRegistry

NEW_FAMILIES = (
    "ovis_image",
    "longcat_image",
    "prx",
    "prx_pixel",
    "dreamlite",
    "hunyuan_video15",
    "kandinsky5",
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


@pytest.mark.parametrize("family", NEW_FAMILIES)
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
