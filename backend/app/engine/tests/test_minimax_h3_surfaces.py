"""minimax_h3 — the run-config keys on every enumerating surface (plan row 3.3).

The family's five run-config keys (``train_audio``, ``audio_loss_weight``,
``sigma_shift_video``, ``sigma_shift_audio``, ``cfg_augment_scale``) travel
form -> job config -> DB -> trainer, and a key missing from ONE of the surfaces
below is a knob the user can set that nothing honours (or a knob nothing lets
the user set). Every key is checked on every surface — 5 x 6 — with the same
predicate, and every predicate is proven non-vacuous inside the test: a key
known present (``learning_rate``) passes it and a key that exists nowhere
fails it.

Surfaces (plan § Configuration contract, "Surfaces every key must appear on"):

1. ``schema``            — ``BaseTrainingConfig.model_json_schema()`` properties
2. ``field_visibility``  — ``resolve_capabilities(defn)["field_visibility"]``:
                            shown for H3, hidden for ``flux1-dev``
3. ``allowlist``         — ``apply_capability_allowlist`` keeps the key for H3,
                            drops it for the image family
4. ``help_text``         — ``frontend/public/config_help.json`` (the ``?`` tooltip
                            the docs generator reads, LANE-72)
5. ``template_round_trip`` — ``portable.build_template_entry`` carries the value
                            through the JSON archive unchanged
6. ``job_config_round_trip`` — the typed config (``model_validate`` ->
                            ``model_dump`` -> JSON) preserves the value AND the
                            omitted case resolves to the schema default
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.core.template import portable
from app.engine.core.archetypes import resolve_capabilities
from app.engine.core.config_allowlist import apply_capability_allowlist
from app.engine.models.base import BaseTrainingConfig
from app.engine.models.registry import ModelRegistry

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_HELP = _REPO_ROOT / "frontend" / "public" / "config_help.json"

H3_ID = "minimax-h3-t2va"
IMAGE_ID = "flux1-dev"  # is_video False, has_audio False — every H3 key hidden
CONTROL_KEY = "learning_rate"  # universal: present on every surface
ABSENT_KEY = "zzz_key_that_exists_on_no_surface"

# key -> a NON-default value the round-trip surfaces must carry unchanged.
KEYS: dict[str, Any] = {
    "train_audio": False,
    "audio_loss_weight": 0.5,
    "sigma_shift_video": 7.0,
    "sigma_shift_audio": 2.0,
    "cfg_augment_scale": 2.0,
}
_MINIMAL_DATASETS = [{"dataset_name": "ds"}]


@pytest.fixture(scope="module")
def registry():
    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    return ModelRegistry


def _defn(registry, def_id: str):
    d = registry._definitions.get(def_id)
    assert d is not None, f"definition {def_id} not loaded"
    return d


# ── One predicate per surface: (key, value) -> None or a failure reason ──


def _schema(key: str, value: Any, registry) -> str | None:
    props = BaseTrainingConfig.model_json_schema()["properties"]
    return None if key in props else f"{key!r} is not a BaseTrainingConfig field"


def _field_visibility(key: str, value: Any, registry) -> str | None:
    h3 = resolve_capabilities(_defn(registry, H3_ID))["field_visibility"]
    image = resolve_capabilities(_defn(registry, IMAGE_ID))["field_visibility"]
    if key == CONTROL_KEY:
        # A universal key is NOT in the gated map on purpose (shown everywhere).
        return None if key not in h3 and key not in image else f"{key!r} gated"
    if key not in h3:
        return f"{key!r} has no _FIELD_RULES row (absent from field_visibility)"
    if not h3[key]["supported"]:
        return f"{key!r} hidden for {H3_ID}: {h3[key].get('reason')}"
    if image.get(key, {}).get("supported", True):
        return f"{key!r} is SHOWN for {IMAGE_ID} (must be gated off)"
    return None


def _allowlist(key: str, value: Any, registry) -> str | None:
    kept = {key: value, CONTROL_KEY: 1e-4}
    dropped = apply_capability_allowlist(kept, _defn(registry, H3_ID))
    if key not in kept or dropped:
        return f"{key!r} dropped for {H3_ID} (dropped={dropped})"
    if key == CONTROL_KEY:
        return None
    image_cfg = {key: value}
    dropped = apply_capability_allowlist(image_cfg, _defn(registry, IMAGE_ID))
    if key in image_cfg or dropped != [key]:
        return f"{key!r} survives on {IMAGE_ID} (dropped={dropped})"
    return None


def _help_text(key: str, value: Any, registry) -> str | None:
    help_map = json.loads(_CONFIG_HELP.read_text(encoding="utf-8"))
    entry = help_map.get(key)
    if not isinstance(entry, dict):
        return f"{key!r} has no entry in {_CONFIG_HELP.name}"
    if not str(entry.get("tip", "")).strip() or not str(entry.get("detail", "")).strip():
        return f"{key!r} help entry lacks a tip or a detail"
    return None


def _template_round_trip(key: str, value: Any, registry) -> str | None:
    defn = _defn(registry, H3_ID)
    row = {"name": "t", "definition_id": H3_ID, "config": {key: value}}
    entry = portable.build_template_entry("training", row, defn.model_dump())
    back = json.loads(json.dumps(entry))["config"]
    if back.get(key, "<missing>") != value:
        return f"{key!r} did not survive the template archive: {back}"
    # Import side: the carried config is applied to a job on this definition —
    # the allowlist must keep the key and the typed config must accept it.
    dropped = apply_capability_allowlist(back, defn)
    if dropped:
        return f"{key!r} dropped by the allowlist on template import: {dropped}"
    if key not in BaseTrainingConfig.model_fields:
        return f"{key!r} is not a typed field — an imported template would lose it"
    typed = BaseTrainingConfig.model_validate({**back, "datasets": _MINIMAL_DATASETS})
    if typed.model_dump().get(key, "<missing>") != value:
        return f"{key!r} lost by the typed config after import"
    return None


def _job_config_round_trip(key: str, value: Any, registry) -> str | None:
    if key not in BaseTrainingConfig.model_fields:
        return f"{key!r} is not a typed config field"
    typed = BaseTrainingConfig.model_validate({key: value, "datasets": _MINIMAL_DATASETS})
    back = json.loads(typed.model_dump_json())
    if back.get(key, "<missing>") != value:
        return f"{key!r} explicit value lost through the typed job config: {back.get(key)}"
    omitted = BaseTrainingConfig.model_validate({"datasets": _MINIMAL_DATASETS})
    default = BaseTrainingConfig.model_fields[key].default
    if getattr(omitted, key) != default:
        return f"{key!r} omitted resolves to {getattr(omitted, key)!r}, schema default {default!r}"
    return None


SURFACES = {
    "schema": _schema,
    "field_visibility": _field_visibility,
    "allowlist": _allowlist,
    "help_text": _help_text,
    "template_round_trip": _template_round_trip,
    "job_config_round_trip": _job_config_round_trip,
}


@pytest.mark.parametrize("key", sorted(KEYS))
@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_key_on_surface(surface: str, key: str, registry):
    check = SURFACES[surface]
    # Positive control: the predicate accepts a key known to be everywhere.
    assert check(CONTROL_KEY, 1e-4, registry) is None, f"{surface}: control key rejected"
    # Negative control: the predicate can see an absence at all.
    assert check(ABSENT_KEY, 1.0, registry) is not None, f"{surface}: blind to an absent key"
    reason = check(key, KEYS[key], registry)
    assert reason is None, f"surface={surface} key={key}: {reason}"
