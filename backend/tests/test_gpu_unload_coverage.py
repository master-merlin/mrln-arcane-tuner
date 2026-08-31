"""The GPU-unload control must enumerate EVERY GPU-plugin service.

LANE-42 shipped a topbar control that frees the three services which hold GPU
models. The recurrence this file exists to stop is the one CLAUDE.md already
names for model families: **a fourth service is added, and the surface that
ENUMERATES them silently misses it.** The symptom would be invisible — the
button works, the toast says it freed something, and one service's VRAM stays
resident forever with nothing in the product able to report it.

So the enumeration is not allowed to be hand-kept. `unload_gpu_plugins` is the
one primitive that releases a GPU plugin dict; anything that calls it is by
definition a GPU-plugin service, and this test derives that set from the source
and requires `system_routes._GPU_SERVICES` to cover it.

It also pins the two properties a registered row must have, because a row that
is present but wrong fails in exactly the same silent way:

* the class really has the active-key attribute the row names (a typo'd
  attribute makes `getattr(..., None)` return `None` forever, so the service
  reports "nothing loaded" no matter what is resident);
* `unload_models` accepts `skip_if_batch_active`, so the route cannot free a
  model out from under a running batch.

This has the collect-offenders shape, so it carries a vacuity control: a
scanner that finds nothing passes for the wrong reason.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from app.api import system_routes

# Anchored on this file, never on the CWD (D10): pytest is run from the repo
# root and from `backend/` alike.
_APP_ROOT = Path(__file__).resolve().parents[1] / "app"

_CALL_RE = re.compile(r"^\s*unload_gpu_plugins\(", re.MULTILINE)


def _modules_calling_unload_gpu_plugins() -> set[str]:
    """Dotted module paths under `app/` that CALL the shared unload primitive.

    Source-level rather than import-level on purpose: importing every module
    under `app/` to look for callers would drag in the whole plugin stack and
    make this guard as expensive as the thing it guards.
    """
    found: set[str] = set()
    for path in _APP_ROOT.rglob("*.py"):
        if path.name == "gpu_unload.py":  # the definition, not a caller
            continue
        text = path.read_text(encoding="utf-8")
        if _CALL_RE.search(text):
            rel = path.relative_to(_APP_ROOT.parent).with_suffix("")
            found.add(".".join(rel.parts))
    return found


def test_the_scan_is_not_blind():
    """Vacuity control: the scanner must find the callers that exist today.

    Without this, a regex that stopped matching would make every assertion
    below pass over an empty set — the check would be blind, not satisfied.
    """
    found = _modules_calling_unload_gpu_plugins()
    assert len(found) >= 3, (
        f"unload_gpu_plugins caller scan found {len(found)} module(s), expected at "
        "least 3 (caption, masking, scoring) — the check is blind, not satisfied"
    )


def test_every_gpu_plugin_service_is_registered_in_the_unload_route():
    """A service that releases GPU plugins but is not enumerated here can never
    be freed by the user, and nothing in the product would say so."""
    registered = {module for _id, _label, module, _cls, _attr in system_routes._GPU_SERVICES}
    missing = _modules_calling_unload_gpu_plugins() - registered
    assert not missing, (
        "GPU-plugin service(s) not enumerated in system_routes._GPU_SERVICES: "
        f"{sorted(missing)} — the 'free GPU memory' control would silently skip them"
    )


@pytest.mark.parametrize(
    "service_id,label,module_path,class_name,active_attr",
    system_routes._GPU_SERVICES,
    ids=[row[0] for row in system_routes._GPU_SERVICES],
)
def test_a_registered_row_names_a_real_active_key_attribute(
    service_id, label, module_path, class_name, active_attr
):
    cls = system_routes._gpu_service_class(module_path, class_name)
    assert hasattr(cls, active_attr), (
        f"{class_name}.{active_attr} does not exist — residency for '{service_id}' "
        "would read None forever and the control would never appear"
    )


@pytest.mark.parametrize(
    "service_id,label,module_path,class_name,active_attr",
    system_routes._GPU_SERVICES,
    ids=[row[0] for row in system_routes._GPU_SERVICES],
)
def test_a_registered_service_supports_the_batch_guarded_unload(
    service_id, label, module_path, class_name, active_attr
):
    cls = system_routes._gpu_service_class(module_path, class_name)
    sig = inspect.signature(cls.unload_models)
    param = sig.parameters.get("skip_if_batch_active")
    assert param is not None, (
        f"{class_name}.unload_models has no skip_if_batch_active parameter — the "
        "global unload could free its model out from under a running batch"
    )
    assert param.kind is inspect.Parameter.KEYWORD_ONLY, (
        "skip_if_batch_active must be keyword-only, so no caller enables the "
        "no-op mode by positional accident"
    )
    assert param.default is False, (
        "the guarded mode must be OPT-IN: internal callers (a batch unloading "
        "its own model) must never silently no-op"
    )
