"""Guard: a VRAM estimate's device-derived rows must never decide a gate.

LANE-30. ``VRAMEstimator.estimate`` reads live NVML telemetry on every call
(`app/engine/utils/vram_estimator.py:746`) and copies it into the report, so
two estimates taken milliseconds apart differ whenever another process on the
box allocates or frees VRAM. Two suite tests compared two estimates for
whole-dict equality and therefore also compared two live readings; on
2026-08-29 one went red because ComfyUI's footprint moved 120 MB between the
calls, and the next run passed with no change to any code.

The fix is ``backend/tests/conftest.py::frozen_gpu_snapshot``, which replays
ONE real reading for both calls. These are its controls:

* the NEGATIVE control proves the estimator without the freeze really does
  produce different dicts when the device moves — a "fix" that made the
  comparison unable to notice the device at all would be worse than the flake;
* the POSITIVE control proves the freeze holds a MOVING device still all the
  way through the estimator, not just at the monitor;
* the SURFACE control pins the exact set of report fields the device feeds, so
  a new live-derived field cannot be added without this guard firing and
  sending its author to the fixture.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.engine.models.registry import registry
from app.engine.utils.vram_estimator import VRAMEstimator

_IMAGE_DEF = "sdxl_base_1.0"
_CONFIG = {"quantization": "none"}

# Every ``to_dict`` key whose value comes from the live GPU reading rather than
# from the model + config. ``fit_known`` is deliberately NOT here: it records
# that the query SUCCEEDED, not what it returned, so it is constant across two
# successful reads. Keep this set in step with vram_estimator.py section 9/10.
_LIVE_DERIVED = {"used_mb", "total_mb", "available_mb", "fits", "warnings"}


@pytest.fixture(scope="module", autouse=True)
def _loaded_registry():
    registry.discover_families()
    registry.load_definitions("app/engine/models/definitions")
    return registry


def _gpu(total_mb: int, used_mb: int):
    """A minimal stand-in for one ``GPUStatus`` — the estimator reads two fields."""
    return SimpleNamespace(vram_total_mb=total_mb, vram_used_mb=used_mb)


@pytest.fixture
def moving_gpu(monkeypatch):
    """A device whose free VRAM changes on EVERY read, like a shared box.

    Patched at ``SystemMonitor._gpu_snapshot`` rather than at ``snapshot()`` so
    that ``frozen_gpu_snapshot`` — which calls the real ``snapshot()`` once —
    still exercises its own code path on top of it.
    """
    from app.core import system_monitor as sm

    state = {"n": 0}

    def _moving():
        state["n"] += 1
        # 100 MB of drift per read: the same order as the observed flake
        # (used_mb 50851 vs 50731), and far below any fit boundary.
        return [_gpu(81920, 50000 + state["n"] * 100)]

    monkeypatch.setattr(sm.SystemMonitor, "_gpu_snapshot", staticmethod(_moving))
    return state


def test_unfrozen_estimates_differ_when_the_device_moves(moving_gpu):
    """NEGATIVE control — the flake itself, reproduced deterministically.

    Without the freeze the estimator faithfully reports two different devices,
    which is correct behaviour and exactly why an equality assertion may not
    take two readings.
    """
    defn = registry.get_definition(_IMAGE_DEF)
    assert defn is not None

    first = VRAMEstimator.estimate(defn, _CONFIG).to_dict()
    second = VRAMEstimator.estimate(defn, _CONFIG).to_dict()

    assert first != second
    assert first["used_mb"] != second["used_mb"]
    assert moving_gpu["n"] == 2  # both estimates really did read the device


def test_frozen_snapshot_holds_a_moving_device_still(moving_gpu, frozen_gpu_snapshot):
    """POSITIVE control — same moving device, one reading, identical dicts.

    ``moving_gpu`` is requested FIRST so the freeze is taken on top of it.
    """
    defn = registry.get_definition(_IMAGE_DEF)
    assert defn is not None

    first = VRAMEstimator.estimate(defn, _CONFIG).to_dict()
    second = VRAMEstimator.estimate(defn, _CONFIG).to_dict()

    assert first == second
    # One read total: the fixture's, none from the two estimates.
    assert moving_gpu["n"] == 1


def test_live_derived_report_fields_are_exactly_the_declared_set(monkeypatch):
    """SURFACE control — which report rows the DEVICE owns, asserted, not listed.

    Two crafted devices differing in every observable way (capacity, usage, and
    a fit that flips) must move exactly ``_LIVE_DERIVED`` and nothing else. Add
    a new field fed by the snapshot and this fails; drop one and it fails too.
    """
    from app.core import system_monitor as sm

    defn = registry.get_definition(_IMAGE_DEF)
    assert defn is not None

    roomy = SimpleNamespace(gpus=[_gpu(81920, 2048)])  # 78 GB free → fits
    cramped = SimpleNamespace(gpus=[_gpu(24576, 23552)])  # 1 GB free → does not

    monkeypatch.setattr(sm.system_monitor, "snapshot", lambda: roomy)
    a = VRAMEstimator.estimate(defn, _CONFIG).to_dict()
    monkeypatch.setattr(sm.system_monitor, "snapshot", lambda: cramped)
    b = VRAMEstimator.estimate(defn, _CONFIG).to_dict()

    # The two devices really are distinguishable, including the verdict.
    assert a["fits"] is True and b["fits"] is False
    assert a["fit_known"] is True and b["fit_known"] is True

    differing = {k for k in a if a[k] != b[k]}
    assert differing == _LIVE_DERIVED, differing
