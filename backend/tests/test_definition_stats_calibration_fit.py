"""The end-of-job stats recompute passes no fit verdict (LANE-92 real-job run).

A completed int8 job ended its log with ``vram_estimate peak_mb=74581
available_mb=56350 fits=false``. Mechanism: the trainer's end-of-job
``definition_stats_service.recompute`` calls ``_analytic_vram`` for the
calibration ratios (measured / analytic, component by component); the estimator
then judged ``fits`` against the device's FREE memory while the finishing job
itself still held the card. The verdict is meaningless there and nothing reads
it — only the analytic breakdown is used.

Production caller: ``definition_stats_service._aggregate`` ->
``_analytic_vram``. The GPU is faked at ``SystemMonitor._gpu_snapshot`` (the
device, not the seam): a 96 GB card with 41 GB held.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from structlog.testing import capture_logs

from app.core.stats import definition_stats_service as svc
from app.core.system_monitor import SystemMonitor
from app.engine.models.registry import registry
from app.engine.utils.vram_estimator import VRAMEstimator

_DEF = "sdxl_base_1.0"
_CONFIG = {"quantization": "none"}
_COMPONENTS = ("model_weights_mb", "activations_mb", "caching_peak_mb", "training_peak_mb", "peak_mb")


@pytest.fixture(scope="module", autouse=True)
def _loaded_registry():
    registry.discover_families()
    registry.load_definitions("app/engine/models/definitions")
    return registry


@pytest.fixture
def card_held_by_the_job(monkeypatch):
    """Free memory far below any estimate: every fit check says `fits=false`."""
    gpu = SimpleNamespace(vram_total_mb=97887, vram_used_mb=97000)
    monkeypatch.setattr(SystemMonitor, "_gpu_snapshot", lambda self: [gpu])


def _estimate_lines(logs: list[dict]) -> list[dict]:
    return [e for e in logs if e.get("event") == "vram_estimate"]


def test_the_calibration_estimate_carries_no_fit_verdict(card_held_by_the_job):
    defn = registry._definitions[_DEF]
    with capture_logs() as logs:
        analytic = svc._analytic_vram(defn, dict(_CONFIG))
    assert analytic is not None
    assert analytic["fit_known"] is False, "a fit verdict was computed against the finishing job's own memory"
    lines = _estimate_lines(logs)
    assert len(lines) == 1
    assert lines[0]["fits"] is None, f"the log still states a verdict: {lines[0]}"
    assert not any("free" in w.lower() for w in analytic["warnings"]), analytic["warnings"]


def test_the_analytic_breakdown_the_ratios_divide_by_is_unchanged(card_held_by_the_job):
    defn = registry._definitions[_DEF]
    analytic = svc._analytic_vram(defn, dict(_CONFIG))
    preflight = VRAMEstimator.estimate(defn, dict(_CONFIG)).to_dict()
    assert {k: analytic[k] for k in _COMPONENTS} == {k: preflight[k] for k in _COMPONENTS}


def test_the_preflight_estimate_still_judges_fit_against_free_memory(card_held_by_the_job):
    """Positive control: the estimate the UI asks for keeps its verdict."""
    defn = registry._definitions[_DEF]
    with capture_logs() as logs:
        report = VRAMEstimator.estimate(defn, dict(_CONFIG)).to_dict()
    assert report["fit_known"] is True and report["fits"] is False
    assert report["available_mb"] == 887
    assert _estimate_lines(logs)[0]["fits"] is False
