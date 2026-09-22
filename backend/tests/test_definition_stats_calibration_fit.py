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

import contextlib
import logging
from types import SimpleNamespace

import pytest
import structlog
from structlog.testing import capture_logs

from app.core.logger import setup_logging
from app.core.stats import definition_stats_service as svc
from app.core.system_monitor import SystemMonitor
from app.engine.models.registry import registry
from app.engine.utils import vram_estimator as _vram_estimator_module
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


def _drop_cached_bind() -> None:
    """Forget the estimator proxy's cached ``bind`` (see below for why).

    A cached bind holds the processor LIST INSTANCE that was configured when it
    was made, so it must be dropped both BEFORE a capture (so the capture's list
    is the one bound) and AFTER any test that reconfigured logging (so the next
    ordinary capture is not stranded on a list nobody configures any more).
    """
    _vram_estimator_module.logger.__dict__.pop("bind", None)


@contextlib.contextmanager
def capture_estimate_logs():
    """``capture_logs`` on its own does not always see ``vram_estimate``.

    ``setup_logging`` configures with ``cache_logger_on_first_use=True``
    (app/core/logger.py:234), so the estimator's module-level proxy
    (app/engine/utils/vram_estimator.py:24) freezes a bound logger — and with
    it the processor LIST INSTANCE that was configured at that moment — the
    first time anything in the process logs (structlog/_config.py:385-389).
    ``capture_logs`` installs itself by refilling the CURRENTLY configured list
    IN PLACE (structlog/testing.py:86-93), whereas ``configure(processors=...)``
    rebinds ``_CONFIG.default_processors`` to a NEW list
    (structlog/_config.py:246-247). So any ``setup_logging`` between the first
    log call and the capture strands the frozen logger on a list the capture
    never touches, and the capture comes back empty — which xdist worker that
    lands in is luck (LANE-111: red on CI, green locally, same commit).

    Dropping the proxy's cached ``bind`` makes it re-bind against whatever list
    is configured now, i.e. the one the capture is holding.
    """
    _drop_cached_bind()
    with capture_logs() as logs:
        yield logs


def test_the_calibration_estimate_carries_no_fit_verdict(card_held_by_the_job):
    defn = registry._definitions[_DEF]
    with capture_estimate_logs() as logs:
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
    with capture_estimate_logs() as logs:
        report = VRAMEstimator.estimate(defn, dict(_CONFIG)).to_dict()
    assert report["fit_known"] is True and report["fits"] is False
    assert report["available_mb"] == 887
    lines = _estimate_lines(logs)
    assert lines, "the vram_estimate event escaped the capture"
    assert lines[0]["fits"] is False


# Every logger whose LEVEL ``setup_logging`` moves: ``config_log_level``
# (app/core/logger.py:146-162, root + uvicorn* + fastapi) and
# ``_quiet_noisy_loggers`` (app/core/logger.py:173-176, the download libraries).
# Restoring only root would leave those thresholds raised for the rest of the
# worker, so a later test asserting on a filelock/urllib3/websockets INFO line
# would read an empty log for no reason of its own.
_LEVELS_SETUP_LOGGING_MOVES = (
    "",
    "uvicorn",
    "uvicorn.access",
    "uvicorn.error",
    "fastapi",
    "filelock",
    "urllib3",
    "hf_xet",
    "websockets",
)
# ``setup_logging`` also empties these three loggers' own handler lists and
# forces ``propagate`` (app/core/logger.py:298-301).
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.access", "uvicorn.error")


@contextlib.contextmanager
def logging_restored_ctx():
    """Undo everything ``setup_logging`` touches, including the cached bind.

    It REPLACES root's handler list (app/core/logger.py:243), moves the level of
    every logger in ``_LEVELS_SETUP_LOGGING_MOVES``, empties the uvicorn
    loggers' handlers, and rebinds structlog's global config — and a test that
    leaves any of those moved would silently change every later test in this
    worker.

    The cached bind is the subtle one. Restoring the config rebinds
    ``_CONFIG.default_processors`` to the ORIGINAL list
    (structlog/_config.py:246-247), while the estimator proxy still holds a bind
    made against the list that was live during the test. The next ORDINARY
    ``capture_logs`` — in any test written before this file, which does no
    bind-dropping of its own — refills the configured list in place
    (structlog/testing.py:86-93) and comes back EMPTY, because the proxy is
    still logging through the abandoned list. So the teardown drops the bind
    too, which is what ``test_an_ordinary_capture_after_this_fixture_still_sees_the_event``
    pins. The proxy repaired is the one this file's tests bind; a test that
    binds another module's proxy under this fixture must drop that one as well.

    Exposed as a context manager (not only as a fixture) so that the teardown
    happens INSIDE a test and can therefore be asserted on.
    """
    config = structlog.get_config()
    root = logging.getLogger()
    handlers = root.handlers[:]
    levels = {name: logging.getLogger(name).level for name in _LEVELS_SETUP_LOGGING_MOVES}
    uvicorn_state = {
        name: (logging.getLogger(name).handlers[:], logging.getLogger(name).propagate)
        for name in _UVICORN_LOGGERS
    }
    try:
        yield
    finally:
        structlog.configure(**config)
        root.handlers = handlers
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)
        for name, (uv_handlers, propagate) in uvicorn_state.items():
            uv_log = logging.getLogger(name)
            uv_log.handlers, uv_log.propagate = uv_handlers, propagate
        _drop_cached_bind()


@pytest.fixture
def logging_restored():
    with logging_restored_ctx():
        yield


def test_the_estimate_log_is_captured_across_a_logging_reconfiguration(
    card_held_by_the_job, logging_restored
):
    """The assertion above reads the log, so the capture must not be luck.

    ``setup_logging`` runs with ``cache_logger_on_first_use=True``
    (app/core/logger.py:234), so the estimator's module logger freezes a bound
    logger — and the processor LIST INSTANCE configured at that moment — the
    first time anything logs in the process (structlog/_config.py:385-389).
    ``capture_logs`` installs itself by refilling the CURRENTLY configured list
    in place (structlog/testing.py:86-93), while ``configure(processors=...)``
    rebinds ``_CONFIG.default_processors`` to a NEW list
    (structlog/_config.py:246-247). A reconfiguration therefore strands the
    capture, and which xdist worker that happens in is luck — which is how
    LANE-111's CI red arose while the same file passed locally.
    """
    defn = registry._definitions[_DEF]
    VRAMEstimator.estimate(defn, dict(_CONFIG))  # freezes the module logger's bind
    setup_logging(include_file_handler=False)  # strands it on the previous list

    with capture_estimate_logs() as logs:
        report = VRAMEstimator.estimate(defn, dict(_CONFIG)).to_dict()

    assert report["fit_known"] is True and report["fits"] is False
    lines = _estimate_lines(logs)
    assert lines, "the vram_estimate event escaped the capture after logging was reconfigured"
    assert lines[0]["fits"] is False


def test_a_reconfiguration_inside_the_capture_window_loses_the_event(
    card_held_by_the_job, logging_restored
):
    """BOUNDARY, not desired behaviour: reconfiguring INSIDE the window is unrepairable.

    The test above reconfigures BEFORE the capture, and dropping the proxy's
    cached ``bind`` repairs that. Reconfiguring AFTER the capture is installed
    cannot be repaired from inside the helper, by construction:
    ``capture_logs`` installs itself by refilling the CURRENTLY configured
    processor list IN PLACE (structlog/testing.py:86-93), while
    ``configure(processors=...)`` rebinds ``_CONFIG.default_processors`` to a
    NEW list (structlog/_config.py:246-247) — so the reconfiguration orphans
    the capture list AFTER it was installed, and nothing done before the
    capture can reach forward to it. Repairing it would mean monkeypatching
    ``structlog.configure`` for the duration of every capture.

    This test therefore PINS the limitation instead of dropping the
    requirement: any test that reconfigures logging mid-capture reads an empty
    log, and that emptiness means nothing about the code under test. If a
    future structlog makes ``capture_logs`` survive a reconfiguration, this
    test goes RED and tells us the limitation lifted.

    Measured 2026-09-22, each order in its own process: reconfigure-before -> 1
    ``vram_estimate`` line, no reconfiguration -> 1 line, reconfigure-inside ->
    0 lines.
    """
    defn = registry._definitions[_DEF]
    VRAMEstimator.estimate(defn, dict(_CONFIG))  # freezes the module logger's bind

    with capture_estimate_logs() as logs:
        setup_logging(include_file_handler=False)  # orphans the capture's list
        report = VRAMEstimator.estimate(defn, dict(_CONFIG)).to_dict()

    # The estimate really ran and really emits this event elsewhere in this
    # file, so the empty capture below is the reconfiguration and not a
    # no-op call.
    assert report["fit_known"] is True and report["fits"] is False
    assert not _estimate_lines(logs), (        "capture_logs survived a reconfiguration inside its own window: the "
        "structlog limitation this test pins has lifted, so the helper and the "
        "docstrings that cite structlog/testing.py:86-93 can be revisited"
    )


def test_an_ordinary_capture_after_this_fixture_still_sees_the_event(card_held_by_the_job):
    """The repair above must not become the very defect this lane removes.

    The population at risk is every capture-using test written before this
    file: it calls ``capture_logs`` plainly and drops no cached bind. If the
    ``logging_restored`` teardown left the estimator proxy bound to the list
    that was live during the reconfiguring test, that later capture would come
    back empty in this xdist worker — an unrelated red, weeks from now, in a
    file nobody touched.

    So this test runs the reconfiguring test's SHAPE inside the fixture's
    context manager, lets the teardown run, and only then performs an ORDINARY
    capture. Measured 2026-09-22 with the bind-drop removed from the teardown:
    the ordinary capture below returns 0 lines.
    """
    defn = registry._definitions[_DEF]

    with logging_restored_ctx():
        VRAMEstimator.estimate(defn, dict(_CONFIG))  # freezes the module logger's bind
        setup_logging(include_file_handler=False)  # strands it on the previous list
        with capture_estimate_logs() as inside:
            VRAMEstimator.estimate(defn, dict(_CONFIG))
        assert _estimate_lines(inside), "the regression test's own shape stopped capturing"

    with capture_logs() as after:  # ORDINARY: no bind-dropping of its own
        VRAMEstimator.estimate(defn, dict(_CONFIG))

    assert _estimate_lines(after), (
        "the logging_restored teardown leaked a stale cached bind: an ordinary "
        "capture_logs after it reads an empty log, so every capture-using test "
        "later in this worker would fail for a reason that is not its own"
    )
