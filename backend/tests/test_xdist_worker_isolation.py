"""Per-worker isolation of the files the suite writes outside tmp_path (LANE-63).

Two files used to be shared by every pytest process on the box:

* ``backend/server.log`` — ``app/main.py`` calls ``setup_logging`` at import,
  which unlinks and reopens ``logger.SERVER_LOG_PATH``; a test run therefore
  wiped the log of the backend live on this machine (ledgered), and N xdist
  workers would all do it to one file;
* ``backend/tests/tests.log`` — the session fixture in ``tests/conftest.py``
  redirects all logging there; N workers interleaving into it is unreadable and
  on Windows can fail a worker on the open.

The root ``backend/conftest.py`` diverts the first via ``MRLN_SERVER_LOG_PATH``
before any ``app`` import; the fixture suffixes the second. These tests assert
on the OBSERVED paths of this very process (so they hold in serial AND under
every xdist worker), then run a child pytest under an invented worker id and
prove the negative: cut the root conftest out with ``--confcutdir`` and the
same check goes red.
"""
import logging
import os
import subprocess
import sys
from pathlib import Path

from tests.support.worker_paths import WORKER_ENV, current_worker, worker_suffixed

BACKEND = Path(__file__).resolve().parents[1]
TESTS = BACKEND / "tests"
LIVE_SERVER_LOG = BACKEND / "server.log"
LIVE_PREVIOUS_SERVER_LOG = BACKEND / "server.prev.log"


# ── the rule ──────────────────────────────────────────────────────────────


def test_serial_name_is_unchanged_and_worker_name_carries_the_id(monkeypatch):
    # `worker=None` means "this process's worker", so the serial case is the
    # env var ABSENT — not a None argument (under xdist that is the worker id).
    monkeypatch.delenv(WORKER_ENV, raising=False)
    assert worker_suffixed(Path("x/tests.log")) == Path("x/tests.log")
    monkeypatch.setenv(WORKER_ENV, "gw7")
    assert worker_suffixed(Path("x/tests.log")) == Path("x/tests-gw7.log")
    assert worker_suffixed(Path("x/tests.log"), "gw3") == Path("x/tests-gw3.log")
    assert worker_suffixed(Path("x/server-test.log"), "gw12") == Path("x/server-test-gw12.log")


# ── this process, observed ────────────────────────────────────────────────


def test_this_process_never_points_at_the_live_server_log():
    from app.core import logger

    expected = worker_suffixed(TESTS / "server-test.log", current_worker())
    assert logger.SERVER_LOG_PATH == expected, (
        f"SERVER_LOG_PATH is {logger.SERVER_LOG_PATH}; the root conftest should have "
        f"diverted it to {expected} before app.core.logger was imported"
    )
    assert logger.SERVER_LOG_PATH != LIVE_SERVER_LOG


def test_the_rotation_target_follows_the_divert():
    """``setup_logging`` MOVES the current server log onto
    ``PREVIOUS_SERVER_LOG_PATH`` (LANE-56) — so the divert has to carry that
    second path too. Any test process that re-runs ``setup_logging`` (every
    ``importlib.reload(app.main)`` does) would otherwise rotate the LIVE
    backend's ``server.log`` onto ``server.prev.log`` and destroy exactly the
    evidence LANE-56 keeps; under xdist, four workers would do it at once.
    Observed on this process, so it holds serially and in every worker."""
    from app.core import logger

    assert logger.PREVIOUS_SERVER_LOG_PATH != logger.SERVER_LOG_PATH
    assert logger.PREVIOUS_SERVER_LOG_PATH.parent == logger.SERVER_LOG_PATH.parent
    assert logger.PREVIOUS_SERVER_LOG_PATH != LIVE_PREVIOUS_SERVER_LOG


def test_this_process_logs_to_its_own_tests_log():
    expected = worker_suffixed(TESTS / "tests.log", current_worker())
    files = [
        Path(h.baseFilename).resolve()
        for h in logging.getLogger().handlers
        if isinstance(h, logging.FileHandler)
    ]
    assert files == [expected.resolve()], f"root logger file handlers: {files}"


def test_this_process_never_initialised_the_live_database_at_import():
    """``app.main`` builds the DatabaseEngine at import; the file it opened
    must be this process's own import-time DB, never ``app/arcane_tuner.db``
    (the -n 16 stall, LANE-63). Observed on the env seam AND on the engine:
    the session fixture swaps the singleton later, so the import-time path is
    read back from the environment the engine consulted."""
    import time

    from app.core.db.engine import DatabaseEngine

    engine_started = time.time()
    expected = worker_suffixed(TESTS / "import-time.db", current_worker()).resolve()
    assert Path(os.environ["MRLN_DB_PATH"]).resolve() == expected
    live = (BACKEND / "app" / "arcane_tuner.db").resolve()
    assert Path(DatabaseEngine.get_instance().db_path).resolve() != live
    # A fresh engine with no explicit path consults the seam exactly as the
    # import-time construction does; initialising it must open the diverted
    # file (not merely name it). Done here rather than asserted on the
    # import-time engine's file, which exists only once some module has
    # imported app.main — order the selection, not the test, decides.
    engine = DatabaseEngine(db_path=None)
    assert Path(engine.db_path).resolve() == expected
    engine.initialize()
    try:
        assert expected.exists(), "the engine did not open the diverted file"
        assert not live.exists() or live.stat().st_mtime < engine_started, "the live DB was touched"
    finally:
        engine.close()


# ── a child pytest under an invented worker id ────────────────────────────


def _child(worker: str, *extra: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "MRLN_SERVER_LOG_PATH"}
    env[WORKER_ENV] = worker
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *extra,
         f"{__file__}::test_this_process_never_points_at_the_live_server_log",
         f"{__file__}::test_this_process_logs_to_its_own_tests_log"],
        cwd=str(BACKEND), env=env, capture_output=True, text=True, timeout=300,
    )


def _cleanup(worker: str) -> None:
    for name in (f"tests-{worker}.log", f"server-test-{worker}.log", f"import-time-{worker}.db",
                 f"import-time-{worker}.db-wal", f"import-time-{worker}.db-shm"):
        (TESTS / name).unlink(missing_ok=True)


def test_an_arbitrary_worker_id_gets_its_own_files():
    worker = "gwZ"
    try:
        proc = _child(worker)
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1500:]
        assert (TESTS / f"tests-{worker}.log").exists(), "the worker's tests log was never opened"
    finally:
        _cleanup(worker)


def test_an_operator_pointer_does_not_leak_into_the_workers():
    """Workers inherit the controller's env (and the controller has already set
    the un-suffixed name). A `setdefault` would keep that value and put every
    worker back on ONE file — this is the mutation that variant fails."""
    worker = "gwY"
    try:
        proc = _child(worker, env_extra={"MRLN_SERVER_LOG_PATH": str(TESTS / "server-test.log")})
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1500:]
    finally:
        _cleanup(worker)


def test_without_the_root_conftest_the_divert_is_gone():
    """Prove the negative: `--confcutdir=backend/tests` keeps tests/conftest.py
    but drops backend/conftest.py, and the live-log check must go RED."""
    worker = "gwX"
    try:
        proc = _child(worker, f"--confcutdir={TESTS}")
        assert proc.returncode != 0, "the check passed without the root conftest — it pins nothing"
        assert "test_this_process_never_points_at_the_live_server_log" in proc.stdout
    finally:
        _cleanup(worker)
