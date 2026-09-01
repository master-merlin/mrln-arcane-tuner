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


# ── the rule ──────────────────────────────────────────────────────────────


def test_serial_name_is_unchanged_and_worker_name_carries_the_id():
    assert worker_suffixed(Path("x/tests.log"), None) == Path("x/tests.log")
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


def test_this_process_logs_to_its_own_tests_log():
    expected = worker_suffixed(TESTS / "tests.log", current_worker())
    files = [
        Path(h.baseFilename).resolve()
        for h in logging.getLogger().handlers
        if isinstance(h, logging.FileHandler)
    ]
    assert files == [expected.resolve()], f"root logger file handlers: {files}"


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
    for name in (f"tests-{worker}.log", f"server-test-{worker}.log"):
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
