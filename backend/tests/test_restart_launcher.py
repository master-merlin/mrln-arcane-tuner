"""A UI-triggered restart must never fail invisibly (LANE-51).

Every test here drives the REAL launcher as a REAL subprocess and asserts on
the artefact a human or the app can open afterwards — the restart log, the
child's own argv, the launcher's exit code. Nothing asserts on the kwargs
handed to ``subprocess``: the defect these pin was precisely that the spawn
looked right while its output went nowhere.

The parent defect, stated so it cannot be re-traded: ``e2e3cfc8`` stopped the
replacement inheriting the parent's stdio (a dead pipe froze the event loop)
by sending it to DEVNULL, which bought silence — a start that failed left no
trace on the console, in ``server.log`` (unlinked by the next startup) or in
the app. The fix must keep the first property and lose the second.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
LAUNCHER = BACKEND / "restart_launcher.py"

sys.path.insert(0, str(BACKEND))
import restart_launcher  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_launcher(tmp_path: Path, child: list[str], *, env_extra: dict | None = None,
                  ready_timeout: str = "8", port_free_timeout: str = "2",
                  timeout: float = 60.0) -> tuple[subprocess.CompletedProcess, Path]:
    """Run the REAL launcher as a real process, logging into *tmp_path*.

    The log path is a module constant (the production value must not be
    configurable by an environment variable a user could point elsewhere), so
    the test process rebinds it before calling ``main`` — everything below that
    line is the shipped code path.
    """
    log = tmp_path / "restart.log"
    env = {**_base_env(), **(env_extra or {})}
    cmd = [
        sys.executable, "-c",
        "import sys, restart_launcher;"
        f"restart_launcher.RESTART_LOG_PATH = r'{log}';"
        "raise SystemExit(restart_launcher.main(sys.argv[1:]))",
        "--old-pid", "0",
        "--ready-timeout", ready_timeout,
        "--port-free-timeout", port_free_timeout,
        "--", *child,
    ]
    proc = subprocess.run(cmd, cwd=str(BACKEND), env=env, capture_output=True,
                          text=True, timeout=timeout)
    return proc, log


def _base_env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND)
    # A settings file the test controls, never the developer's own.
    env.pop("PORT", None)
    env.pop("MRLN_CONTAINER", None)
    return env


def _records(log: Path) -> list[dict]:
    out = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def _events(log: Path) -> list[str]:
    return [r.get("event") for r in _records(log)]


def _settings(tmp_path: Path, body: str) -> dict:
    path = tmp_path / "settings.json"
    path.write_text(body, encoding="utf-8")
    return {"MRLN_SETTINGS_PATH": str(path)}


def _child_that_dies(tmp_path: Path, marker: str) -> list[str]:
    """A replacement that writes *marker* to stderr and dies.

    The marker is read from a FILE rather than embedded in the command, and
    that is load-bearing: the launcher records the command it was given, so a
    marker inside it would appear in the log even when the child's output was
    thrown away — measured, the first version of this test passed with the
    child's stdio back on DEVNULL, i.e. it could not fail on the defect it
    exists to catch. It also writes no trailing newline, like a real traceback.
    """
    source = tmp_path / "marker.txt"
    source.write_text(marker, encoding="utf-8")
    return [sys.executable, "-c",
            f"import sys; sys.stderr.write(open(r'{source}').read()); sys.exit(3)"]


def _child_that_serves(port: int, argv_dump: Path | None = None) -> list[str]:
    dump = f"open(r'{argv_dump}','w').write(' '.join(sys.argv));" if argv_dump else ""
    return [sys.executable, "-c",
            "import sys, socket, time;"
            + dump +
            "p=int(sys.argv[sys.argv.index('--port')+1]);"
            "s=socket.socket();s.bind(('127.0.0.1',p));s.listen(5);"
            "time.sleep(30)",
            "--host", "127.0.0.1", "--port", str(port)]


# ── the defect: a failed start must be discoverable ──────────────────────


class TestAFailedStartIsLoud:

    def test_the_replacement_dying_is_recorded_with_its_own_output(self, tmp_path):
        """The negative LANE-51 proves: before the fix this produced nothing,
        anywhere. Now the child's stderr AND a failure record are in the log."""
        proc, log = _run_launcher(tmp_path, _child_that_dies(tmp_path, "BIND-FAILED-XYZ"),
                                  env_extra=_settings(tmp_path, '{"application": {"backend_port": %d}}' % _free_port()))

        assert proc.returncode == 1, proc.stderr
        text = log.read_text(encoding="utf-8", errors="replace")
        assert "BIND-FAILED-XYZ" in text, "the child's own output was discarded"
        failed = [r for r in _records(log) if r["event"] == "restart_failed"]
        assert failed, _events(log)
        assert failed[0]["exit_code"] == 3
        assert failed[0]["level"] == "error"
        assert "exited with code 3" in failed[0]["message"]

    def test_the_childs_output_goes_to_the_file_and_not_to_our_stdio(self, tmp_path):
        """Keeps e2e3cfc8's property while losing its silence: the output is
        durable, and it is NOT written through the launcher's own handles —
        which are the ones that may be a dead pipe."""
        proc, log = _run_launcher(tmp_path, _child_that_dies(tmp_path, "ONLY-IN-THE-FILE"),
                                  env_extra=_settings(tmp_path, '{"application": {"backend_port": %d}}' % _free_port()))

        assert "ONLY-IN-THE-FILE" in log.read_text(encoding="utf-8")
        assert "ONLY-IN-THE-FILE" not in proc.stdout
        assert "ONLY-IN-THE-FILE" not in proc.stderr
        assert proc.stdout.strip() == "", proc.stdout

    def test_a_replacement_that_serves_is_recorded_as_ready(self, tmp_path):
        """Positive control for the two above: the same machinery on a healthy
        start records readiness and exits 0, so 'no failure record' cannot be
        confused with 'the launcher never got that far'."""
        port = _free_port()
        proc, log = _run_launcher(tmp_path, _child_that_serves(port),
                                  env_extra=_settings(tmp_path, '{"application": {"backend_port": %d}}' % port))

        assert proc.returncode == 0, log.read_text(encoding="utf-8")
        ready = [r for r in _records(log) if r["event"] == "restart_ready"]
        assert ready and ready[0]["port"] == port


# ── the overlap: two servers must never contend for one port ─────────────


class TestNoOverlapOnThePort:

    def test_the_replacement_is_not_started_while_the_port_is_held(self, tmp_path):
        """The old code spawned first and exited 1.0s later. Here the port is
        still held, so the replacement must not run at all — observable as its
        marker file never appearing."""
        port = _free_port()
        marker = tmp_path / "child-ran.txt"
        holder = socket.socket()
        holder.bind(("127.0.0.1", port))
        holder.listen(5)
        try:
            proc, log = _run_launcher(
                tmp_path,
                [sys.executable, "-c", f"open(r'{marker}','w').write('ran')",
                 "--host", "127.0.0.1", "--port", str(port)],
                env_extra=_settings(tmp_path, '{"application": {"backend_port": %d}}' % port),
                port_free_timeout="1",
            )
        finally:
            holder.close()

        assert proc.returncode == 2
        assert not marker.exists(), "the replacement was started onto a held port"
        refused = [r for r in _records(log) if r["event"] == "restart_refused"]
        assert refused and refused[0]["port"] == port
        assert "still held" in refused[0]["message"]

    def test_the_replacement_runs_once_the_port_is_free(self, tmp_path):
        """Positive control: the same command with nothing holding the port."""
        port = _free_port()
        marker = tmp_path / "child-ran.txt"
        proc, log = _run_launcher(
            tmp_path,
            [sys.executable, "-c", f"open(r'{marker}','w').write('ran')",
             "--host", "127.0.0.1", "--port", str(port)],
            env_extra=_settings(tmp_path, '{"application": {"backend_port": %d}}' % port),
            ready_timeout="2",
        )

        assert marker.exists(), log.read_text(encoding="utf-8")
        assert proc.returncode == 1  # it exits without serving — reported, correctly


# ── the latent defect: the port is resolved, never replayed ──────────────


class TestThePortIsResolvedNotReplayed:

    def test_settings_moved_the_port_so_the_replacement_moves_with_it(self, tmp_path):
        """Change the port in settings, restart: the replacement must bind the
        NEW port. Replaying sys.orig_argv kept the old one while the app
        believed the new one — the disagreement start_backend.bat's
        ONE-producer comment exists to prevent."""
        new_port = _free_port()
        stale_port = _free_port()
        argv_dump = tmp_path / "child-argv.txt"
        proc, log = _run_launcher(
            tmp_path,
            _child_that_serves(stale_port, argv_dump),   # argv still says the OLD port
            env_extra=_settings(tmp_path, '{"application": {"backend_port": %d}}' % new_port),
        )

        assert proc.returncode == 0, log.read_text(encoding="utf-8")
        dumped = argv_dump.read_text(encoding="utf-8")
        assert f"--port {new_port}" in dumped
        assert f"--port {stale_port}" not in dumped
        changed = [r for r in _records(log) if r["event"] == "restart_port_changed"]
        assert changed and changed[0]["previous_port"] == stale_port
        assert changed[0]["port"] == new_port

    def test_an_unreadable_settings_file_refuses_instead_of_guessing(self, tmp_path):
        """Same refusal start_backend.bat makes: a port that cannot be resolved
        must not become a guess the app then disagrees with."""
        marker = tmp_path / "child-ran.txt"
        proc, log = _run_launcher(
            tmp_path,
            [sys.executable, "-c", f"open(r'{marker}','w').write('ran')"],
            env_extra=_settings(tmp_path, "{not json at all"),
        )

        assert proc.returncode == 2
        assert not marker.exists()
        refused = [r for r in _records(log) if r["event"] == "restart_refused"]
        assert refused and "not valid JSON" in refused[0]["reason"]


# ── the report the next server makes ─────────────────────────────────────


class TestPendingFailureReport:

    def _log_with(self, tmp_path: Path, *lines: str) -> Path:
        path = tmp_path / "restart.log"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_a_failure_is_pending_until_it_is_reported(self, tmp_path):
        log = self._log_with(
            tmp_path,
            'INFO:     Uvicorn running on http://127.0.0.1:8000',   # raw child output
            json.dumps({"event": "restart_failed", "exit_code": 3}),
        )
        assert restart_launcher.pending_failure(str(log))["exit_code"] == 3

        restart_launcher.mark_reported(str(log))
        assert restart_launcher.pending_failure(str(log)) is None

    def test_a_successful_restart_leaves_nothing_pending(self, tmp_path):
        log = self._log_with(
            tmp_path,
            json.dumps({"event": "restart_failed", "exit_code": 3}),
            json.dumps({"event": "restart_ready", "port": 8000}),
        )
        assert restart_launcher.pending_failure(str(log)) is None

    def test_raw_console_output_is_never_mistaken_for_an_outcome(self, tmp_path):
        """Positive control on the parser: the file interleaves the child's own
        non-JSON console output, which must not be read as a record."""
        log = self._log_with(
            tmp_path,
            json.dumps({"event": "restart_failed", "exit_code": 3}),
            "Traceback (most recent call last):",
            '  File "x.py", line 1, in <module>',
            "restart_ready",           # bare word, not a record
            '{"event": "restart_ready"',   # truncated JSON, not a record
        )
        assert restart_launcher.pending_failure(str(log))["exit_code"] == 3

    def test_a_missing_log_is_not_an_error(self, tmp_path):
        assert restart_launcher.pending_failure(str(tmp_path / "nope.log")) is None


# ── the bound on the log ─────────────────────────────────────────────────


def test_the_restart_log_is_bounded(tmp_path):
    path = tmp_path / "restart.log"
    path.write_text("x" * (restart_launcher.MAX_LOG_BYTES + 10), encoding="utf-8")

    handle = restart_launcher.open_log(str(path))
    handle.close()

    text = path.read_text(encoding="utf-8")
    assert "xxxx" not in text
    assert "restart_log_truncated" in text


def test_the_launcher_refuses_a_command_it_cannot_understand():
    with pytest.raises(SystemExit):
        restart_launcher._parse_args(["--old-pid", "1"])
    with pytest.raises(SystemExit):
        restart_launcher._parse_args(["--"])


def test_port_is_free_answers_the_operating_system():
    port = _free_port()
    assert restart_launcher.port_is_free("127.0.0.1", port)
    holder = socket.socket()
    holder.bind(("127.0.0.1", port))
    holder.listen(1)
    try:
        assert not restart_launcher.port_is_free("127.0.0.1", port)
    finally:
        holder.close()
    deadline = time.monotonic() + 5
    while not restart_launcher.port_is_free("127.0.0.1", port) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert restart_launcher.port_is_free("127.0.0.1", port)
