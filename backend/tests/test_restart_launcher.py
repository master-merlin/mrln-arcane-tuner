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
                  timeout: float = 60.0, console: Path | None = None,
                  heartbeat: float | None = None,
                  ) -> tuple[subprocess.CompletedProcess, Path]:
    """Run the REAL launcher as a real process, logging into *tmp_path*.

    The log path is a module constant (the production value must not be
    configurable by an environment variable a user could point elsewhere), so
    the test process rebinds it before calling ``main`` — everything below that
    line is the shipped code path.

    ``CONSOLE_PATH`` is rebound for the same reason and with the same care: its
    production value is ``CONOUT$``, the console DEVICE, which bypasses every
    redirection there is — a test that left it alone would print onto the
    developer's terminal and could assert nothing. Point it at a file and the
    exact lines the user's terminal would receive become observable.
    """
    log = tmp_path / "restart.log"
    console_path = console if console is not None else Path(os.devnull)
    env = {**_base_env(), **(env_extra or {})}
    beat = (f"restart_launcher.HEARTBEAT_INTERVAL = {heartbeat!r};"
            if heartbeat is not None else "")
    cmd = [
        sys.executable, "-c",
        "import sys, restart_launcher;"
        f"restart_launcher.RESTART_LOG_PATH = r'{log}';"
        f"restart_launcher.CONSOLE_PATH = r'{console_path}';"
        + beat +
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


# ── LANE-56: the terminal the user is watching ───────────────────────────
#
# UAT-5.9, the user, verbatim and for the second round running: "The restart
# still isn't coming back in the same terminal as it was in the past - the
# frontend says it cannot connect ... even so it is in the tasklist. Killing the
# task and restarting it with the batch file in the initially used terminal
# brings it back."
#
# MEASURED 2026-09-01 on a spare port, and it settles what the exit code meant:
# a warm restart takes ~46 s (29 s of it waiting for the outgoing server to
# release the port, then 15 s of import) and a COLD start of this app was
# measured at 6.5 minutes. `taskkill /F` on the replacement mid-import produced
# exactly the record the user's restart.log holds - `exit_code: 1`, no traceback
# above it - because on Windows that is TerminateProcess(h, 1). Neither uvicorn
# failure mode can produce a 1: a bind failure and a lifespan failure both
# `sys.exit(STARTUP_FAILURE)`, and uvicorn/config.py:80 sets that to 3.
#
# So the replacement did not crash. It was still starting, nothing anywhere said
# so, and the user ended it. These pin the narration that makes waiting a
# choice, and the record that no longer accuses a healthy boot of failing.


class TestTheTerminalIsToldWhatIsHappening:

    def test_a_successful_restart_narrates_the_console_from_start_to_serving(self, tmp_path):
        port = _free_port()
        console = tmp_path / "console.txt"
        proc, log = _run_launcher(
            tmp_path, _child_that_serves(port), console=console,
            env_extra=_settings(tmp_path, '{"application": {"backend_port": %d}}' % port))

        assert proc.returncode == 0, proc.stderr
        text = console.read_text(encoding="utf-8")
        assert "restarting" in text, text
        assert "replacement started" in text, text
        assert f"serving http://127.0.0.1:{port}" in text, text
        assert "this terminal is live again" in text, text

    def test_a_slow_start_keeps_saying_so_instead_of_going_silent(self, tmp_path):
        """The whole defect in one assertion: a boot that takes minutes must
        keep speaking. The heartbeat is shortened so the test costs seconds; the
        code path is the shipped one and the interval is the only knob."""
        port = _free_port()
        console = tmp_path / "console.txt"
        slow = [sys.executable, "-c",
                "import sys, socket, time;"
                "p=int(sys.argv[sys.argv.index('--port')+1]);"
                "time.sleep(2.5);"
                "s=socket.socket();s.bind(('127.0.0.1',p));s.listen(5);"
                "time.sleep(30)",
                "--host", "127.0.0.1", "--port", str(port)]
        proc, log = _run_launcher(
            tmp_path, slow, console=console, heartbeat=0.5,
            env_extra=_settings(tmp_path, '{"application": {"backend_port": %d}}' % port))

        assert proc.returncode == 0, proc.stderr
        text = console.read_text(encoding="utf-8")
        beats = [ln for ln in text.splitlines() if "still starting" in ln]
        assert len(beats) >= 2, text
        assert "Do NOT end this task" in text, text

    def test_narration_never_travels_through_the_launchers_own_stdio(self, tmp_path):
        """Keeps e2e3cfc8's property. The console is a DEVICE this process opens
        for itself; it is never the inherited handle that may be a dead pipe,
        and the launcher's own stdout stays empty exactly as before."""
        port = _free_port()
        console = tmp_path / "console.txt"
        proc, log = _run_launcher(
            tmp_path, _child_that_serves(port), console=console,
            env_extra=_settings(tmp_path, '{"application": {"backend_port": %d}}' % port))

        assert proc.stdout.strip() == "", proc.stdout
        assert "replacement started" not in proc.stdout
        assert "replacement started" not in proc.stderr
        assert "replacement started" in console.read_text(encoding="utf-8")

    def test_no_console_is_not_a_failure(self, tmp_path):
        """A container, a service, a detached start: the open fails and the
        restart proceeds anyway. Narration may never be a reason not to
        restart."""
        port = _free_port()
        proc, log = _run_launcher(
            tmp_path, _child_that_serves(port),
            console=tmp_path / "no" / "such" / "device",
            env_extra=_settings(tmp_path, '{"application": {"backend_port": %d}}' % port))

        assert proc.returncode == 0, proc.stderr
        assert "restart_ready" in _events(log)

    def test_every_narration_line_survives_a_codepage_850_console(self, tmp_path):
        """The terminal is a `start_backend.bat` window, not a UTF-8 one: an em
        dash arrived there as `â€` (measured). A line whose whole job is to stop
        the user ending a process must not look like line noise."""
        target = tmp_path / "console.txt"
        with open(target, "w", encoding="utf-8") as fh:
            restart_launcher.say(fh, "an em dash — an ellipsis … a quote ’")
        body = target.read_text(encoding="utf-8")

        assert body.strip() == "[restart] an em dash - an ellipsis ... a quote '"
        body.encode("ascii")  # raises if anything non-ASCII survived

    def test_a_real_restart_writes_nothing_the_console_cannot_render(self, tmp_path):
        """The control for the above on the ACTUAL messages, not a synthetic
        one: transliteration in `say` is only worth having if the lines the
        launcher really emits go through it."""
        port = _free_port()
        console = tmp_path / "console.txt"
        _run_launcher(tmp_path, _child_that_serves(port), console=console,
                      env_extra=_settings(tmp_path,
                                          '{"application": {"backend_port": %d}}' % port))

        console.read_bytes().decode("ascii")

    def test_open_console_returns_none_rather_than_raising(self, monkeypatch, tmp_path):
        monkeypatch.setattr(restart_launcher, "CONSOLE_PATH",
                            str(tmp_path / "nope" / "nope"))
        assert restart_launcher.open_console() is None
        restart_launcher.say(None, "must not raise")


class TestAFailureRecordDoesNotAccuseAHealthyBoot:

    def test_the_record_says_how_far_the_child_got_and_how_long_it_ran(self, tmp_path):
        """The user's own record said only 'exited with code 1'. What it needed
        to say is that the last thing the child managed was a startup log line
        45 s in - that it was still booting, not that it had failed."""
        marker = tmp_path / "marker.txt"
        marker.write_text("LAST-THING-IT-MANAGED", encoding="utf-8")
        child = [sys.executable, "-c",
                 "import sys, time;"
                 f"sys.stdout.write(open(r'{marker}').read());"
                 "sys.stdout.write(chr(10)); sys.stdout.flush();"
                 "time.sleep(1.2); sys.exit(1)"]
        proc, log = _run_launcher(
            tmp_path, child,
            env_extra=_settings(tmp_path, '{"application": {"backend_port": %d}}' % _free_port()))

        failed = [r for r in _records(log) if r["event"] == "restart_failed"]
        assert failed, _events(log)
        assert failed[0]["last_child_output"] == "LAST-THING-IT-MANAGED"
        assert failed[0]["elapsed_seconds"] >= 1.0, failed[0]

    @pytest.mark.skipif(os.name != "nt",
                        reason="TerminateProcess exit code is Windows-specific")
    def test_exit_code_1_on_windows_names_the_end_task_it_could_be(self, tmp_path):
        """Reproduced live (LANE-56): `taskkill /F` on the replacement gives
        exit 1 and no traceback, byte-identical to what the user's log held.
        uvicorn cannot produce a 1 - both of its startup failure paths exit
        STARTUP_FAILURE == 3 (uvicorn/config.py:80) - so a bare 1 must never be
        reported as the server having failed on its own."""
        child = [sys.executable, "-c", "import sys; sys.exit(1)"]
        proc, log = _run_launcher(
            tmp_path, child,
            env_extra=_settings(tmp_path, '{"application": {"backend_port": %d}}' % _free_port()))

        failed = [r for r in _records(log) if r["event"] == "restart_failed"]
        assert failed, _events(log)
        assert failed[0]["exit_code"] == 1
        assert "End task" in failed[0]["message"], failed[0]["message"]
        assert "taskkill" in failed[0]["message"]

    def test_an_ordinary_crash_is_not_dressed_up_as_an_end_task(self, tmp_path):
        """The negative control for the note above: exit 3 is uvicorn actually
        failing, and it must read as a failure with no 'maybe you ended it'."""
        proc, log = _run_launcher(
            tmp_path, _child_that_dies(tmp_path, "REAL-TRACEBACK-HERE"),
            env_extra=_settings(tmp_path, '{"application": {"backend_port": %d}}' % _free_port()))

        failed = [r for r in _records(log) if r["event"] == "restart_failed"]
        assert failed[0]["exit_code"] == 3
        assert "End task" not in failed[0]["message"]
        assert failed[0]["last_child_output"] == "REAL-TRACEBACK-HERE"


def test_a_refusal_says_that_nothing_is_serving_and_how_to_get_back():
    """OBSERVED LIVE TWICE (LANE-56, 21:02:05Z and 21:13:18Z): the port was
    still held at the bound, the launcher refused, and the outcome was no
    backend at all - the user's "backend did not come back after restart". The
    old message sent him to look for a process holding the port, which is the
    wrong errand: the holder is the outgoing server, on its way out. What he
    needs is to be told that nothing is running and to start it again."""
    port = _free_port()
    holder = socket.socket()
    holder.bind(("127.0.0.1", port))
    holder.listen(1)
    console = None
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            console = tmp / "console.txt"
            proc, log = _run_launcher(
                tmp, _child_that_serves(port), port_free_timeout="1",
                console=console,
                env_extra=_settings(tmp, '{"application": {"backend_port": %d}}' % port))

            assert proc.returncode == 2, proc.stderr
            refused = [r for r in _records(log) if r["event"] == "restart_refused"]
            assert refused, _events(log)
            assert "NOTHING IS SERVING NOW" in refused[0]["message"]
            assert "started again by hand" in refused[0]["message"]
            body = console.read_text(encoding="utf-8")
            assert "NOTHING IS SERVING NOW" in body, body
            assert "start_backend.bat" in body, body
    finally:
        holder.close()


def test_the_port_free_bound_is_generous_enough_for_a_measured_handover():
    """The bound was 30 s and the measured handover exceeded it twice, each time
    costing the user the whole backend. Waiting is cheap and narrated; refusing
    is not. Pinned so it is not tightened back without re-measuring."""
    assert restart_launcher.PORT_FREE_TIMEOUT >= 60.0


def test_the_outgoing_server_leaves_the_port_even_when_its_loop_is_blocked():
    """MEASURED 2026-09-01, and it is a third of every restart: the outgoing
    server's `await asyncio.sleep(0.25); os._exit(0)` took 26 SECONDS, because
    the loop had other work and never came back to that coroutine. It held the
    port for all of it and the launcher could only wait. The exit window must
    therefore not be schedulable on the loop it is waiting to leave.

    Asserts the observable output - the process is gone, in time - and never
    that a Timer was constructed. A blocked loop is simulated with a real
    blocking `time.sleep` inside the coroutine, which is what starvation is.
    """
    script = (
        "import asyncio, sys, time, restart_launcher\n"
        "async def main():\n"
        "    restart_launcher.schedule_exit(0.05)\n"
        "    time.sleep(30)\n"          # the starved loop
        "asyncio.run(main())\n"
        "sys.exit(9)\n"                 # only reachable if the exit was starved
    )
    started = time.monotonic()
    proc = subprocess.run([sys.executable, "-c", script], cwd=str(BACKEND),
                          env=_base_env(), capture_output=True, text=True,
                          timeout=45)
    elapsed = time.monotonic() - started

    assert proc.returncode == 0, (proc.returncode, proc.stderr)
    assert elapsed < 10, f"the exit waited on the blocked loop: {elapsed:.1f}s"


def test_schedule_exit_leaves_with_the_code_it_was_given():
    """LANE-56: under a supervisor the outgoing server exits with the sentinel
    (``restart_contract.RESTART_EXIT_CODE``) and the supervisor relaunches it.
    The code is the whole message, so it must survive the timer thread: asserted
    on the child's returncode — the observable — never on the Timer's args.
    Mutation: a ``schedule_exit`` that ignores its code exits 0 and turns this
    red. The default stays 0 for the launcher path (``schedule_exit(0.25)``)."""
    from app.core.restart_contract import RESTART_EXIT_CODE

    script = (
        "import sys, time, restart_launcher\n"
        f"restart_launcher.schedule_exit(0.05, code={RESTART_EXIT_CODE})\n"
        "time.sleep(30)\n"
        "sys.exit(9)\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], cwd=str(BACKEND),
                          env=_base_env(), capture_output=True, text=True,
                          timeout=45)
    assert proc.returncode == RESTART_EXIT_CODE, (proc.returncode, proc.stderr)


def test_last_child_output_never_returns_our_own_records(tmp_path):
    """Our records and the child's output share one file. If the tail could
    return a launcher record, every failure would 'last have managed' to say
    that it failed - a mirror, not evidence."""
    log = tmp_path / "restart.log"
    log.write_text(
        "CHILD-SAID-THIS\n"
        '{"timestamp": "x", "level": "info", "service": "restart-launcher",'
        ' "event": "restart_child_spawned"}\n',
        encoding="utf-8")
    assert restart_launcher.last_child_output(str(log)) == "CHILD-SAID-THIS"


def test_last_child_output_is_bounded_and_survives_a_missing_file(tmp_path):
    log = tmp_path / "restart.log"
    log.write_text("x" * 200_000 + "\nTHE-TAIL\n", encoding="utf-8")
    assert restart_launcher.last_child_output(str(log)) == "THE-TAIL"
    assert restart_launcher.last_child_output(str(tmp_path / "absent.log")) is None
