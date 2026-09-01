"""Supervise the replacement server a UI-triggered restart spawns.

WHY THIS EXISTS (LANE-51, measured 2026-08-31):

``/api/system/restart`` used to spawn the replacement itself with
``stdin/stdout/stderr=DEVNULL`` and then exit. That kept a real property —
never inherit the parent's stdio, because it may be a dead pipe whose full
buffer blocks console logging *while holding the logging lock* and freezes the
event loop (``e2e3cfc8``, live incident 2026-07-16) — but it paid for it with
total silence: **measured on this branch's parent commit**, the console's last
line is the restart request itself and nothing is ever written to it again,
while the replacement logs only to ``server.log``, which the replacement then
*deletes* on startup. A replacement that failed to start therefore left no
trace anywhere: not on the console, not in the server log, not in the app.

The trade is kept and the silence is not: the child's console output goes to a
**file** (never a pipe, so no reader can ever block it), and a start that fails
is written there as a record, loudly, instead of being discarded.

This module is LAUNCHER TIER, like ``port_resolver``: it imports only stdlib
plus ``port_resolver`` and never anything from ``app``. It runs before and
around the server, so it must not pay the app's import cost and must not be
able to fail because the app cannot be imported.

WHAT IT GUARANTEES, in order:

1. the old server's port is **free** before the replacement is started, so the
   two never contend for one socket (the previous code slept 1.0 s and exited,
   overlapping the two processes by construction);
2. the port is **re-resolved** through ``port_resolver`` — the one producer —
   instead of replaying the old ``--port``. A restart after changing the port
   in Server Control used to keep serving the old one while the app believed
   the new one: the exact disagreement ``start_backend.bat``'s ONE-producer
   comment exists to prevent. A change is reported, never silent;
3. the replacement is **watched** until it listens, dies, or the bound expires,
   and each of those three outcomes is a record in the restart log.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # backend/
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

import port_resolver  # noqa: E402 - requires the sys.path line above

#: Where the replacement's console output and this module's records go.
#: NOT ``server.log``: ``setup_logging`` unlinks that file on every startup, so
#: a failed restart's evidence would be destroyed by the very next successful
#: start — which is the user's recovery action.
RESTART_LOG_PATH = os.path.join(_THIS_DIR, "restart.log")

#: Bound on the restart log. It accumulates one boot's console output per
#: restart, so it grows without one (invariant: every buffer bounded).
MAX_LOG_BYTES = 2_000_000

#: How long to wait for the outgoing server to release the port. It exits with
#: ``os._exit`` immediately after spawning us, so this is a bound on a
#: pathology (a wedged process), not on the normal case.
PORT_FREE_TIMEOUT = 30.0

#: How long to wait for the replacement to start listening. This must exceed a
#: COLD start, where importing torch dominates — on a cold page cache a full
#: boot to first bind was measured at ~150 s, so a bound in the tens of seconds
#: would report a healthy server as failed. Expiry is REPORTED, not fatal: the
#: child is left running, because killing a server that may be seconds from
#: ready is worse than saying we stopped watching.
READY_TIMEOUT = 300.0

_POLL_INTERVAL = 0.25

DEFAULT_PROBE_HOST = "127.0.0.1"


class RestartRefused(RuntimeError):
    """The replacement must not be started; the reason is already recorded."""


# ── The restart log ───────────────────────────────────────────────────────


def open_log(path: str | None = None):
    """Open the restart log for append, bounded, and return the file object.

    A FILE, deliberately, and this is the whole point of the module: a file
    handle cannot be a dead pipe with a gone reader, so handing it to the child
    as stdout/stderr restores observability without restoring the freeze that
    ``e2e3cfc8`` removed.
    """
    target = RESTART_LOG_PATH if path is None else path
    try:
        if os.path.getsize(target) > MAX_LOG_BYTES:
            with open(target, "w", encoding="utf-8"):
                pass  # truncate; the note below says so in the log itself
            dropped = True
        else:
            dropped = False
    except OSError:
        dropped = False
    handle = open(target, "a", encoding="utf-8", errors="replace", buffering=1)
    if dropped:
        record(handle, "restart_log_truncated", limit_bytes=MAX_LOG_BYTES,
               message="the restart log passed its size bound; earlier entries were dropped")
    return handle


def record(handle, event: str, level: str = "info", **fields) -> None:
    """Append one single-line JSON record (_docs/LOGGING.md schema).

    Never raises: a diagnostic that can take the restart down is worse than the
    silence it replaces.
    """
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
        "level": level,
        "service": "restart-launcher",
        "event": event,
    }
    payload.update(fields)
    try:
        # Leading newline, not cosmetic: the child writes into this same file
        # through its own descriptor and may leave a partial line (a traceback's
        # last line has no terminator). Without this, our record would be glued
        # onto the end of that line and stop being a record at all — measured,
        # the failure record vanished from a run that had failed.
        handle.write("\n" + json.dumps(payload, default=str) + "\n")
        handle.flush()
    except Exception:  # noqa: BLE001 - see docstring
        pass


#: Records that conclude one restart attempt. ``pending_failure`` reads the LAST
#: of these and nothing else, so the child's own console output — which is
#: interleaved into the same file and is not JSON — can never be mistaken for an
#: outcome.
_OUTCOME_EVENTS = {"restart_ready", "restart_failed", "restart_refused",
                   "restart_not_ready"}
_FAILURE_EVENTS = {"restart_failed", "restart_refused", "restart_not_ready"}
_REPORTED_EVENT = "restart_failure_reported"


def _tail_records(path: str, limit: int = 400) -> list[dict]:
    """The last *limit* JSON records in the restart log, oldest first."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()[-limit:]
    except OSError:
        return []
    out = []
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue  # raw child console output, not a record
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict) and "event" in parsed:
            out.append(parsed)
    return out


def pending_failure(path: str | None = None) -> dict | None:
    """The last restart outcome if it FAILED and has not been reported yet.

    Dedupe carries no clock and rewrites nothing: reporting appends a
    ``restart_failure_reported`` record, and a failure with such a record after
    it is spent. A file that cannot be read yields ``None`` — this is a
    diagnostic on a startup path and may never be a reason not to start.
    """
    target = RESTART_LOG_PATH if path is None else path
    last: dict | None = None
    for entry in _tail_records(target):
        event = entry.get("event")
        if event in _OUTCOME_EVENTS:
            last = entry
        elif event == _REPORTED_EVENT:
            last = None
    if last is not None and last.get("event") in _FAILURE_EVENTS:
        return last
    return None


def mark_reported(path: str | None = None) -> None:
    """Record that a pending failure has been surfaced to the user."""
    target = RESTART_LOG_PATH if path is None else path
    try:
        with open(target, "a", encoding="utf-8") as fh:
            record(fh, _REPORTED_EVENT)
    except OSError:
        pass


# ── Port handling ─────────────────────────────────────────────────────────


def _split_flag(argv: list[str], flag: str) -> tuple[int | None, str | None]:
    """Index and value of ``--flag V`` / ``--flag=V`` in *argv*."""
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return i, argv[i + 1]
        if arg.startswith(flag + "="):
            return i, arg.split("=", 1)[1]
    return None, None


def port_is_free(host: str, port: int) -> bool:
    """True when a server could bind (host, port) right now.

    Asks the operating system rather than a process table: the question is
    whether the replacement can bind, and only a bind answers it. No
    ``SO_REUSEADDR`` — asyncio does not set it on Windows either, so this probe
    fails exactly when uvicorn's own bind would.
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def wait_for_port_free(host: str, port: int, timeout: float,
                       sleep=time.sleep) -> bool:
    """Block until (host, port) is bindable, bounded by *timeout*."""
    deadline = time.monotonic() + timeout
    while True:
        if port_is_free(host, port):
            return True
        if time.monotonic() >= deadline:
            return False
        sleep(_POLL_INTERVAL)


def is_listening(host: str, port: int) -> bool:
    """True when something accepts connections on (host, port)."""
    probe_host = DEFAULT_PROBE_HOST if host in ("0.0.0.0", "::", "") else host
    try:
        with socket.create_connection((probe_host, port), timeout=1.0):
            return True
    except OSError:
        return False


def replacement_command(child_argv: list[str], environ=None) -> tuple[list[str], int, int | None]:
    """The command to run, its port, and the port it replaced (if different).

    The old ``--port`` is REPLACED, not replayed. ``port_resolver`` is the one
    producer of the port for every launcher (``start_backend.bat`` lines 19-33),
    and a restart is a launch: replaying argv made the restarted server keep a
    port the app's own settings had already moved away from.

    Raises ``RestartRefused`` when the settings file exists but cannot be
    understood — the same refusal ``start_backend.bat`` makes, for the same
    reason: starting on a guessed port while the app believes another one is
    the silent disagreement this whole path exists to remove.
    """
    env = os.environ if environ is None else environ
    try:
        resolved = port_resolver.resolve_port([], environ=env)
    except port_resolver.PortResolutionError as exc:
        raise RestartRefused(str(exc)) from exc

    argv = list(child_argv)
    index, raw = _split_flag(argv, "--port")
    previous: int | None = None
    if raw is not None:
        try:
            previous = int(raw.strip())
        except ValueError:
            previous = None
    if index is None:
        argv += ["--port", str(resolved)]
    elif argv[index] == "--port":
        argv[index + 1] = str(resolved)
    else:
        argv[index] = f"--port={resolved}"
    return argv, resolved, (previous if previous != resolved else None)


def bind_host(child_argv: list[str]) -> str:
    """The ``--host`` the replacement will bind, or the loopback default."""
    _, raw = _split_flag(list(child_argv), "--host")
    return raw.strip() if raw else DEFAULT_PROBE_HOST


# ── Entry point ───────────────────────────────────────────────────────────


def _parse_args(argv: list[str]) -> tuple[dict, list[str]]:
    """``[--old-pid N] [--ready-timeout S] -- <command…>``."""
    if "--" not in argv:
        raise SystemExit("restart_launcher: expected `-- <command>`")
    split = argv.index("--")
    head, child = argv[:split], argv[split + 1:]
    opts: dict = {"old_pid": None, "ready_timeout": READY_TIMEOUT,
                  "port_free_timeout": PORT_FREE_TIMEOUT}
    _, pid = _split_flag(head, "--old-pid")
    if pid:
        opts["old_pid"] = pid
    _, ready = _split_flag(head, "--ready-timeout")
    if ready:
        opts["ready_timeout"] = float(ready)
    # Both bounds are overridable so a test can drive the timeout paths in
    # seconds instead of minutes — the production defaults are the constants.
    _, free = _split_flag(head, "--port-free-timeout")
    if free:
        opts["port_free_timeout"] = float(free)
    if not child:
        raise SystemExit("restart_launcher: the command after `--` is empty")
    return opts, child


def main(argv: list[str] | None = None) -> int:
    """Wait out the old server, start the replacement, and watch it.

    Exit codes: 0 the replacement is listening, 1 it failed or never became
    ready, 2 the restart was refused before anything was started.
    """
    opts, child_argv = _parse_args(list(sys.argv[1:] if argv is None else argv))
    log = open_log()
    try:
        record(log, "restart_launcher_started", old_pid=opts["old_pid"],
               command=child_argv)

        try:
            command, port, previous_port = replacement_command(child_argv)
        except RestartRefused as exc:
            record(log, "restart_refused", level="error", reason=str(exc),
                   message="the replacement server was NOT started: " + str(exc))
            return 2

        if previous_port is not None:
            record(log, "restart_port_changed", level="warning",
                   previous_port=previous_port, port=port,
                   message=(f"the replacement binds {port}, not {previous_port}: the port "
                            "is resolved from settings/PORT on every launch, never replayed"))

        host = bind_host(command)
        free_timeout = opts["port_free_timeout"]
        if not wait_for_port_free(host, port, free_timeout):
            record(log, "restart_refused", level="error", host=host, port=port,
                   timeout_seconds=free_timeout,
                   message=(f"{host}:{port} is still held after {free_timeout:.0f}s by some "
                            "process — the outgoing server should have exited immediately. "
                            "The replacement was NOT started, because two servers on one "
                            "port is the failure this waits to avoid. Find what holds the "
                            "port, end it, and start the backend again."))
            return 2

        try:
            child = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                close_fds=True,
                cwd=os.getcwd(),
            )
        except OSError as exc:
            record(log, "restart_failed", level="error", error=str(exc),
                   command=command,
                   message="the replacement server could not be started: " + str(exc))
            return 1

        record(log, "restart_child_spawned", pid=child.pid, port=port, host=host)
        return _watch(log, child, host, port, opts["ready_timeout"])
    finally:
        log.close()


def _watch(log, child, host: str, port: int, ready_timeout: float) -> int:
    """Watch the replacement until it listens, dies, or the bound expires."""
    deadline = time.monotonic() + ready_timeout
    while True:
        code = child.poll()
        if code is not None:
            record(log, "restart_failed", level="error", pid=child.pid,
                   exit_code=code, host=host, port=port,
                   message=(f"the replacement server exited with code {code} before it "
                            f"served {host}:{port}. Its output is in this file, directly "
                            "above this line."))
            return 1
        if is_listening(host, port):
            record(log, "restart_ready", pid=child.pid, host=host, port=port,
                   message=f"the replacement server is serving {host}:{port}")
            return 0
        if time.monotonic() >= deadline:
            record(log, "restart_not_ready", level="error", pid=child.pid,
                   host=host, port=port, timeout_seconds=ready_timeout,
                   message=(f"the replacement server (pid {child.pid}) was still not serving "
                            f"{host}:{port} after {ready_timeout:.0f}s. It is still running "
                            "and was NOT killed; its output is in this file."))
            return 1
        time.sleep(_POLL_INTERVAL)


if __name__ == "__main__":
    raise SystemExit(main())
