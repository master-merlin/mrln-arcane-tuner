"""Resolve the port the backend serves on. ONE producer, no app imports.

WHY THIS IS NOT IN ``app/core/`` — measured, not assumed:

    from app.core.container_config import resolve_port      ~246 ms
    import json, os                                          ~14 ms

``app.core.container_config`` pulls ``app.core.compat`` (238 ms), which pulls
``structlog`` (152 ms), which pulls ``structlog.dev`` (106 ms). Every launcher
must know the port BEFORE Python starts the server, so that cost would be paid
on every launch to read one integer. This module imports ``json``, ``os`` and
``sys`` and nothing else, ever — all three are already resident in a bare
interpreter, so the import is free.

``app.core.container_config.resolve_port`` DELEGATES here rather than keeping a
second copy of the precedence rules (RULE-21). That is the whole reason this
module takes ``settings_fallback`` and ``warn``: the in-app caller already holds
the loaded settings and owns a structured logger, and neither of those may
become a reason to fork the logic.

WHY THE SHELL DOES NOT PARSE THE SETTINGS FILE:

The obvious implementation is four launchers parsing ``settings.json`` — which
would create a second producer of the port in three languages (RULE-21), the
exact defect the pipeline fix removed. Instead the shell is a CARRIER: it asks
this module for the port, passes it as ``--port``, and the app confirms by
reading ``--port`` back out of its own ``sys.argv``. The app observes what was
actually applied rather than recomputing what it thinks should have been — the
same inversion the bind-host guard uses for ``--host``.

FAILURE POLICY, and the distinction that matters most here:

An ABSENT settings file is a FIRST LAUNCH, not an error. ``SettingsManager``
writes its defaults on first construction, which happens after this runs, so on
a fresh install this module will always find no file. Refusing there would make
a new installation fail to start — a worse defect than the one this exists to
fix. A file that exists but cannot be understood is different: that is a
misconfiguration, and starting on a guessed port while the app believes another
is precisely the silent disagreement being eliminated. So: absent is 8000,
unreadable is a refusal.
"""

from __future__ import annotations

import json
import os
import sys

DEFAULT_PORT = 8000

#: The registered/dynamic port range. 0 is excluded deliberately — "let the OS
#: choose" cannot work here, because the launcher must tell the app which port
#: it picked before the socket exists.
MIN_PORT = 1
MAX_PORT = 65535

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # backend/


class PortResolutionError(RuntimeError):
    """The settings file exists but does not yield a usable port."""


def settings_path(environ=None) -> str:
    """Where ``SettingsManager`` would look, resolved identically.

    ``MRLN_SETTINGS_PATH`` wins (the container points it at the persistent
    volume); otherwise ``backend/settings.json``. Anchored on ``__file__``,
    never the working directory — a launcher may be invoked from anywhere.

    Takes the same *environ* override ``resolve_port`` does. Reading
    ``os.environ`` here while the caller supplied a different mapping would let
    the two disagree about which environment is in force — the same class of
    split-brain this module exists to remove, one level down.
    """
    env = os.environ if environ is None else environ
    explicit = env.get("MRLN_SETTINGS_PATH")
    if explicit:
        return explicit
    return os.path.join(_THIS_DIR, "settings.json")


def _port_from_argv(argv: list[str]) -> int | None:
    """``--port N`` / ``--port=N`` from a command line, or None."""
    for i, arg in enumerate(argv):
        if arg == "--port":
            if i + 1 < len(argv):
                raw = argv[i + 1].strip()
                break
            return None
        if arg.startswith("--port="):
            raw = arg.split("=", 1)[1].strip()
            break
    else:
        return None
    try:
        return int(raw)
    except ValueError:
        # An explicit but unparseable --port is uvicorn's problem to reject;
        # this module must not invent a different answer than the one the
        # command line asked for.
        return None


def _port_from_settings(path: str) -> int | None:
    """The saved ``backend_port``, or None when there is no file at all.

    Raises ``PortResolutionError`` when the file exists but cannot be turned
    into a port. See the module docstring for why those two cases differ.

    Paths are interpolated bare, never ``!r``: on Windows ``repr()`` doubles
    every backslash, so ``D:\\Coding\\…`` reaches the user as ``D:\\\\Coding\\\\…``
    and a search for their own path finds nothing. Values keep ``!r`` — there
    the quoting is what distinguishes ``"8000"`` from ``8000``.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None  # first launch
    except OSError as exc:
        raise PortResolutionError(
            f"cannot read the settings file {path}: {exc.strerror or exc}. "
            "Fix the file's permissions, or delete it to start from defaults."
        ) from exc
    except json.JSONDecodeError as exc:
        raise PortResolutionError(
            f"the settings file {path} is not valid JSON ({exc}). "
            "Fix it, or delete it to start from defaults."
        ) from exc

    if not isinstance(data, dict):
        raise PortResolutionError(
            f"the settings file {path} does not contain a JSON object."
        )

    application = data.get("application")
    if not isinstance(application, dict) or "backend_port" not in application:
        # A settings file without the key is not corrupt — it predates the key,
        # or holds other modules only. Defaulting is right; refusing is not.
        return None

    raw = application["backend_port"]
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        raise PortResolutionError(
            f"backend_port in {path} is {raw!r}, which is not a port number."
        )
    try:
        port = int(raw)
    except ValueError:
        raise PortResolutionError(
            f"backend_port in {path} is {raw!r}, which is not a number."
        ) from None

    if not (MIN_PORT <= port <= MAX_PORT):
        raise PortResolutionError(
            f"backend_port in {path} is {port}, outside {MIN_PORT}-{MAX_PORT}."
        )
    return port


def _settings_port_quietly(path: str) -> int | None:
    """The saved port, or None — never raising, for the container case only.

    A PRINCIPLED DIVERGENCE FROM ``_port_from_settings``, not an exception, and
    the distinction is what stops someone harmonising the two later:

    The desktop refusal exists *because the file is the port's source there* —
    an unreadable file means the port is unknown, and starting on a guess while
    the app believes something else is the defect this module removes. Inside a
    container the file is **not** the source; its value is discarded whatever it
    says. So refusing there would refuse for a reason unrelated to the
    malformation's actual consequence — a non sequitur that converts a file the
    operator cannot see into a container that will not start.

    This exists ONLY to say a more useful sentence than "your setting was
    ignored" — which port was set, and why it did not take. If it cannot find
    out, it says the generic thing instead.
    """
    try:
        return _port_from_settings(path)
    except PortResolutionError:
        return None
    except Exception:  # noqa: BLE001 - a diagnostic must never be the failure
        return None


def _container_port_notice(env, complain) -> None:
    """Explain why the saved port does not apply here, if it would have.

    KNOWN LIMIT of the detection this rides on, stated rather than asserted
    away: ``MRLN_CONTAINER`` is set by **this project's** ``entrypoint.sh``. A
    derived image with its own entrypoint, or the backend started inside a
    container some other way, will not have it and will get **desktop**
    behaviour — settings-driven port and all, including the stranding this
    guards against.

    That is the same declared-vs-observed shape as ``bind_host()``, which reads
    the ``--host`` uvicorn was *told* rather than the socket it *bound*. It is
    accepted for the same reason: every alternative container-detection
    heuristic (``/.dockerenv``, cgroup inspection, ``/proc/1/comm``) is more
    fragile and fails differently across runtimes. Meet this as a documented
    boundary, not as a surprise.
    """
    saved = _settings_port_quietly(settings_path(env))
    if saved is None or saved == DEFAULT_PORT:
        # Nothing to correct — stay quiet rather than logging on every boot.
        return
    complain(
        event="settings_port_ignored_in_container",
        saved=saved,
        using=DEFAULT_PORT,
        why=(
            "this is a container: the published port mapping (docker -p / the "
            "platform's pod template) lives outside this namespace, so a port "
            "chosen in Server Control cannot move it and would only make the "
            "container unreachable. Set the port on the host side of -p, or "
            "set PORT to match what the platform publishes."
        ),
    )


def _warn_to_stderr(**fields) -> None:
    """Default complaint channel: a launcher has no structured logger.

    Not ``pass``. A bad ``PORT`` is tolerated (see below) but must never be
    tolerated silently — that is the shape of defect this whole lane exists to
    remove (invariant #4).
    """
    detail = " ".join(f"{k}={v!r}" for k, v in fields.items())
    sys.stderr.write(f"[port_resolver] {detail}\n")


def resolve_port(
    argv: list[str] | None = None,
    *,
    environ=None,
    settings_fallback: int | None = None,
    warn=None,
) -> int:
    """The port this server should serve on. ONE producer (RULE-21).

    Precedence, and the reason for each step:

    1. ``--port`` on the command line — what uvicorn will actually honour, so
       nothing may contradict it. This is the same inversion ``bind_host()``
       uses for ``--host``: observe what was applied, do not recompute it.
    2. ``PORT`` in the environment — what the container sets, and what Docker's
       published-port mapping is built around (DECISION-11).
    3. ``settings_fallback`` when the caller already holds the loaded settings
       (``main.py`` does), otherwise the saved ``backend_port`` read from disk.
       Same value by two routes — a caller that has it must not pay for a
       second read, and a launcher that has no app must still get an answer.
    4. 8000.

    An unparseable ``PORT`` warns and falls through rather than refusing: it is
    pinned that way by the pipeline's tests, and a stray environment variable
    must not take a training run down. An unparseable SETTINGS FILE does refuse
    — see the module docstring for why those two differ.
    """
    argv = list(argv) if argv is not None else []
    env = os.environ if environ is None else environ
    complain = _warn_to_stderr if warn is None else warn

    explicit = _port_from_argv(argv)
    if explicit is not None:
        return explicit

    raw_env = (env.get("PORT") or "").strip()
    if raw_env:
        try:
            return int(raw_env)
        except ValueError:
            complain(event="invalid_port_env", value=raw_env)

    # THE SETTINGS FILE IS NOT A PORT SOURCE INSIDE A CONTAINER, and this is the
    # step that makes DECISION-11 true rather than merely stated.
    #
    # Without it: RunPod sets PORT, but a plain `docker run -p 8000:8000` does
    # NOT, so the saved setting would drive the bind while the published mapping
    # stays 8000 in the daemon — where nothing inside the namespace can reach
    # it. An operator who changes the port in Server Control and restarts would
    # strand themselves, and `PORT="${PORT:-8000}"` used to make that
    # unreachable. Restoring the property, rather than relying on a UI note
    # nobody may read.
    #
    # 8000 and not a refusal: someone who set a port before upgrading would get
    # a container that stops booting, which turns a stranding into an outage.
    # The setting is ignored loudly instead — see _container_port_notice.
    if env.get("MRLN_CONTAINER") == "1":
        _container_port_notice(env, complain)
        return DEFAULT_PORT

    if settings_fallback is not None:
        return settings_fallback

    from_settings = _port_from_settings(settings_path(env))
    if from_settings is not None:
        return from_settings

    return DEFAULT_PORT


def _main() -> int:
    """Print the resolved port, for a launcher to capture.

    Prints ONLY the number on success so a shell can use it directly. On
    failure prints the reason to stderr and exits non-zero, so the launcher
    refuses rather than starting on a port the app will not agree with.
    """
    try:
        sys.stdout.write(f"{resolve_port(sys.argv[1:])}\n")
    except PortResolutionError as exc:
        sys.stderr.write(f"cannot determine the backend port: {exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
