"""The restart contract between this server and whatever started it (LANE-56).

THE ONE PRODUCER (RULE-21) of two facts the restart path needs:

* ``RESTART_EXIT_CODE`` — the sentinel exit code that means "relaunch me";
* ``is_supervised()`` — whether something is listening for that sentinel.

The contract, stated once, here, for anyone who writes a supervisor:

1. The supervisor sets ``MRLN_SUPERVISED=1`` in the server's environment
   before launching it. Nothing else sets it: not the app, not a user, not a
   settings file.
2. When the server exits with code 75 (``RESTART_EXIT_CODE``), the supervisor
   relaunches it **in the same console** — the terminal the user started it
   from is where the shutdown and the new startup are both seen.
3. The supervisor sets ``MRLN_RESTART=1`` on the relaunch (the app's one
   reader is ``app/main.py``; it suppresses the browser auto-open and reports
   a pending restart failure) and **re-resolves the port** before it (the
   settings may have moved it; ``port_resolver.py`` is the one producer).
4. Any other exit code is the supervisor's own exit code too — **a crash
   must not loop**. Only the sentinel relaunches.

Why an exit code and not a spawn: the process that is dying cannot be the one
that starts the replacement — its child is an orphan by construction, invisible
in the console that launched the original (``e2e3cfc8`` had to cut the stdio
inheritance because a dead IDE pipe wedged the child's logging lock). On
Windows the console belongs to the process that launched the original server,
so the thing that owns the terminal must do the relaunch. Both supervisors —
``backend/start_backend.bat`` and ``entrypoint.sh`` — implement this contract
and carry the literal ``75``; their tests pin it to the constant below.

Absent ``MRLN_SUPERVISED`` (a bare ``uvicorn`` launch), the restart falls back
to ``restart_launcher.py`` exactly as LANE-51/56 left it, and says so.

Children of a supervised server (the frontend dev server, trainer processes)
inherit ``MRLN_SUPERVISED=1`` harmlessly: none of them exits 75, and none of
them reads this module.

Stdlib only, on purpose: this sits on the restart path of both callers (the
route and the self-updater) and must never fail to import (ARCHITECTURE D1).
"""

from __future__ import annotations

import os

#: "Relaunch me." BSD ``EX_TEMPFAIL``: collides with nothing a server on this
#: platform emits — 1 (``TerminateProcess`` / an uncaught Python error), 2
#: (usage), 3 (uvicorn ``STARTUP_FAILURE``, ``uvicorn/config.py:80``), 130/143
#: (signals). Reserved in ECOSYSTEM §6 2026-09-02; frozen at the release that
#: ships it (ARCHITECTURE D2).
RESTART_EXIT_CODE = 75

#: The environment variable a supervisor sets. Its value is exactly ``"1"``.
SUPERVISED_ENV = "MRLN_SUPERVISED"


def is_supervised() -> bool:
    """Is a supervisor listening for ``RESTART_EXIT_CODE``?

    Strict ``== "1"`` and read at call time, never at import: an operator's
    stray ``MRLN_SUPERVISED=yes`` in a bare terminal must keep the launcher
    fallback, because an exit that nothing answers is a server that never
    comes back.
    """
    return os.environ.get(SUPERVISED_ENV) == "1"
