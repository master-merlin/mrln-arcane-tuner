"""Container / environment-driven configuration.

Centralizes the env-var overrides used when running inside a container
(e.g. on RunPod). Every value falls back to local-dev defaults so a
normal ``start_backend`` run is unaffected.
"""
from __future__ import annotations

import ipaddress
import os
import sys

from app.core.logger import get_logger

# this file: backend/app/core/container_config.py
_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/app
_BACKEND_DIR = os.path.dirname(_APP_DIR)                                 # backend
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)                            # project root

logger = get_logger(__name__)


def is_container() -> bool:
    """True when running inside the container image (set by entrypoint).

    The container entrypoint sets ``MRLN_CONTAINER`` to exactly ``"1"`` —
    no other truthy values are accepted intentionally, so a future caller
    should not try to broaden this to generic boolean parsing.
    """
    return os.environ.get("MRLN_CONTAINER") == "1"


def resolve_port(default: int = 8000) -> int:
    """Single exposed port. ``PORT`` env wins, else the provided default."""
    raw = os.environ.get("PORT")
    if raw:
        try:
            return int(raw)
        except ValueError:
            logger.warning("invalid_port_env", value=raw, fallback=default)
    return default


def auth_token() -> str:
    """Shared access token; empty string means auth is disabled."""
    return os.environ.get("MRLN_AUTH_TOKEN", "").strip()


#: Uvicorn's own default when no ``--host`` is given.
DEFAULT_BIND_HOST = "127.0.0.1"

#: Hosts that mean "this machine only". ``::`` and ``0.0.0.0`` are the
#: wildcards; everything else is treated as reachable unless it resolves to a
#: loopback address.
_LOOPBACK_LITERALS = frozenset({"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"})


def _host_from_argv(argv: list[str] | None = None) -> str | None:
    """Read ``--host`` out of this process's own command line, if present.

    WHY THIS EXISTS, precisely: uvicorn binds the socket AFTER the ASGI
    lifespan runs (``uvicorn.server.Server.startup`` awaits
    ``lifespan.startup()`` before ``loop.create_server``, verified against
    uvicorn 0.52.0), so the application CANNOT observe its real bound address
    from inside lifespan — there is nothing bound yet to observe. That is why
    the audit's "add a startup check" was unimplementable as originally
    written.

    The command line is the next best evidence and it is genuine evidence:
    every launcher in this repo starts the server with ``python -m uvicorn …``,
    so ``--host`` is literally in ``sys.argv`` of this process. Reading it
    covers the case the environment variable cannot — somebody running uvicorn
    by hand with ``--host 0.0.0.0`` and no ``MRLN_BIND_HOST`` set.

    KNOWN LIMIT, stated rather than papered over: a programmatic
    ``uvicorn.run(host=...)`` puts nothing in argv, and a socket passed in by a
    process manager bypasses ``--host`` entirely. No launcher in this repo does
    either. This resolves the DECLARED bind address, not the observed one.
    """
    args = sys.argv if argv is None else argv
    for i, arg in enumerate(args):
        if arg == "--host":
            if i + 1 < len(args):
                return args[i + 1].strip()
            return None
        if arg.startswith("--host="):
            return arg.split("=", 1)[1].strip()
    return None


def bind_host(argv: list[str] | None = None) -> str:
    """The address this process intends to serve on. ONE producer (RULE-21).

    Precedence is deliberate: an explicit ``--host`` on the command line is
    what uvicorn will actually honour, so it outranks the environment
    variable. ``MRLN_BIND_HOST`` is what the launchers set; the default
    matches uvicorn's own so the answer is right even when neither is given.
    """
    explicit = _host_from_argv(argv)
    if explicit:
        return explicit
    return os.environ.get("MRLN_BIND_HOST", "").strip() or DEFAULT_BIND_HOST


def is_loopback_host(host: str) -> bool:
    """True when *host* reaches only this machine.

    An unparseable host is treated as NON-loopback. Failing closed on a value
    we do not understand is the whole point of the check — the alternative is
    a typo silently disabling the guard.
    """
    h = (host or "").strip().strip("[]").lower()
    if not h:
        # Empty means uvicorn's default, which is loopback.
        return True
    if h in _LOOPBACK_LITERALS:
        return True
    # 0.0.0.0 and :: are the wildcards — every interface, explicitly not loopback.
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def bind_is_exposed_without_auth(argv: list[str] | None = None) -> str | None:
    """Return a human-readable reason to refuse startup, or ``None`` to proceed.

    Returns a string rather than raising so the caller decides how to fail —
    nothing imported at startup may raise (ARCHITECTURE D1), and this module is
    imported at startup.
    """
    host = bind_host(argv)
    if is_loopback_host(host):
        return None
    if auth_token():
        return None
    return (
        f"Refusing to start: the server is bound to {host!r}, which is reachable "
        "from other machines, but MRLN_AUTH_TOKEN is empty so anyone who can "
        "reach this port has full control of your datasets, models and GPU.\n"
        "  Fix (either one):\n"
        "    * Set MRLN_AUTH_TOKEN=<a long random string> and sign in once; or\n"
        "    * Bind to this machine only with MRLN_BIND_HOST=127.0.0.1.\n"
        "  This is a deliberate breaking change in this release: earlier "
        "versions started an unauthenticated server on all interfaces."
    )


def frontend_dist_dir() -> str | None:
    """Directory holding the built Angular SPA, or ``None`` if absent.

    ``MRLN_FRONTEND_DIST`` overrides and is honored in any mode. Otherwise
    the default build output ``frontend/dist/frontend/browser`` is used only
    in container mode — so a stale local ``ng build`` output never changes
    the dev server's behavior.
    """
    explicit = os.environ.get("MRLN_FRONTEND_DIST")
    if explicit:
        if os.path.isdir(explicit):
            return explicit
        logger.warning("frontend_dist_missing", path=explicit)
        return None
    if not is_container():
        return None
    candidate = os.path.join(
        _PROJECT_ROOT, "frontend", "dist", "frontend", "browser"
    )
    return candidate if os.path.isdir(candidate) else None
