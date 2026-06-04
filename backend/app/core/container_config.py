"""Container / environment-driven configuration.

Centralizes the env-var overrides used when running inside a container
(e.g. on RunPod). Every value falls back to local-dev defaults so a
normal ``start_backend`` run is unaffected.
"""
from __future__ import annotations

import os

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


def frontend_dist_dir() -> str | None:
    """Directory holding the built Angular SPA, or ``None`` if absent.

    ``MRLN_FRONTEND_DIST`` overrides; otherwise the default build output
    ``frontend/dist/frontend/browser`` is used when it exists.
    """
    explicit = os.environ.get("MRLN_FRONTEND_DIST")
    if explicit:
        if os.path.isdir(explicit):
            return explicit
        logger.warning("frontend_dist_missing", path=explicit)
        return None
    candidate = os.path.join(
        _PROJECT_ROOT, "frontend", "dist", "frontend", "browser"
    )
    return candidate if os.path.isdir(candidate) else None
