"""Shared path-safety utilities for API routes.

Provides:
- Path traversal validation (containment checks)
- Safe file deletion with Windows retry logic
- Safe directory removal with Windows retry logic
- Filename sanitization for uploads
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import HTTPException

from app.core.logger import get_logger

logger = get_logger(__name__)

_MAX_RETRIES = 2
_RETRY_DELAY = 0.25  # seconds


def validate_path_within(candidate: str | Path, root: str | Path) -> Path:
    """Resolve *candidate* and verify it lives inside *root*.

    Returns the resolved ``Path`` on success.
    Raises ``HTTPException(403)`` if the resolved path escapes *root*.
    """
    root_resolved = Path(root).resolve()
    candidate_resolved = Path(candidate).resolve()

    # Use is_relative_to (Python 3.9+)
    if not candidate_resolved.is_relative_to(root_resolved):
        logger.warning(
            "path_traversal_blocked",
            candidate=str(candidate),
            root=str(root_resolved),
        )
        raise HTTPException(
            status_code=403,
            detail="Access denied: path is outside the allowed directory.",
        )
    return candidate_resolved


def sanitize_filename(raw: str) -> str:
    """Strip directory components from an uploaded filename.

    ``../../etc/passwd`` → ``passwd``
    ``C:\\Users\\secret\\file.txt`` → ``file.txt``
    """
    return Path(raw).name


def safe_remove(path: str | Path) -> bool:
    """Delete a file with one retry on ``PermissionError`` (Windows AV / indexer).

    Returns ``True`` if deleted, ``False`` if the file did not exist.
    Logs a warning and re-raises on persistent failure.
    """
    p = Path(path)
    for attempt in range(_MAX_RETRIES):
        try:
            p.unlink(missing_ok=True)
            return True
        except PermissionError:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY)
            else:
                logger.warning("safe_remove_failed", path=str(p))
                raise
    return False


def safe_rmtree(path: str | Path) -> bool:
    """Remove a directory tree with retry-on-error for Windows locking.

    Uses the Python 3.12 ``onexc`` parameter for ``shutil.rmtree``.
    Returns ``True`` if removed, ``False`` if the directory did not exist.
    """
    import shutil

    p = Path(path)
    if not p.is_dir():
        return False

    def _on_error(func, err_path, exc):  # noqa: ANN001
        """Retry once after a short sleep (Windows file locking)."""
        time.sleep(_RETRY_DELAY)
        try:
            func(err_path)
        except OSError:
            logger.warning("rmtree_retry_failed", path=str(err_path), error=str(exc))

    shutil.rmtree(p, onexc=_on_error)
    return True
