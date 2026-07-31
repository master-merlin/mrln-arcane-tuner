"""Shared path-safety utilities for API routes.

Provides:
- Path traversal validation (containment checks)
- Safe file deletion with Windows retry logic
- Safe directory removal with Windows retry logic
- Filename sanitization for uploads
- Audio/image-op guard (reject image-only operations on audio files)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import HTTPException

from app.core.dataset.media_types import AUDIO_EXTENSIONS
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


# ── Operator-tool filesystem roots ───────────────────────────────────────
#
# The operator-facing tools that take an ABSOLUTE path from the client
# (checkpoint inspect, LoRA inspect/resize, folder browse, upscale-model
# listing) previously each carried their own private copy of this list —
# three near-identical ``_ALLOWED_ROOTS`` definitions that had already
# drifted, plus one route (upscale list-models) with no check at all.
#
# Two changes over those copies:
#   * Anchored on ``__file__``, NOT the process CWD. ``Path("outputs").resolve()``
#     depended on where the server was launched from — it lands on
#     ``backend/outputs`` for the normal ``cwd=backend`` start but on
#     ``<repo>/outputs`` (a directory that does not exist) if launched from the
#     repo root, silently widening or narrowing the allowlist.
#   * Scoped to the three DATA trees instead of the whole of ``backend/``.
#     The old root admitted ``venv/``, ``app/`` and ``arcane_tuner.db``, which
#     matters because ``/tools/lora/resize`` WRITES to a client-named path —
#     the broad root let it overwrite the database or a venv binary.
_BACKEND_ROOT = Path(__file__).resolve().parents[2]  # backend/

ALLOWED_FS_ROOTS: tuple[Path, ...] = (
    _BACKEND_ROOT / "outputs",
    _BACKEND_ROOT / "models",
    _BACKEND_ROOT / "datasets",
)


def validate_path_in_allowed_roots(candidate: str | Path) -> Path:
    """Resolve *candidate* and verify it lives under one of :data:`ALLOWED_FS_ROOTS`.

    The absolute-path counterpart to :func:`validate_path_within` (which
    contains a client path inside one specific dataset root). Returns the
    resolved ``Path``; raises ``HTTPException(403)`` naming the allowed roots
    so the user knows where to move the file.
    """
    resolved = Path(candidate).resolve()
    if any(resolved.is_relative_to(root) for root in ALLOWED_FS_ROOTS):
        return resolved

    logger.warning(
        "path_outside_allowed_roots",
        candidate=str(candidate),
        roots=[str(r) for r in ALLOWED_FS_ROOTS],
    )
    allowed = ", ".join(str(root) for root in ALLOWED_FS_ROOTS)
    raise HTTPException(
        status_code=403,
        detail=(
            "Access denied: path is outside allowed directories. "
            f"Allowed: {allowed}"
        ),
    )


def reject_audio_op(path: str | Path, op: str = "This operation") -> None:
    """Raise ``HTTPException(400)`` if *path*'s extension is an audio type.

    Shared guard for the image-only routes (crop, adjustments, upscale,
    color-match, histogram, overlay/pipeline render, masking) that open the
    file with ``PIL.Image`` and would otherwise surface an opaque decode
    crash (or a generic 500) for an audio file. Matched case-insensitively.
    """
    ext = os.path.splitext(str(path))[1].lower()
    if ext in AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"{op} is not supported for audio files.",
        )


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
