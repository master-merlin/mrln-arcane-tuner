"""Safe save for safetensors — avoids Windows OS error 1224.

On Windows, ``safetensors.torch.load_file`` memory-maps the file.  If a
later ``save_file`` targets the *same path*, it fails because Windows
forbids writing to a file with an active memory-mapped section.

This module writes to a ``.tmp`` sibling first, then uses
``os.replace()`` to atomically swap the directory entry, which works
even when the old file is still memory-mapped.
"""

import os
import time
from collections.abc import Callable

from safetensors.torch import save_file

from app.core.logger import get_logger

logger = get_logger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY = 1.0  # seconds

# Windows refuses a rename while any process holds a handle on the file or inside
# the folder: access denied (5) or a sharing violation (32).
_FILE_LOCK_WINERRORS = frozenset({5, 32})
_FILE_LOCK_BACKOFF = (0.2, 0.5, 1.0, 2.0, 4.0, 8.0)  # seconds


def is_file_lock_error(error: BaseException) -> bool:
    """Whether ``error`` is Windows reporting a lock, not a real permission problem."""
    return isinstance(error, PermissionError) and getattr(error, "winerror", None) in _FILE_LOCK_WINERRORS


def retry_file_lock(rename: Callable[[str, str], None], src: str, dst: str) -> None:
    """``rename(src, dst)``, retried within a bound while Windows reports a file lock.

    Any other error raises at once, and so does the lock error once the bound
    is used up.
    """
    for delay in _FILE_LOCK_BACKOFF:
        try:
            rename(src, dst)
            return
        except PermissionError as e:
            if not is_file_lock_error(e):
                raise
            logger.warning("file_lock_retry", src=src, dst=dst, delay=delay, error=str(e))
            time.sleep(delay)
    rename(src, dst)


def safe_save_file(
    tensors: dict,
    path: str | os.PathLike,
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    """Write tensors to *path* without conflicting with existing mmap locks.

    1. Serialize to ``<path>.tmp``
    2. ``os.replace`` the temp file onto *path* (atomic on NTFS/ext4)
    3. Retry up to 3 times on Windows OS error 1224 (mmap lock)
    """
    path = str(path)
    tmp_path = path + ".tmp"

    for attempt in range(_MAX_RETRIES):
        try:
            # Clean up any stale temp file from a previous failed attempt
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            save_file(tensors, tmp_path, metadata=metadata)
            retry_file_lock(os.replace, tmp_path, path)
            return  # success
        except (OSError, Exception) as e:
            err_str = str(e)
            is_mmap_error = "1224" in err_str or "user-mapped" in err_str.lower()

            if is_mmap_error and attempt < _MAX_RETRIES - 1:
                logger.warning(
                    "safe_save_retry",
                    path=path,
                    attempt=attempt + 1,
                    delay=_RETRY_DELAY,
                )
                time.sleep(_RETRY_DELAY)
                continue

            # Final attempt failed or non-mmap error — clean up and raise
            logger.warning("safe_save_failed", path=path, error=err_str)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    logger.debug("tmp_cleanup_failed", tmp_path=tmp_path)
            raise

