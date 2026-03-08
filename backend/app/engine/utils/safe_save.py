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

from safetensors.torch import save_file

from app.core.logger import get_logger

logger = get_logger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY = 1.0  # seconds


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
            os.replace(tmp_path, path)
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

