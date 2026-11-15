"""Write a file by replacing its name, never by writing through it.

An output name that already exists may be a link: a symlink, or a hard link
that no path check can tell from a regular file. Opening it for writing would
write into whatever it points at. :func:`replace_with_fresh_file` writes to a
temp file it creates itself (exclusively, beside *dst*), then ``os.replace``\\ s
it onto *dst*: a link at *dst* is replaced as a name, a stale or planted
``*.tmp`` is never touched, and a reader never sees a half-written file.
"""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


@contextmanager
def replace_with_fresh_file(dst: str | Path) -> Iterator[BinaryIO]:
    """Yield a new binary file in *dst*'s directory; on success move it onto
    *dst*, on any failure remove it and re-raise."""
    dst = os.fspath(dst)
    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(dst) or ".", prefix=f"{os.path.basename(dst)}.", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as f:
            yield f
        # mkstemp creates 0600; keep the replaced file's permission bits (a
        # new file gets what a plain open would usually give it).
        try:
            mode = stat.S_IMODE(os.stat(dst).st_mode)
        except OSError:
            mode = 0o644
        os.chmod(tmp, mode)
        os.replace(tmp, dst)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
