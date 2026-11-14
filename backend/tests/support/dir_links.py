"""Make a name inside a test tree that resolves somewhere else.

Containment tests need a REAL link: a directory symlink on POSIX, a junction
on Windows (no privilege needed, unlike a symlink). A test that cannot make
one skips rather than passing vacuously.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


def link_dir(link: Path, target: Path) -> None:
    """Make *link* a directory link to *target* (junction on Windows)."""
    target.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        import _winapi

        try:
            _winapi.CreateJunction(str(target), str(link))
            return
        except OSError:
            pass
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"cannot create a directory link here: {e}")


def link_name_outside(link: Path, outside_file: Path) -> Path:
    """Make the file name *link* resolve outside its directory.

    A file symlink to *outside_file* where the platform allows one. Windows
    without the symlink privilege gets a junction to *outside_file*'s
    directory instead: the name still resolves outside, which is what a
    containment check has to refuse. Returns what the name now points at.
    """
    try:
        os.symlink(outside_file, link)
        return outside_file
    except (OSError, NotImplementedError):
        if sys.platform != "win32":
            raise
    link_dir(link, outside_file.parent)
    return outside_file.parent
