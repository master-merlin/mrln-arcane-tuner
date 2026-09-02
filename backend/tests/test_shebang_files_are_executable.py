"""Every tracked file that opens with a shebang carries the executable bit in git.

Why this exists: CI runs ruff on Linux, where ``EXE001`` ("shebang is present
but file is not executable") fires; on the Windows box the rule is silent,
so ``update.py`` shipped with ``#!/usr/bin/env python3`` at mode 100644 and
the gate's first-ever CI run went red on a finding no local run could show.
The mode is git metadata, not a filesystem fact, so this reads the index
(``git ls-files -s``) — the same thing CI's checkout materialises.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _tracked_files_with_modes() -> dict[str, str]:
    out = subprocess.run(
        ["git", "ls-files", "-s"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    modes: dict[str, str] = {}
    for line in out.splitlines():
        # "<mode> <object> <stage>\t<path>"
        meta, _, path = line.partition("\t")
        modes[path] = meta.split()[0]
    return modes


def _has_shebang(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(2) == b"#!"
    except OSError:
        return False


def test_every_shebang_file_is_executable_in_git() -> None:
    if subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=REPO_ROOT,
                      capture_output=True).returncode != 0:
        pytest.skip("not a git checkout — nothing to read the modes from")
    modes = _tracked_files_with_modes()
    offenders = sorted(
        path for path, mode in modes.items()
        if path.endswith((".py", ".sh")) and mode != "100755"
        and _has_shebang(REPO_ROOT / path)
    )
    assert offenders == [], (
        "tracked files with a shebang but no executable bit (ruff EXE001 on Linux): "
        f"{offenders} — fix with `git update-index --chmod=+x <path>`"
    )


def test_the_check_sees_a_shebang(tmp_path: Path) -> None:
    """Positive control: the shebang detector is not vacuous."""
    p = tmp_path / "x.py"
    p.write_bytes(b"#!/usr/bin/env python3\n")
    assert _has_shebang(p)
    q = tmp_path / "y.py"
    q.write_bytes(b"import os\n")
    assert not _has_shebang(q)
