"""The shipped Windows scripts must be pure ASCII, and must actually parse.

On 2026-09-04 `backend/install.ps1` was on `main` in a state where Windows
PowerShell could not parse it at all -- 11 syntax errors, the installer dead for
every Windows user who ran it.

The mechanism is worth stating precisely, because it is invisible in a diff.
Windows PowerShell reads a .ps1 with no BOM as **cp1252**, not UTF-8. A UTF-8
em dash is `E2 80 94`; cp1252 decodes those three bytes as three characters, the
last of which is U+201D, a right double quotation mark -- and PowerShell accepts
smart quotes as STRING DELIMITERS. So a non-ASCII character sitting inside a
string literal silently closes that string early, and everything after it is
parsed as code. In `install.ps1` the trigger was a package emoji:

    Write-Host "<emoji> Installing $SD (--no-deps) ..." -ForegroundColor Cyan

which ended the string before `--no-deps`, and the parser reported
"Expression missing after unary operator '--'" -- on a line that was fine.
**The reported error location is not the offending line**, which is most of why
this is hard to see.

How it got there: the file had shipped with a BOM, which made PowerShell read it
as UTF-8, and a repo-wide encoding repair removed the BOM as damage. That is
correct for `requirements.txt`, where a BOM breaks `packaging.Requirement`, and
exactly wrong here. One rule cannot serve both, so these files are held to the
stricter one: **ASCII**. A BOM would also work for .ps1 but not for .bat
(cmd.exe mishandles a leading BOM), and a BOM must be preserved by every editor,
formatter and repair tool that ever touches the file -- one of which is what
removed it. ASCII cannot be got wrong.

Two checks, deliberately different in kind. The ASCII one is the cause and runs
everywhere. The parse one is the effect: it is the only check that would still
catch this if someone invented a new way to write an unparsable script, and it
is Windows-only, so it SKIPS elsewhere rather than passing quietly.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

SCRIPT_SUFFIXES = (".ps1", ".psm1", ".bat", ".cmd")


def _tracked_scripts() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    return [REPO_ROOT / f for f in out if f.lower().endswith(SCRIPT_SUFFIXES)]


def _non_ascii(path: Path) -> list[str]:
    """Line numbers and text of every line carrying a byte above 0x7F."""
    raw = path.read_bytes()
    hits = []
    for number, line in enumerate(raw.split(b"\n"), 1):
        if any(byte > 0x7F for byte in line):
            hits.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{number}")
    return hits


def test_every_shipped_script_is_pure_ascii():
    scripts = _tracked_scripts()
    assert scripts, "walked zero tracked scripts -- the glob or `git ls-files` is wrong"

    offenders = [hit for path in scripts for hit in _non_ascii(path)]
    assert not offenders, (
        f"these shipped scripts contain non-ASCII bytes: {offenders}. Windows "
        "PowerShell reads a BOM-less .ps1 as cp1252, where a UTF-8 em dash "
        "decodes to a right double quotation mark -- which PowerShell accepts as "
        "a string delimiter. A non-ASCII character inside a string literal closes "
        "it early and the rest of the file is parsed as code. Use ASCII: '-' for "
        "dashes and box rules, '[OK]'/'[!]'/'[+]' for status glyphs."
    )


# -- the effect, not the cause: does PowerShell actually accept the file? ----

_PARSE = (
    "$e=$null;$t=$null;"
    "[void][System.Management.Automation.Language.Parser]::ParseFile('{path}',[ref]$t,[ref]$e);"
    "if($e){{$e.Count}}else{{0}}"
)


def _powershell() -> str | None:
    if sys.platform != "win32":
        return None
    return shutil.which("powershell.exe")


def _parse_error_count(path: Path, shell: str) -> int:
    proc = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", _PARSE.format(path=path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return int((proc.stdout or "0").strip() or 0)


@pytest.mark.skipif(_powershell() is None, reason="Windows PowerShell only")
def test_every_shipped_powershell_script_parses():
    shell = _powershell()
    scripts = [p for p in _tracked_scripts() if p.suffix.lower() in (".ps1", ".psm1")]
    assert scripts, "walked zero tracked PowerShell scripts"

    broken = {
        p.relative_to(REPO_ROOT).as_posix(): n
        for p in scripts
        if (n := _parse_error_count(p, shell))
    }
    assert not broken, (
        f"these scripts do not parse under Windows PowerShell: {broken}. A text "
        "search proves a script SAYS the right thing; only the parser proves it "
        "RUNS. Note the reported error line is often not the offending line."
    )


@pytest.mark.skipif(_powershell() is None, reason="Windows PowerShell only")
def test_the_parse_check_actually_fails_on_the_original_defect(tmp_path):
    """Vacuity control, built as the real defect rather than described.

    The damaged file is CONSTRUCTED here (utf-8 bytes, no BOM) instead of being
    committed as a fixture, because a committed one would be non-ASCII and its
    own sibling check above would flag it -- the same trap the repo-wide
    encoding guard hit when its negative control became its own offender.
    """
    shell = _powershell()

    good = tmp_path / "good.ps1"
    good.write_text('Write-Host "Installing pkg (--no-deps) ..."\n', encoding="ascii")
    assert _parse_error_count(good, shell) == 0, "the control's clean file must parse"

    bad = tmp_path / "bad.ps1"
    bad.write_bytes('Write-Host "— Installing pkg (--no-deps) ..."\n'.encode("utf-8"))
    assert _parse_error_count(bad, shell) > 0, (
        "a BOM-less utf-8 .ps1 with an em dash inside a string literal must fail "
        "to parse -- if this passes, the checker is not reading the file the way "
        "PowerShell does and proves nothing about the real scripts"
    )
