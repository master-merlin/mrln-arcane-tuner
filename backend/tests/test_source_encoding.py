"""Every tracked text file is plain UTF-8: no BOM, nothing double-encoded.

This exists because on 2026-09-03 a PowerShell write put a UTF-8 BOM on
`backend/requirements.txt`. `packaging.Requirement` cannot parse a line that
starts with U+FEFF, so nine tests in `test_server_boot_contract.py` failed at
once with a syntax error about a package name -- pointing at the parser, not at
the byte. The dependency data in the file was entirely correct.

The same write double-encoded the em-dashes in that file's comments, and a
sweep of the repository then found the damage was not new: six tracked files
carried a BOM and four carried mojibake, including one that had been shipping a
corrupted em-dash inside a user-facing toast message ("Imported 12 modules
<mojibake> 9 matched").

That range is the point. The BOM is fatal to exactly one parser and inert
everywhere else -- YAML, TypeScript, Python and PowerShell all accept it -- so
nothing failed until the one file that feeds `packaging` acquired one. Mojibake
never fails anything at all; it just degrades text, in comments where nobody
looks and occasionally in a string where users do. Neither is visible in a diff
unless you already suspect it. So the guard has to be mechanical and repo-wide,
not a rule about how to write files.

Repairing it is mechanical too, and the repair is NOT retyping the line:
`damaged.encode("cp1252").decode("utf-8")` restores the original text exactly,
because that is precisely the transformation that damaged it.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

#: Text formats a human edits and a tool parses. Binary and generated files are
#: excluded: a `.svg` or a lockfile is not hand-written prose, and `.po`/`.ipynb`
#: legitimately carry escaped byte sequences that would read as false positives.
TEXT_SUFFIXES = frozenset(
    {
        ".bat", ".cfg", ".css", ".html", ".ini", ".json", ".md", ".ps1", ".py",
        ".scss", ".sh", ".toml", ".ts", ".txt", ".yaml", ".yml",
    }
)

#: The leading byte of a UTF-8 multi-byte sequence as cp1252 renders it. A cheap
#: trigger only -- the decision is the round trip below, never this set.
_MOJIBAKE_LEAD = ("Â", "Ã", "â")

_BOM = "﻿"


def _tracked_text_files() -> list[pathlib.Path]:
    try:
        listing = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO,
            capture_output=True,
            check=True,
            timeout=60,
        ).stdout.decode("utf-8")
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git is not available to enumerate tracked files")
    names = [n for n in listing.split("\0") if n]
    return [
        REPO / n
        for n in names
        if pathlib.PurePosixPath(n).suffix.lower() in TEXT_SUFFIXES
    ]


def _is_mojibake(line: str) -> bool:
    """True when this line is UTF-8 that was read as cp1252 and written back.

    The round trip is the definition of the damage, not an approximation of it.
    Legitimately accented prose fails the decode -- an i-diaeresis is a lone
    0xEF byte, which is not a valid UTF-8 sequence -- and is left alone, which
    matters because this repository's comments are written in prose.
    """
    if not any(lead in line for lead in _MOJIBAKE_LEAD):
        return False
    try:
        return line.encode("cp1252").decode("utf-8") != line
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False


def test_no_tracked_text_file_starts_with_a_byte_order_mark():
    """The half that is silently fatal, in one file, to one parser."""
    files = _tracked_text_files()
    assert files, "enumerated zero tracked text files -- the filter is wrong"
    offenders = [
        f.relative_to(REPO).as_posix()
        for f in files
        if f.read_text(encoding="utf-8", errors="replace").startswith(_BOM)
    ]
    assert not offenders, (
        "these tracked files begin with a UTF-8 BOM (U+FEFF): "
        f"{offenders}. Most parsers ignore it and one does not -- "
        "`packaging.Requirement` fails on the first line of requirements.txt, "
        "which is how nine boot-contract tests went red on 2026-09-03. Rewrite "
        "each file as UTF-8 with no BOM. In PowerShell that means "
        "`[IO.File]::WriteAllText(path, text, (New-Object Text.UTF8Encoding "
        "$false))` -- `Out-File`/`>` write the BOM by default."
    )


def test_no_tracked_text_file_contains_double_encoded_text():
    """The half that never fails anything and quietly rots the prose."""
    files = _tracked_text_files()
    assert files, "enumerated zero tracked text files -- the filter is wrong"
    offenders: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            if _is_mojibake(line):
                offenders.append(f"{path.relative_to(REPO).as_posix()}:{number}")
                break
    assert not offenders, (
        f"these tracked files contain double-encoded text: {offenders}. UTF-8 "
        "was read as cp1252 and written back as UTF-8, so an em-dash became "
        "three characters. Nothing errors on it, which is why it accumulated "
        "unnoticed into a user-facing toast message. Repair rather than retype: "
        '`damaged.encode("cp1252").decode("utf-8")` restores the exact original.'
    )


def test_the_detector_would_actually_catch_the_2026_09_03_damage():
    """Negative control. Both assertions above pass on a clean tree, so without
    this they are indistinguishable from two assertions that check nothing.

    The damaged fixture is BUILT here rather than written out. A literal one
    makes this file itself an offender, which is not hypothetical: the first
    version of this test was written with the mojibake inline, and the moment
    the file was committed the repo-wide assertion above failed on line 132 of
    its own guard. An exclusion list would have been the wrong answer -- it
    would carve a permanent hole in the check for the one file most likely to
    contain examples of what it looks for.
    """
    clean = "hpsv2==1.2.0  # its dev deps leaked — it never imports them"
    damaged = clean.encode("utf-8").decode("cp1252")
    assert damaged != clean, "the fixture did not actually get damaged"
    assert _is_mojibake(damaged)
    assert damaged.encode("cp1252").decode("utf-8") == clean
    assert not _is_mojibake(clean), "the em-dash original must not be flagged"
    assert (_BOM + "accelerate==1.14.0").startswith(_BOM)


def test_the_detector_leaves_legitimate_non_ascii_prose_alone():
    """The other half of the control: a checker that flagged every non-ASCII
    byte would be unusable here, where comments carry em-dashes and arrows."""
    for good in (
        "numpy==2.3.5  # held — sam3 caps <2.4",
        "# cap 0: full-length valid → 20 user tokens survive",
        "# a naïve reader would tidy this line away",
        "# ── Task 1: Family Registration ──",
    ):
        assert not _is_mojibake(good), f"false positive on {good!r}"
