"""SECURITY.md's claims, pinned to the code that implements them.

`DOCS_STYLE.md` §2 lists SECURITY.md as a public surface whose role is "what the
product does with files, keys, network". It did not exist until 2026-08-28; the
release audit found the gap.

Its first draft named `backend/tests/test_api_path_traversal.py`, which has never
existed -- the real pin is `api/test_path_traversal_containment.py` -- and said
the container "runs as a non-root user" when the Dockerfile deliberately starts
as root to chown a mounted volume and drops privileges in the entrypoint. Two
false claims in a document whose own preamble promises that every claim names the
file implementing it. Written from memory rather than from the tree, which is
precisely the failure this file exists to prevent.

Two-sided, per the README-claims lesson: a path must EXIST (so a reference cannot
rot into a dead pointer) and a stated NUMBER must equal the constant it describes
(so the two cannot drift apart while each looks fine alone). A check on only the
first half passes a document that names real files and describes them wrongly.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SECURITY = REPO / "SECURITY.md"

#: Top-level files the document references as SOURCE, which have no directory
#: prefix to identify them by.
_TOP_LEVEL_SOURCE = frozenset({"Dockerfile", "entrypoint.sh"})


def _text() -> str:
    return SECURITY.read_text(encoding="utf-8")


def _claimed_paths() -> set[str]:
    """Backticked tokens that reference a file IN THIS REPOSITORY.

    The document backticks two different kinds of thing and only one of them is
    a repo path. `backend/app/core/url_guard.py` is a file a reader can open;
    `settings.json`, `./outputs` and `captions/<definition>/` are runtime state
    that exists on the user's machine, at a location the user chooses, and
    asserting those exist here would fail on every clean checkout.

    So the rule is "rooted at a real top-level directory of this repository",
    plus a named set for the two source files that live at the root. That is a
    property of the token rather than a list of exceptions, which matters: an
    exception list only ever knows about the false positives someone already
    met, and the next runtime noun added to the document would fail the gate.
    """
    found = set()
    for token in re.findall(r"`([^`]+)`", _text()):
        if "<" in token or token.startswith("./") or re.search(r"[ (),]", token):
            continue
        if token in _TOP_LEVEL_SOURCE:
            found.add(token)
            continue
        head, _, rest = token.partition("/")
        if rest and (REPO / head).is_dir():
            found.add(token)
    return found


def test_security_md_exists_at_all():
    """DOCS_STYLE §2 lists it as a public surface. It was missing for months."""
    assert SECURITY.is_file(), (
        "SECURITY.md is gone. DOCS_STYLE §2 lists it as a public surface; a public "
        "repository without one tells a finder to open an issue instead."
    )


def test_the_extractor_finds_claims():
    """Anti-vacuity: the path check below passes on an empty set."""
    assert len(_claimed_paths()) >= 8, (
        f"only {len(_claimed_paths())} paths extracted from SECURITY.md; the document's "
        "shape changed and the check below is now vacuous"
    )


def test_every_path_the_document_names_exists():
    missing = sorted(p for p in _claimed_paths() if not (REPO / p).exists())
    assert not missing, (
        f"SECURITY.md names paths that do not exist: {missing}. Its own preamble promises "
        "that every claim names the file implementing it, so a dead pointer there is worse "
        "than no pointer -- it reads as verifiable and is not."
    )


def test_the_upload_ceiling_matches_the_constant():
    """The number, not just the file that holds it."""
    guard = (REPO / "backend" / "app" / "api" / "_upload_guard.py").read_text(encoding="utf-8")
    m = re.search(r"MAX_UPLOAD_BYTES\s*=\s*(\d+)\s*\*\s*1024\*\*(\d)", guard)
    assert m, "MAX_UPLOAD_BYTES is no longer written as `<n> * 1024**<p>`; re-read it here"
    unit = {2: "MiB", 3: "GiB", 4: "TiB"}[int(m.group(2))]
    assert f"{m.group(1)} {unit}" in _text(), (
        f"SECURITY.md does not state the real upload ceiling ({m.group(1)} {unit}). A cap "
        "quoted in prose and a cap enforced in code drift apart silently, and the prose is "
        "what a user plans around."
    )


def test_the_default_bind_host_matches_the_constant():
    cfg = (REPO / "backend" / "app" / "core" / "container_config.py").read_text(encoding="utf-8")
    m = re.search(r'DEFAULT_BIND_HOST\s*=\s*"([^"]+)"', cfg)
    assert m, "DEFAULT_BIND_HOST moved; re-read it here rather than deleting this check"
    assert f"`{m.group(1)}`" in _text(), (
        f"SECURITY.md does not name the real default bind host ({m.group(1)}). This is the "
        "claim that tells a reader whether the app is reachable from their network."
    )


@pytest.mark.parametrize("var", ["MRLN_AUTH_TOKEN"])
def test_environment_variables_it_names_are_real(var: str):
    hits = list((REPO / "backend" / "app").rglob("*.py"))
    assert any(var in f.read_text(encoding="utf-8", errors="replace") for f in hits), (
        f"SECURITY.md tells the user to set {var}, and no Python file under backend/app "
        "reads it. Instructions for a variable nothing consumes are worse than none."
    )
    assert var in _text(), f"{var} left SECURITY.md; move this check with it"
