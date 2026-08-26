"""Publication hygiene: the private dev harness must never become public.

`origin` is a public repository, so "is this path tracked?" is the same question
as "is this path published?". The dev factory -- `_harness/` (rulebook, ledgers,
plans, UAT packs), `.claude/` (agent and skill definitions), `.agent/` (scratch,
audit drops) and `.superpowers/` -- is private, and today it is held back only by
blanket `.gitignore` rules (`.*/` on line 4 and `_harness/` on line 74).

Blanket rules are exactly what release hardening tends to narrow: re-including a
public surface (`!.github/`, `!SECURITY.md`, ...) edits the same few lines that
hide the factory, and a single over-broad `!` -- or one `git add -f` -- publishes
it with no error and no diff anyone reads. Deletion is not a remedy either: a
blob that reaches a public remote stays in its history.

So this pins the *invariant* (nothing private is tracked) rather than the rules
that currently happen to enforce it. It must keep passing however `.gitignore`
is rewritten.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

# backend/tests/<this file> -> backend/tests -> backend -> repo root.
# Anchored on __file__, never CWD (ARCHITECTURE D10 invariant 9): pytest is
# invoked from the repo root and from backend/ alike.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Path prefixes that are private to the dev factory and must never be tracked.
PRIVATE_PREFIXES = (
    "_harness/",
    ".agent/",
    ".claude/",
    ".superpowers/",
)

#: Tracked exception. The bridge is a *tracked* stub that collects the
#: untracked, junction-shared `_harness/tests/`; it lives at the repo root
#: precisely so the private half stays private. It is not under a private
#: prefix, and is listed here only to document why the root `tests/` directory
#: is allowed to exist at all.
KNOWN_TRACKED_BRIDGE = "tests/test_harness_shared.py"


def _tracked_paths() -> list[str]:
    """Every path git currently tracks, as forward-slash repo-relative strings."""
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip(f"git unavailable or not a repository: {proc.stderr.strip()}")
    return [p for p in proc.stdout.split("\0") if p]


def test_no_private_harness_path_is_tracked() -> None:
    """No dev-factory path is tracked, whatever `.gitignore` currently says."""
    tracked = _tracked_paths()
    assert tracked, "git ls-files returned nothing -- the guard would pass vacuously"

    leaked = sorted(p for p in tracked if p.startswith(PRIVATE_PREFIXES))

    assert not leaked, (
        "private dev-harness paths are tracked and would be published:\n  "
        + "\n  ".join(leaked)
        + "\n\nUntracking them is not enough once they have been pushed -- the blobs "
        "remain in the public history. Check whether this commit has left the machine."
    )


def test_guard_detects_a_leak() -> None:
    """Prove the negative: the matching logic actually catches a private path.

    Without this, a typo in PRIVATE_PREFIXES would make the guard above pass
    for the wrong reason and go unnoticed for as long as nothing leaks.
    """
    sample = [
        "backend/app/main.py",
        "_harness/FACTORY.md",
        ".claude/agents/mrln-coder.md",
    ]
    leaked = sorted(p for p in sample if p.startswith(PRIVATE_PREFIXES))
    # Sorted, so "." (0x2E) precedes "_" (0x5F).
    assert leaked == [".claude/agents/mrln-coder.md", "_harness/FACTORY.md"]


def test_harness_bridge_stays_outside_the_private_tree() -> None:
    """The one tracked harness-adjacent file must not sit under a private prefix.

    If the bridge is ever moved into `_harness/`, the guard above starts failing
    and the real fix would look like loosening PRIVATE_PREFIXES. Pin the layout
    instead so that pressure never arrives.
    """
    assert not KNOWN_TRACKED_BRIDGE.startswith(PRIVATE_PREFIXES)
