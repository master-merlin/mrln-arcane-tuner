"""Publication hygiene: private development tooling must never become public.

`origin` is a public repository, so "is this path tracked?" is the same question
as "is this path published?". What is private is held back by `.gitignore`, and
that is exactly what release hardening tends to narrow: re-including a public
surface (`!.github/`, `!SECURITY.md`, ...) edits the same few lines that hide the
private material, and a single over-broad `!` -- or one `git add -f` -- publishes
it with no error and no diff anyone reads. Deletion is not a remedy either: a
blob that reaches a public remote stays in its history.

So this pins the *invariant* rather than the rules that currently happen to
enforce it: every tracked path starts with a name on a short public list. The
list names what is public, so this file never has to name what is not. It must
keep passing however `.gitignore` is rewritten.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

# backend/tests/<this file> -> backend/tests -> backend -> repo root.
# Anchored on __file__, never CWD (ARCHITECTURE D10 invariant 9): pytest is
# invoked from the repo root and from backend/ alike.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Every top-level name that is public. A new one is added here on purpose.
PUBLIC_TOP_LEVEL = frozenset({
    ".dockerignore",
    ".gitattributes",
    ".github",
    ".gitignore",
    "Dockerfile",
    "LICENSE",
    "NOTICE",
    "README.md",
    "SECURITY.md",
    "backend",
    "docker-build.ps1",
    "docker-verify-commit.ps1",
    "docs",
    "entrypoint.sh",
    "frontend",
    "update.bat",
    "update.ps1",
    "update.py",
    "update.sh",
})


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


def _unexpected_top_level(paths: list[str]) -> list[str]:
    """The paths whose first component is not on the public list, sorted."""
    return sorted(p for p in paths if p.split("/", 1)[0] not in PUBLIC_TOP_LEVEL)


def _staged_by_default(tmp_path: Path, files: list[str]) -> list[str]:
    """What `git add -A` stages from *files* under the real `.gitignore`, in a throwaway repo."""
    (tmp_path / ".gitignore").write_bytes((REPO_ROOT / ".gitignore").read_bytes())
    for rel in files:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x\n", encoding="utf-8")
    for args in (["init", "-q"], ["add", "-A"]):
        subprocess.run(["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True)
    proc = subprocess.run(
        ["git", "ls-files", "-z"], cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    return sorted(p for p in proc.stdout.split("\0") if p and p != ".gitignore")


@pytest.mark.parametrize("rel", [
    "_tooling/config.json",
    "_tooling/tools/run.py",
    "backend/_tooling/tools/run.py",
    "frontend/src/_notes/plan.txt",
    ".hidden-tools/config.json",
    "backend/.hidden-tools/config.json",
    # Under every path a re-include opens: the last matching rule wins.
    ".github/_tooling/config.json",
    ".github/workflows/_tooling/config.yml",
    ".github/.hidden-tools/config.json",
    "docs/_tooling/notes.md",
    "docs/.hidden-tools/notes.md",
    "backend/app/engine/models/families/some_family/vendor/_tooling/notes.md",
])
def test_a_private_directory_is_never_staged_by_default(tmp_path: Path, rel: str) -> None:
    """The guard above only speaks after a commit; ignoring is what stops `git add -A`."""
    assert _staged_by_default(tmp_path, [rel]) == []


@pytest.mark.parametrize("rel", [
    "frontend/src/app/__tests__/new.spec.ts",
    "backend/tests/_lint_data/new.py",
    ".github/workflows/new.yml",
    ".github/ISSUE_TEMPLATE/new.md",
    "docs/guides/new.md",
    "backend/app/engine/models/families/some_family/vendor/NOTES.md",
])
def test_a_public_path_still_stages(tmp_path: Path, rel: str) -> None:
    """Positive control: the rule above must not swallow what is public today."""
    assert _staged_by_default(tmp_path, [rel]) == [rel]


def test_every_tracked_path_is_under_a_public_top_level_name() -> None:
    """Nothing unexpected is tracked, whatever `.gitignore` currently says."""
    tracked = _tracked_paths()
    assert tracked, "git ls-files returned nothing -- the guard would pass vacuously"

    unexpected = _unexpected_top_level(tracked)

    assert not unexpected, (
        "these tracked paths would be published, and their top-level name is not "
        "on the public list:\n  "
        + "\n  ".join(unexpected)
        + "\n\nIf the name is meant to be public, add it to PUBLIC_TOP_LEVEL on purpose. "
        "If not, untracking it is not enough once it has been pushed -- the blobs "
        "remain in the public history. Check whether this commit has left the machine."
    )


def test_the_guard_catches_any_unexpected_top_level_name() -> None:
    """Prove the negative: without this, a dead check would pass for the wrong
    reason and go unnoticed for as long as nothing leaks."""
    sample = [
        "backend/app/main.py",
        "README.md",
        ".github/workflows/gate.yml",
        ".hidden-tools/notes.md",
        "private-tools/notes.md",
        "stray-file.txt",
    ]
    assert _unexpected_top_level(sample) == [
        ".hidden-tools/notes.md",
        "private-tools/notes.md",
        "stray-file.txt",
    ]
