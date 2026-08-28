"""Contracts for the user-facing updater (`update.py` + its launchers).

This script is what a user runs after a release, so its failure modes are the
expensive kind: it either silently leaves a machine out of date, or it does
something destructive to a working tree it does not own. The tests here are
organised around those two, not around line coverage.

The design it pins: the script asks **"does installed match declared?"**, never
"did this pull change it?". Delta-detection is what the server's own
`SelfUpdateService` does, correctly, because a container starts from a known
state. A person's checkout drifts for reasons unrelated to the pull in front of
it — both halves of that were live in this repo on 2026-08-28 (a venv on the
previous test runner, and `node_modules` holding lucide 1.16.0 against a
lockfile saying 1.18.0), and neither was caused by the pull that exposed it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DRIVER = REPO / "update.py"
LAUNCHERS = ("update.sh", "update.ps1", "update.bat")


@pytest.fixture(scope="module")
def upd():
    if not DRIVER.is_file():
        pytest.skip("update.py not in this checkout")
    spec = importlib.util.spec_from_file_location("mrln_update_under_test", str(DRIVER))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── The container split ─────────────────────────────────────────────────────

def test_container_detection_matches_the_backend_contract(upd):
    """The script copies `is_container` instead of importing it, because it must
    run before the venv is usable. That copy is only safe while it agrees.

    `container_config.is_container` accepts `"1"` and nothing else, deliberately
    — a comment there warns against broadening it to truthy parsing. If either
    side changes, this fails rather than letting the updater disagree with the
    server about where it is running.
    """
    from app.core.container_config import is_container as backend_is_container

    cases = ["1", "0", "", "true", "false", "True", "yes", " 1", "1 "]
    for value in cases:
        os.environ["MRLN_CONTAINER"] = value
        assert upd.is_container() == backend_is_container(), (
            f"MRLN_CONTAINER={value!r}: the updater and the backend disagree "
            "about whether this is a container"
        )
    os.environ.pop("MRLN_CONTAINER", None)
    assert upd.is_container() == backend_is_container()


def test_it_refuses_to_run_inside_a_container(upd):
    """Two updaters writing one checkout is the failure being prevented.

    Behavioural, not a unit call: the refusal must happen before any git or
    install work, so the whole program is run and its exit code read.
    """
    env = {**os.environ, "MRLN_CONTAINER": "1"}
    result = subprocess.run(
        [sys.executable, str(DRIVER), "--check", "--no-pull"],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert result.returncode == 2, (
        f"expected the container refusal (exit 2), got {result.returncode}"
    )
    assert "container" in result.stdout.lower()
    assert "update" in result.stdout.lower(), "the refusal must point somewhere"


def test_outside_a_container_it_proceeds(upd):
    """The negative half. Without it, a refusal that fired unconditionally
    would satisfy the test above and break the script for every real user."""
    env = {k: v for k, v in os.environ.items() if k != "MRLN_CONTAINER"}
    result = subprocess.run(
        [sys.executable, str(DRIVER), "--check", "--no-pull"],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        check=False,
    )
    assert result.returncode in (0, 1), (
        f"expected a check result (0 in sync / 1 out of date), got "
        f"{result.returncode}:\n{result.stdout}\n{result.stderr}"
    )
    assert "Backend packages" in result.stdout


# ── It must not damage a working tree it does not own ───────────────────────

def test_it_refuses_a_dirty_working_tree(upd, tmp_path, monkeypatch):
    """Uncommitted work is the user's. The script stops rather than stashing,
    resetting or merging around it."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, timeout=120)
    (tmp_path / "file.txt").write_text("uncommitted", encoding="utf-8")
    monkeypatch.setattr(upd, "REPO", tmp_path)
    assert upd.preflight() is None, "a dirty tree must stop the update"


def test_it_never_resets_or_stashes(upd):
    """A guard on the SHAPE of the git commands, because the destructive verbs
    are the ones a future edit is most likely to reach for when a pull refuses.

    `reset --hard` is exactly what `SelfUpdateService` does, and it is right
    there — so the tempting fix for "my branch diverged" is to copy it.

    Matched at the CALL SITE, not on the word. A first draft searched the whole
    file for "reset" and failed on the sentence explaining what the container's
    updater does — a guard that forbids naming a thing punishes documenting it,
    which is its own entry in LESSONS. What matters is what `git(...)` is asked
    to run, so that is what is read.
    """
    import re

    source = DRIVER.read_text(encoding="utf-8")
    invocations = re.findall(r"\bgit\(\s*(.*?)\)", source, re.S)
    assert invocations, "no git(...) calls found - this guard is reading nothing"

    forbidden = ("reset", "--hard", "stash", "clean", "-f", "--force")
    for call in invocations:
        args = re.findall(r'"([^"]*)"', call)
        for arg in args:
            assert arg not in forbidden, (
                f"update.py runs `git {' '.join(args)}`. This script "
                "acts on a user's working tree; destructive git verbs belong in "
                "SelfUpdateService, whose checkout is disposable."
            )
    flat = [a for call in invocations for a in re.findall(r'"([^"]*)"', call)]
    assert "--ff-only" in flat, "the pull must stay fast-forward-only"


# ── The sync checks, and the false positive that nearly shipped ─────────────

def _lock(packages: dict) -> str:
    return json.dumps({"name": "frontend", "lockfileVersion": 3, "packages": packages})


def test_absent_optional_packages_are_not_drift(upd, tmp_path, monkeypatch):
    """A lockfile lists every platform's binaries; npm installs this platform's.

    The first version of this check reported **165** packages "differing" on a
    perfectly healthy tree, all of them `optional: true` entries like
    `@esbuild/aix-ppc64` that npm is correct to skip. A check that cries drift
    on a clean tree gets ignored, and then the real drift is ignored with it.
    """
    fe = tmp_path / "frontend"
    (fe / "node_modules").mkdir(parents=True)
    declared = {
        "node_modules/real": {"version": "1.0.0"},
        "node_modules/@esbuild/aix-ppc64": {
            "version": "0.28.0", "optional": True, "os": ["aix"], "cpu": ["ppc64"],
        },
    }
    installed = {"node_modules/real": {"version": "1.0.0"}}
    (fe / "package-lock.json").write_text(_lock(declared), encoding="utf-8")
    (fe / "node_modules" / ".package-lock.json").write_text(
        _lock(installed), encoding="utf-8"
    )
    monkeypatch.setattr(upd, "FRONTEND", fe)
    monkeypatch.setattr(upd, "LOCKFILE", fe / "package-lock.json")
    monkeypatch.setattr(upd, "INSTALLED_LOCK", fe / "node_modules" / ".package-lock.json")

    assert upd.frontend_out_of_sync() is None, (
        "an absent optional package was reported as drift"
    )


def test_a_real_version_mismatch_is_drift(upd, tmp_path, monkeypatch):
    """The positive control for the test above (CONVENTIONS rule 11).

    Without it, a `frontend_out_of_sync` that always returned None would pass
    the optional-package test and report every tree as healthy forever. This is
    the exact defect that was live: lucide 1.16.0 installed, 1.18.0 in the lock.
    """
    fe = tmp_path / "frontend"
    (fe / "node_modules").mkdir(parents=True)
    (fe / "package-lock.json").write_text(
        _lock({"node_modules/@lucide/angular": {"version": "1.18.0"}}), encoding="utf-8"
    )
    (fe / "node_modules" / ".package-lock.json").write_text(
        _lock({"node_modules/@lucide/angular": {"version": "1.16.0"}}), encoding="utf-8"
    )
    monkeypatch.setattr(upd, "FRONTEND", fe)
    monkeypatch.setattr(upd, "LOCKFILE", fe / "package-lock.json")
    monkeypatch.setattr(upd, "INSTALLED_LOCK", fe / "node_modules" / ".package-lock.json")

    reason = upd.frontend_out_of_sync()
    assert reason is not None, "a real version mismatch was not detected"
    assert "1.16.0" in reason and "1.18.0" in reason, (
        f"the message must name both versions so a user can act on it: {reason}"
    )


def test_a_missing_node_modules_is_drift(upd, tmp_path, monkeypatch):
    fe = tmp_path / "frontend"
    fe.mkdir(parents=True)
    (fe / "package-lock.json").write_text(_lock({}), encoding="utf-8")
    monkeypatch.setattr(upd, "FRONTEND", fe)
    monkeypatch.setattr(upd, "LOCKFILE", fe / "package-lock.json")
    monkeypatch.setattr(upd, "INSTALLED_LOCK", fe / "node_modules" / ".package-lock.json")
    assert upd.frontend_out_of_sync() is not None


def test_the_split_torch_stack_is_never_compared(upd):
    """requirements.txt pins the LOCAL torch trio; the container bakes another.

    Comparing them would report drift on every container and, worse, "fixing"
    it would clobber whichever build is correct for that machine.
    """
    pins = upd.declared_pins()
    for name in ("torch", "torchvision", "torchaudio", "triton", "triton-windows"):
        assert name not in pins, (
            f"{name} is in the comparison set; the updater would try to "
            "reinstall a split-stack package"
        )
    assert pins, "the comparison set is empty - nothing would ever be checked"
    assert "fastapi" in pins, "an ordinary pinned package should be compared"


# ── The launchers stay thin ────────────────────────────────────────────────

def test_every_launcher_exists_and_delegates(upd):
    """Three launchers, one implementation.

    They exist to find a Python and hand over. If one grows its own copy of the
    logic there are three implementations to keep in step, which is precisely
    the failure `test_dependency_metadata_contracts.py` guards against for the
    five install paths.
    """
    for name in LAUNCHERS:
        path = REPO / name
        assert path.is_file(), f"{name} is missing"
        text = path.read_text(encoding="utf-8")
        assert "update.py" in text, f"{name} does not hand over to update.py"


def test_no_launcher_reimplements_the_work(upd):
    """The drift guard, and it has a positive control below (CONVENTIONS 11)."""
    offenders = []
    for name in LAUNCHERS:
        text = (REPO / name).read_text(encoding="utf-8")
        body = "\n".join(
            line
            for line in text.splitlines()
            if not line.strip().startswith(("#", "REM", "::"))
        )
        for verb in ("npm ci", "npm install", "pip install", "git pull", "git fetch"):
            if verb in body:
                offenders.append(f"{name} runs `{verb}` itself")
    assert not offenders, (
        "a launcher is doing the work instead of delegating: "
        + "; ".join(offenders)
        + ". Put it in scripts/update.py, where there is one copy."
    )


def test_the_launcher_check_can_actually_fail(upd, tmp_path, monkeypatch):
    """Positive control for the guard above.

    It is a collect-offenders-then-assert-empty check, which cannot tell "no
    launcher reimplements the work" from "I failed to read the launchers".
    """
    fake = tmp_path
    (fake / "update.sh").write_text("#!/bin/sh\nnpm ci\n", encoding="utf-8")
    (fake / "update.ps1").write_text("# ok\n", encoding="utf-8")
    (fake / "update.bat").write_text("@echo off\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "REPO", fake)
    try:
        with pytest.raises(AssertionError, match="npm ci"):
            test_no_launcher_reimplements_the_work(upd)
    finally:
        monkeypatch.undo()


def test_the_shipped_updater_is_actually_tracked_by_git():
    """The launchers are useless if the driver never reaches a user.

    This nearly shipped: the driver was first written to `scripts/update.py`,
    and `/scripts/` is gitignored (`.gitignore:40`) as a local scratch area. It
    worked perfectly on the machine that wrote it and would have been absent
    from every clone -- three launchers pointing at a file nobody else has. The
    "works for whoever broke it" shape, in a file whose entire job is to stop
    that shape.
    """
    for name in (DRIVER.name, *LAUNCHERS):
        result = subprocess.run(
            ["git", "check-ignore", "-q", name],
            cwd=str(REPO),
            capture_output=True,
            timeout=120,
            check=False,
        )
        # exit 0 = the path IS ignored
        assert result.returncode != 0, (
            f"{name} is gitignored, so it will not exist in a fresh clone -- "
            "the updater cannot ship from a path git refuses to track"
        )

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", DRIVER.name],
        cwd=str(REPO),
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert tracked.returncode == 0, (
        f"{DRIVER.name} is not tracked by git yet. It is only committed work "
        "that reaches a user."
    )
