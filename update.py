#!/usr/bin/env python3
"""Bring a working copy of MRLN Arcane Tuner in line with `origin`.

Pull, then make the INSTALLED state match what the checkout now declares:
backend packages against `backend/requirements.txt`, frontend packages against
`frontend/package-lock.json`, and — only where this install actually serves a
built SPA — the built frontend against its sources.

WHY THIS EXISTS, AND WHY IT CONVERGES RATHER THAN DIFFS
-------------------------------------------------------
The obvious design is to compare the changed files across the pull ("did
requirements.txt move?") and act on the delta. That is what the server's own
`SelfUpdateService` does, correctly, because a container starts from a known
state every time.

A person's working copy does not. It drifts for reasons that have nothing to do
with the pull in front of you: someone merged a branch, you switched branches,
an install half-finished, you last updated three releases ago. A delta check
sees "nothing changed in this pull" and does nothing, while the tree stays
wrong. Both halves of this were live in this repo on 2026-08-28 — a venv still
on the previous test runner, and `node_modules` holding `@lucide/angular`
1.16.0 against a lockfile that said 1.18.0. Neither was caused by the pull that
exposed it.

So every check here asks **"does installed match declared?"**, never "did this
pull change it?". That makes the script idempotent, and it repairs drift it did
not cause.

IN A CONTAINER, THIS IS THE WRONG TOOL AND IT SAYS SO
------------------------------------------------------
A container already has an updater: `SelfUpdateService`, driven from the UI. It
does the three things this script deliberately refuses to do — `git reset
--hard origin/<branch>`, an unconditional frontend rebuild, and a restart once
in-process tasks have drained — because a container's checkout is disposable
and its lifecycle is the image's to manage. Running this script alongside it
would fight it: two updaters, one checkout, and a restart that lands mid-install.

So the container path is not "the same steps with different flags", it is a
different owner. This script detects the container and stops with a pointer,
rather than doing something subtly different in a place nobody will look.

WHAT IT WILL NOT DO
-------------------
- It will not touch a dirty working tree. Uncommitted work is yours; the script
  stops and says so rather than stashing, resetting or merging around it.
- It will not `git reset --hard`. The server's self-update does, because a
  container's checkout is disposable. Yours is not.
- It will not merge. `--ff-only`: if your branch has diverged from origin, that
  is a decision for you, not for a script.
- It will not install the torch/torchvision/torchaudio/triton stack. Those are a
  SPLIT STACK (see backend/requirements.txt): the container bakes one build, a
  local dev venv installs another, and the pins in requirements.txt are local
  documentation. Reinstalling them here would clobber whichever is correct for
  this machine.

Run it directly, or through `update.sh` / `update.ps1` / `update.bat`, which
exist only to find a Python and hand over to this file. The logic lives here
once: five install paths already have to agree about `--no-deps` packages in
this repo, and adding two more divergent copies of anything is how that becomes
six (see backend/tests/test_dependency_metadata_contracts.py).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Anchored on this file, never on the working directory: the script must work
# when invoked from anywhere, including a launcher double-clicked in Explorer.
REPO = Path(__file__).resolve().parent

#: What --check found to be out of date. The summary line is derived from
#: this rather than written independently: an earlier version reported
#: "Everything is up to date" directly under a list of drifted packages,
#: because check mode returned success and the summary read the wrong thing.
PENDING: list[str] = []

REQUIREMENTS = REPO / "backend" / "requirements.txt"
INSTALL_DEPS = REPO / "backend" / "install-deps.sh"
FRONTEND = REPO / "frontend"
LOCKFILE = FRONTEND / "package-lock.json"
INSTALLED_LOCK = FRONTEND / "node_modules" / ".package-lock.json"

#: Excluded from the backend comparison — see "WHAT IT WILL NOT DO" above.
#: Derived from install-deps.sh's own filter where possible so this cannot
#: drift away from the canonical installer.
TORCH_STACK_FALLBACK = frozenset(
    ["torch", "torchvision", "torchaudio", "triton", "triton-windows"]
)

GREEN, YELLOW, RED, DIM, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"
if os.name == "nt" and not os.environ.get("WT_SESSION"):
    # Old consoles render the escapes literally, which is worse than no colour.
    GREEN = YELLOW = RED = DIM = RESET = ""


def say(msg: str, colour: str = "") -> None:
    print(f"{colour}{msg}{RESET}", flush=True)


def step(msg: str) -> None:
    say(f"\n== {msg}", DIM)


def run(cmd: list[str], cwd: Path | None = None, capture: bool = True):
    """Run *cmd*; return CompletedProcess. Never raises on non-zero."""
    return subprocess.run(
        cmd,
        cwd=str(cwd or REPO),
        capture_output=capture,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def git(*args: str) -> subprocess.CompletedProcess:
    return run(["git", *args])


# ── bash, resolved deliberately (LANE-16) ───────────────────────────────────

def find_bash() -> str | None:
    """A bash that can open Windows drive paths.

    `shutil.which("bash")` on Windows frequently resolves to WSL's
    `System32\\bash.exe`, which cannot see `D:\\...` the way Git Bash does — so
    whichever bash happens to win the PATH race silently changes the result.
    That exact race made a test suite's pass/fail depend on the operator's PATH
    in this repo. Prefer Git's bash explicitly; fall back to PATH only after
    excluding the System32 one.
    """
    if os.name != "nt":
        return shutil.which("bash")

    for candidate in (
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\usr\bin\bash.exe",
    ):
        if Path(candidate).is_file():
            return candidate

    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        return found
    return None


# ── container detection ─────────────────────────────────────────────────────

def is_container() -> bool:
    """Mirrors ``app.core.container_config.is_container``.

    Duplicated on purpose rather than imported: this script must run before the
    venv is usable and must not depend on the backend package importing
    cleanly. Kept honest by ``test_update_script_matches_container_contract``.

    ``== "1"`` exactly, matching that function — no generic truthy parsing, so
    ``MRLN_CONTAINER=false`` is not read as a container.
    """
    return os.environ.get("MRLN_CONTAINER") == "1"


def refuse_in_container() -> None:
    say("This checkout is running inside the MRLN container.", YELLOW)
    say(
        "\nUpdates in the container are owned by the server's own updater "
        "(Settings -> Server -> Update), which resets the checkout to origin, "
        "rebuilds the frontend and restarts once running tasks have drained. "
        "This script deliberately does none of those things, and running both "
        "would have two updaters writing to one checkout.",
    )
    say("\nUse the in-app update instead. Nothing has been changed.", DIM)


# ── preflight ───────────────────────────────────────────────────────────────

def preflight() -> str | None:
    """Return the current branch, or None if it is not safe to proceed."""
    if not (REPO / ".git").exists():
        say(f"Not a git checkout: {REPO}", RED)
        return None

    dirty = git("status", "--porcelain").stdout.strip()
    if dirty:
        say("Your working tree has uncommitted changes:", RED)
        for line in dirty.splitlines()[:10]:
            say(f"    {line}")
        if len(dirty.splitlines()) > 10:
            say(f"    ... and {len(dirty.splitlines()) - 10} more")
        say(
            "\nStopping. Commit or stash them first - this script will not "
            "stash, reset or merge around your work.",
            RED,
        )
        return None

    branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if not branch or branch == "HEAD":
        say("You are on a detached HEAD; check out a branch first.", RED)
        return None
    return branch


# ── git ─────────────────────────────────────────────────────────────────────

def pull(branch: str) -> bool:
    step(f"Updating {branch} from origin")
    before = git("rev-parse", "HEAD").stdout.strip()

    fetched = git("fetch", "origin", branch)
    if fetched.returncode != 0:
        say(f"git fetch failed:\n{fetched.stderr.strip()}", RED)
        return False

    result = git("merge", "--ff-only", f"origin/{branch}")
    if result.returncode != 0:
        say(
            f"Cannot fast-forward {branch} onto origin/{branch}.\n"
            "Your branch has commits origin does not, or has diverged. That is "
            "a decision for you, not for this script - rebase or merge by hand.",
            RED,
        )
        return False

    after = git("rev-parse", "HEAD").stdout.strip()
    if before == after:
        say("Already up to date.", GREEN)
    else:
        count = git("rev-list", "--count", f"{before}..{after}").stdout.strip()
        say(f"Updated {before[:8]} -> {after[:8]} ({count} commits).", GREEN)
    return True


# ── backend ─────────────────────────────────────────────────────────────────

def torch_stack() -> frozenset[str]:
    """The split-stack names, read from install-deps.sh's own filter."""
    try:
        text = INSTALL_DEPS.read_text(encoding="utf-8")
        for match in re.finditer(r"\(([a-z0-9|_-]+)\)", text):
            names = set(match.group(1).split("|"))
            if "torch" in names:
                return frozenset(names & set(TORCH_STACK_FALLBACK))
    except OSError:
        pass
    return TORCH_STACK_FALLBACK


def declared_pins() -> dict[str, str]:
    """`name -> version` from requirements.txt, minus the split stack."""
    skip = torch_stack()
    pins: dict[str, str] = {}
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        match = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*([^\s;]+)", line)
        if not match:
            continue  # a marker-only or unpinned line; nothing to compare
        name = match.group(1).lower().replace("_", "-")
        if name in skip:
            continue
        pins[name] = match.group(2)
    return pins


def venv_python() -> Path | None:
    for rel in ("Scripts/python.exe", "bin/python", "bin/python3"):
        candidate = REPO / "backend" / "venv" / rel
        if candidate.is_file():
            return candidate
    return None


def installed_versions(python: Path) -> dict[str, str] | None:
    """`name -> version` for everything installed in the venv."""
    probe = (
        "import json,importlib.metadata as m;"
        "print(json.dumps({d.metadata['Name'].lower().replace('_','-'):d.version "
        "for d in m.distributions() if d.metadata['Name']}))"
    )
    result = run([str(python), "-c", probe])
    if result.returncode != 0:
        say(f"Could not read the venv's installed packages:\n{result.stderr}", RED)
        return None
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        say("Could not parse the venv's package list.", RED)
        return None


def backend_drift(python: Path) -> list[str] | None:
    """Human-readable drift lines, [] if in sync, None if undeterminable."""
    installed = installed_versions(python)
    if installed is None:
        return None
    drift = []
    for name, want in sorted(declared_pins().items()):
        have = installed.get(name)
        if have is None:
            drift.append(f"{name}: missing, requirements say {want}")
        elif have != want:
            drift.append(f"{name}: installed {have}, requirements say {want}")
    return drift


def update_backend(check_only: bool) -> bool:
    step("Backend packages")
    python = venv_python()
    if python is None:
        say(
            "No virtualenv at backend/venv - this script updates an existing "
            "install. Run backend/install.ps1 (Windows) or backend/install.sh "
            "first.",
            RED,
        )
        return False

    drift = backend_drift(python)
    if drift is None:
        return False
    if not drift:
        say("In sync with requirements.txt.", GREEN)
        return True

    say(f"{len(drift)} package(s) differ from requirements.txt:", YELLOW)
    for line in drift[:15]:
        say(f"    {line}")
    if len(drift) > 15:
        say(f"    ... and {len(drift) - 15} more")

    if check_only:
        PENDING.append("backend packages")
        say("--check: not installing.", DIM)
        return True

    bash = find_bash()
    if bash is None:
        say(
            "install-deps.sh is the canonical installer and needs bash, which "
            "was not found. Git for Windows ships one; otherwise run "
            "backend\\install.ps1 by hand.",
            RED,
        )
        return False

    say(f"Running install-deps.sh ({bash})...", DIM)
    result = run(
        [bash, str(INSTALL_DEPS), str(REQUIREMENTS)],
        cwd=REPO / "backend",
        capture=False,
    )
    if result.returncode != 0:
        say("install-deps.sh failed - see the output above.", RED)
        return False

    remaining = backend_drift(python)
    if remaining:
        # Converging is the whole contract; if it did not converge, say so
        # rather than reporting the install's exit code as success.
        say(f"Still {len(remaining)} package(s) out of sync after installing:", RED)
        for line in remaining[:10]:
            say(f"    {line}")
        return False
    say("Backend packages updated.", GREEN)
    return True


# ── frontend ────────────────────────────────────────────────────────────────

def frontend_out_of_sync() -> str | None:
    """Why node_modules disagrees with the lockfile, or None if it agrees.

    npm maintains `node_modules/.package-lock.json` as the record of what it
    actually installed, so comparing it with `package-lock.json` is the direct
    "is this tree in sync" question — as opposed to inferring it from whether a
    pull happened to touch the lockfile.
    """
    if not LOCKFILE.is_file():
        return None
    if not (FRONTEND / "node_modules").is_dir():
        return "node_modules is missing"
    if not INSTALLED_LOCK.is_file():
        return "node_modules has no install record (.package-lock.json)"
    try:
        want = json.loads(LOCKFILE.read_text(encoding="utf-8")).get("packages", {})
        have = json.loads(INSTALLED_LOCK.read_text(encoding="utf-8")).get("packages", {})
    except (OSError, json.JSONDecodeError) as exc:
        return f"could not compare lockfiles ({exc})"

    differing = []
    for name, spec in want.items():
        if not name or not isinstance(spec, dict) or not spec.get("version"):
            continue
        installed = (have.get(name) or {}).get("version")
        if installed is None:
            # A lockfile lists every platform's binaries. `@esbuild/aix-ppc64`
            # is `optional: true` with `os: [aix]`, and npm is CORRECT to skip
            # it here — 165 such entries made the first version of this check
            # call a healthy tree drifted. Absence is only drift when the
            # package was not optional to begin with.
            if spec.get("optional"):
                continue
            differing.append(f"{name}: not installed, lockfile says {spec['version']}")
        elif installed != spec["version"]:
            differing.append(
                f"{name}: installed {installed}, lockfile says {spec['version']}"
            )

    if differing:
        head = "; ".join(differing[:3])
        more = f" (+{len(differing) - 3} more)" if len(differing) > 3 else ""
        return f"{len(differing)} package(s) differ — {head}{more}"
    return None


def update_frontend(check_only: bool) -> bool:
    step("Frontend packages")
    if not LOCKFILE.is_file():
        say("No frontend/package-lock.json - nothing to do.", DIM)
        return True

    reason = frontend_out_of_sync()
    if reason is None:
        say("node_modules matches package-lock.json.", GREEN)
        return True

    say(f"node_modules is out of sync: {reason}", YELLOW)
    if check_only:
        PENDING.append("frontend packages")
        say("--check: not installing.", DIM)
        return True

    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if npm is None:
        say("npm is not on PATH - install Node.js, then re-run.", RED)
        return False

    say("Running npm ci...", DIM)
    # `npm ci`, never `npm install`: install resolves past the lockfile and
    # rewrites it, which hides exactly the drift this script exists to fix.
    result = run([npm, "ci"], cwd=FRONTEND, capture=False)
    if result.returncode != 0:
        say("npm ci failed - see the output above.", RED)
        return False

    remaining = frontend_out_of_sync()
    if remaining is not None:
        say(f"Still out of sync after npm ci: {remaining}", RED)
        return False
    say("Frontend packages updated.", GREEN)
    return True


# ── built SPA ───────────────────────────────────────────────────────────────

BUILT_SPA_DIRS = ("dist/frontend/browser", "browser")
BUILD_INPUTS = ("src", "angular.json", "package-lock.json")


def built_spa_dir() -> Path | None:
    for rel in BUILT_SPA_DIRS:
        candidate = FRONTEND / rel
        if candidate.is_dir() and (candidate / "index.html").is_file():
            return candidate
    return None


def spa_is_stale(built: Path) -> bool:
    """Is any build input newer than the built index.html?"""
    built_at = (built / "index.html").stat().st_mtime
    for rel in BUILD_INPUTS:
        path = FRONTEND / rel
        if not path.exists():
            continue
        newest = (
            max((p.stat().st_mtime for p in path.rglob("*") if p.is_file()), default=0)
            if path.is_dir()
            else path.stat().st_mtime
        )
        if newest > built_at:
            return True
    return False


def update_spa(mode: str, check_only: bool) -> bool:
    """mode: 'auto' | 'always' | 'never'.

    'auto' rebuilds only when this checkout ALREADY serves a built SPA. A
    developer running `ng serve` has no build to go stale and should not pay
    for a production build on every update; an install that serves
    `frontend/browser` shows stale code until it is rebuilt.
    """
    step("Built frontend")
    if mode == "never":
        say("Skipped (--no-build).", DIM)
        return True

    built = built_spa_dir()
    if built is None and mode == "auto":
        say(
            "This checkout has no built SPA, so nothing serves one - skipping. "
            "(Use --build to force a production build.)",
            DIM,
        )
        return True

    if built is not None and mode == "auto" and not spa_is_stale(built):
        say("Built SPA is newer than its sources.", GREEN)
        return True

    if check_only:
        PENDING.append("frontend build")
        say("--check: would run a production build.", YELLOW)
        return True

    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if npm is None:
        say("npm is not on PATH - cannot build.", RED)
        return False

    say("Running a production build (this takes a few minutes)...", DIM)
    result = run(
        [npm, "run", "build", "--", "--configuration", "production"],
        cwd=FRONTEND,
        capture=False,
    )
    if result.returncode != 0:
        say("The frontend build failed - see the output above.", RED)
        return False
    say("Frontend rebuilt.", GREEN)
    return True


# ── entry point ─────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="update",
        description="Update this MRLN Arcane Tuner checkout and its installed "
        "dependencies to match what the checkout declares.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what is out of date and change nothing",
    )
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help="skip git; only bring installed dependencies in line with the "
        "checkout you already have",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--build",
        action="store_true",
        help="always run a production frontend build",
    )
    group.add_argument(
        "--no-build",
        action="store_true",
        help="never run a production frontend build",
    )
    args = parser.parse_args(argv)

    if is_container():
        refuse_in_container()
        return 2

    say(f"MRLN Arcane Tuner - updating {REPO}", DIM)

    if not args.no_pull:
        branch = preflight()
        if branch is None:
            return 1
        if not pull(branch):
            return 1
    else:
        say("\nSkipping git (--no-pull).", DIM)

    ok = update_backend(args.check)
    ok = update_frontend(args.check) and ok
    mode = "always" if args.build else "never" if args.no_build else "auto"
    ok = update_spa(mode, args.check) and ok

    print()
    if not ok:
        say("Update did not finish cleanly - see the messages above.", RED)
        return 1
    if args.check:
        if PENDING:
            say(f"Out of date: {', '.join(PENDING)}.", YELLOW)
            say("Re-run without --check to bring this checkout in line.", DIM)
            return 1
        say("Everything is up to date.", GREEN)
        return 0
    say("Update complete.", GREEN)
    say("Restart the backend for the changes to take effect.", DIM)
    return 0


if __name__ == "__main__":
    sys.exit(main())
