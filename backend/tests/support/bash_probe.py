"""Find a bash that can actually open the path it is about to be handed.

``shutil.which("bash")`` answers the wrong question on Windows. It returns
whichever bash wins the PATH race, and there are usually at least two:

    C:\\Program Files\\Git\\usr\\bin\\bash.exe     MSYS2 — resolves D:\\... natively
    C:\\WINDOWS\\system32\\bash.exe                WSL   — cannot see D:\\... at all

Both are real POSIX shells, so a skip-guard phrased as *"is there a bash?"* is
satisfied by either. But WSL's needs ``/mnt/d/...`` and mangles the separators
in an argv it is handed, so a suite guarded that way does not skip when it
cannot run — it **fails**, and which of the two you get depends on the
operator's PATH rather than on anything in the tree.

That is not hypothetical. `main` @ `52a2f197` measured **9 failed / 5686
passed** in one session and **0 failed / 5695 passed** in another, on the same
commit; the totals reconcile (5686 + 9 = 5695), so it was the same nine tests
flipping. Putting Git Bash first took it from 9 failed to 3 failed. A gate whose
green moves with the operator's PATH is not evidence, and a release must not
rest on one.

So the question this module asks is the one that matters: **can this shell open
this path?** It is settled by running the candidate, not by inspecting where it
lives — a probe survives a Git installed somewhere unusual (scoop, winget, a
portable copy) and self-heals if WSL ever learns drive paths, neither of which a
hardcoded allowlist can do.

Known locations are still consulted, but only *after* PATH and only as extra
candidates: these tests exist to exercise `entrypoint.sh`, not to honour the
operator's PATH, so a working shell the operator did not put on PATH is still a
better answer than skipping.

**The remaining three failures are not this.** ``TestEntrypointDropsPrivileges``
fails under *both* shells for a different reason — under WSL it cannot open the
script at all, under Git Bash it runs the entrypoint and dies on ``ln: failed to
create symbolic link '/app/backend/datasets'``. That is a filesystem/privilege
question, it is UNEXPLAINED rather than diagnosed, and it must not be folded in
here: a probe that appeared to fix it would only be hiding it.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

__all__ = ["bash_candidates", "bash_skip_reason", "find_bash"]

# Consulted after PATH. A machine can have a perfectly good Git Bash that no
# shell profile ever exported.
_KNOWN_GIT_BASH = (
    r"C:\Program Files\Git\usr\bin\bash.exe",
    r"C:\Program Files\Git\bin\bash.exe",
    r"C:\Program Files (x86)\Git\usr\bin\bash.exe",
    r"C:\Program Files (x86)\Git\bin\bash.exe",
)

# One `test -f` per candidate. Generous enough for a cold WSL start, short
# enough that a hung shell cannot stall the suite (invariant: every wait bounded).
_PROBE_TIMEOUT_S = 30


@lru_cache(maxsize=1)
def bash_candidates() -> tuple[str, ...]:
    """Every bash on this machine, PATH order first, then known installs.

    Deliberately not ``shutil.which``: that collapses the list to one entry,
    which is precisely how the losing shell becomes invisible.
    """
    found: list[str] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        if not path.is_file():
            return
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            found.append(str(path))

    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        for name in ("bash.exe", "bash"):
            add(Path(entry) / name)

    if os.name == "nt":
        for known in _KNOWN_GIT_BASH:
            add(Path(known))

    return tuple(found)


def _can_see(bash: str, target: Path) -> bool:
    """Does ``bash`` agree that ``target`` is a file?

    The path goes through as a positional argument rather than interpolated
    into the script, so a directory name containing a space or a backslash is
    the shell's problem to solve — which is exactly the capability being
    measured. (This repo lives under "MRLN Arcane Tuner"; a probe that quoted
    badly would reject every shell and look like a platform verdict.)
    """
    try:
        proc = subprocess.run(
            [bash, "-c", 'test -f "$1"', "_", str(target)],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def find_bash(must_see: Path | str) -> str | None:
    """The first bash that can open ``must_see``, or None if none can.

    Probe against the path the caller is actually going to hand over — a repo
    script for one suite, a tmp sandbox for another. They are on different
    volumes and a shell can manage one without the other.
    """
    target = Path(must_see)
    if not target.exists():
        # Every shell would fail this probe, and the resulting "no usable bash"
        # would be a lie about the platform. Fail loudly at the caller instead:
        # this is a bug in the test, not a property of the machine.
        raise FileNotFoundError(
            f"bash probe target does not exist: {target}. Probe against a path "
            "that is really there, or the answer describes the missing file "
            "rather than the shell."
        )
    for candidate in bash_candidates():
        if _can_see(candidate, target):
            return candidate
    return None


def bash_skip_reason(must_see: Path | str) -> str | None:
    """None when the suite can run; otherwise a reason that names the cause.

    The message is the point. Nine unexplained failures cost two sessions and a
    release baseline; a skip that says *which* shells were tried and *what* they
    could not open turns the same situation into one line of output.
    """
    target = Path(must_see)
    if not target.exists():
        return f"{target.name} is not present in this checkout"

    candidates = bash_candidates()
    if not candidates:
        return "no bash on PATH or in a known Git installation"

    if find_bash(target) is None:
        listed = ", ".join(candidates)
        return (
            f"found {len(candidates)} bash(es) but none can open {target} "
            f"— on Windows this is usually WSL's System32 bash, which needs "
            f"/mnt/<drive>/... paths. Tried: {listed}"
        )
    return None
