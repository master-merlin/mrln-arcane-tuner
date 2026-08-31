"""A pin file is only proven by resolving it (LESSONS 2026-08-14).

``pip install -r requirements.txt`` failed with ResolutionImpossible for days
-- ``hf-xet 1.2.0`` against ``huggingface-hub 1.27.0``, plus a ``sam3`` ``==``
pin -- while the whole test suite stayed green, because the suite runs inside
an environment that was **already installed**. Nothing in the local gate reads
``requirements.txt`` as a resolver would. The Docker build and the in-app
self-update both take exactly that path.

Two halves, because the resolver needs a network and the local gate must not:

* the ALWAYS-ON half asserts that a machine in the gate does run the resolve --
  ``.github/workflows/gate.yml`` installs the dependencies through the
  canonical installer, in the job that must go green, with nothing suppressing
  its exit status. That is a positive assertion about file contents, so it has
  no offender-scan vacuity to control for (CONVENTIONS rule 11) -- deleting
  the step makes it fail loudly.
* the OPT-IN half actually runs ``pip install --dry-run`` against the file,
  filtered exactly as ``install-deps.sh`` filters it, with the filter READ OUT
  of that script rather than restated here so the two cannot drift. It is
  skipped without ``MRLN_PIP_RESOLVE_CHECK=1`` because it needs PyPI and takes
  tens of seconds; a skip is counted and visible, a silent no-op is not.

Why not simply add the dry-run to the local gate: ``pytest backend`` is run on
every commit and must stay offline and deterministic. CI is the machine whose
job is the network. What was missing was never the command -- it was anything
at all asserting that some machine runs it.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "backend"
REQUIREMENTS = BACKEND / "requirements.txt"
INSTALL_DEPS = BACKEND / "install-deps.sh"
GATE_WORKFLOW = REPO / ".github" / "workflows" / "gate.yml"

#: The one grep in install-deps.sh that decides what the bulk resolve sees.
#: Read, never restated: a package added to (or dropped from) that filter must
#: change what this test resolves, or the test stops being about the installer.
_FILTER_RE = re.compile(r"grep -ivE '\^\[\[:space:\]\]\*\(([a-z0-9|_-]+)\)")


def _installer_filtered_packages() -> set[str]:
    text = INSTALL_DEPS.read_text(encoding="utf-8")
    match = _FILTER_RE.search(text)
    assert match, (
        f"could not find install-deps.sh's bulk-install filter in {INSTALL_DEPS}. "
        "The installer changed shape and this module no longer filters "
        "requirements.txt the way the installers do -- fix the regex rather "
        "than dropping the check."
    )
    return set(match.group(1).split("|"))


def _filtered_requirements_text() -> str:
    """``requirements.txt`` minus the lines install-deps.sh excludes.

    The torch stack is a SPLIT STACK: the Docker image bakes its own build and
    the local venv installs another, so those pins are deliberately local-only
    and a resolver must never see them. scenedetect / sam3 / hpsv2 are
    re-installed ``--no-deps`` for their broken metadata.
    """
    excluded = _installer_filtered_packages()
    name_re = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
    kept = []
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        match = name_re.match(line)
        if match and match.group(1).lower().replace("_", "-") in excluded:
            continue
        kept.append(line)
    return "\n".join(kept) + "\n"


def test_the_installer_filter_is_still_readable_and_non_empty():
    """The two tests below are only about requirements.txt if this filter is
    the installer's real one. A regex that silently matches nothing would make
    the resolve test resolve the WRONG file (torch stack included) and the
    workflow test assert about a name that no longer exists."""
    packages = _installer_filtered_packages()
    assert "torch" in packages and "hpsv2" in packages, (
        f"install-deps.sh's filter parsed as {sorted(packages)}, which does not "
        "look like the split-stack + broken-metadata exclusion list"
    )

    filtered = _filtered_requirements_text()
    assert "\ntorch==" not in "\n" + filtered, (
        "the torch stack survived the filter; a resolver would try to install "
        "the local-only CUDA pins from PyPI"
    )
    assert len(filtered.splitlines()) > 20, "the filter ate requirements.txt"


def _gate_backend_steps() -> list[dict]:
    workflow = yaml.safe_load(GATE_WORKFLOW.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    assert "backend" in jobs, (
        f"{GATE_WORKFLOW} has no `backend` job any more; the gate this test "
        f"asserts about was renamed or removed (jobs: {sorted(jobs)})"
    )
    return jobs["backend"]["steps"]


@pytest.mark.skipif(not GATE_WORKFLOW.is_file(), reason="no .github/ in this checkout")
def test_the_ci_gate_resolves_requirements_through_the_canonical_installer():
    """Some machine must run the resolver, and its exit status must count.

    This is the guard the 2026-08-14 entry owed. It is deliberately pinned to
    the CANONICAL installer rather than to a literal ``pip install`` string:
    ``install-deps.sh`` is shared by the Docker build and the runtime
    self-update precisely so those paths cannot diverge, and CI joining them
    is what makes a green CI run evidence about them.
    """
    steps = _gate_backend_steps()
    resolving = [
        step
        for step in steps
        if "install-deps.sh" in str(step.get("run", ""))
    ]
    assert resolving, (
        "the gate's backend job no longer installs backend dependencies from "
        "requirements.txt through backend/install-deps.sh. Nothing anywhere "
        "then resolves the pin file, and a ResolutionImpossible reaches "
        "contributors, the Docker build and the in-app self-update with a "
        "green suite behind it (LESSONS 2026-08-14)."
    )
    for step in resolving:
        assert not step.get("continue-on-error"), (
            f"the install step {step.get('name')!r} is continue-on-error: an "
            "unresolvable requirements.txt would be reported and ignored"
        )

    names = [str(step.get("run", "")) for step in steps]
    install_at = min(i for i, run in enumerate(names) if "install-deps.sh" in run)
    pytest_at = next(
        (i for i, run in enumerate(names) if "pytest" in run),
        None,
    )
    assert pytest_at is not None and install_at < pytest_at, (
        "the dependency install must precede the pytest step, or the suite is "
        "running against something other than what requirements.txt resolves to"
    )


@pytest.mark.skipif(
    os.environ.get("MRLN_PIP_RESOLVE_CHECK") != "1",
    reason=(
        "needs PyPI and ~1 min; run with MRLN_PIP_RESOLVE_CHECK=1, or rely on "
        "the CI gate step pinned by "
        "test_the_ci_gate_resolves_requirements_through_the_canonical_installer"
    ),
)
def test_pip_can_resolve_the_filtered_requirements_file():
    """The resolver itself, on the file, the way a stranger's clone runs it.

    ``--ignore-installed`` is load-bearing: without it pip treats this venv's
    already-satisfied pins as resolved and short-circuits the very conflict
    this exists to find. ``--dry-run`` means nothing is written to the venv.
    """
    with tempfile.TemporaryDirectory() as tmp:
        req = Path(tmp) / "requirements.filtered.txt"
        req.write_text(_filtered_requirements_text(), encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--dry-run",
                "--ignore-installed",
                "--no-input",
                "--disable-pip-version-check",
                "-r",
                str(req),
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )

    assert result.returncode == 0, (
        "pip cannot resolve backend/requirements.txt as the installers feed it. "
        "The Docker build and the in-app self-update both take this path.\n\n"
        f"--- stdout (tail) ---\n{result.stdout[-3000:]}\n"
        f"--- stderr (tail) ---\n{result.stderr[-3000:]}"
    )
