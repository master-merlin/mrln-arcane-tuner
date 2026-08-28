"""Packages whose DECLARED metadata is wrong, and the proof that skipping it is safe.

Three packages in `requirements.txt` are installed with ``--no-deps`` because
their declared requirements are wrong in ways that would otherwise abort the
resolve: `scenedetect` (declares GUI `opencv-python`, which would clobber the
pinned headless build), `sam3` (a stale `huggingface-hub<1.0` ceiling), and
`hpsv2` (its pytest dev dependencies leaked into its INSTALL requirements).

``--no-deps`` is a claim, and the claim is not "the package installs" — pip will
happily install anything with its dependencies switched off. The claim is **"the
dependencies we skipped were already satisfied by our own pins."** That is what
rots: a future release of one of these packages can add a REAL dependency, and
``--no-deps`` will skip it in silence. The install succeeds, the import may even
succeed, and the failure surfaces later inside a feature. So the load-bearing
test in this file is not the import — it is
``test_every_real_hpsv2_dependency_is_satisfied_by_our_own_pins``.

`sam3`'s import contract already lives in ``test_transformers5_compat.py``
(``test_sam3_imports_cleanly_despite_declared_hub_pin``) and is NOT restated
here — one producer per fact.
"""

from __future__ import annotations

import importlib.metadata as md
import pathlib
import re
import subprocess
import sys

import pytest

#: hpsv2 declares these as install requirements. They are its DEV dependencies,
#: leaked into `Requires-Dist` by a packaging error upstream. Skipping them is
#: the entire reason hpsv2 is installed with --no-deps.
HPSV2_LEAKED_TEST_DEPS = {"pytest", "pytest-split"}


def _hpsv2_requirements() -> list[str]:
    try:
        return md.metadata("hpsv2").get_all("Requires-Dist") or []
    except md.PackageNotFoundError:
        pytest.skip("hpsv2 is not installed in this environment")


def _dist_name(requirement: str) -> str:
    """The distribution name at the head of a PEP 508 requirement string."""
    head = requirement.split(";")[0].strip()
    for sep in ("==", ">=", "<=", "~=", "!=", ">", "<", "[", " ", "("):
        head = head.split(sep)[0]
    return head.strip().lower().replace("_", "-")


def test_hpsv2_declares_a_pytest_pin_we_do_not_honour():
    """The precondition, asserted so nothing below can pass vacuously.

    If a future hpsv2 drops the bogus pin, or if someone quietly reverts the
    runner to 7.2.0, every other test here would still pass while proving
    nothing. This one fails instead, and says which of the two happened.
    """
    declared = {_dist_name(r) for r in _hpsv2_requirements()}
    assert "pytest" in declared, (
        "hpsv2 no longer declares a pytest pin. The --no-deps treatment in "
        "install-deps.sh may now be unnecessary — re-check before removing it."
    )

    pinned = next(r for r in _hpsv2_requirements() if _dist_name(r) == "pytest")
    running = md.version("pytest")
    assert not pinned.replace(" ", "").endswith(f"=={running}"), (
        f"pytest {running} now MATCHES hpsv2's declared pin {pinned!r}. Either "
        "the runner was reverted to 7.2.0, or hpsv2 moved its pin. Until this "
        "is understood, nothing else in this file is evidence of anything."
    )


def test_hpsv2_registers_no_pytest_plugin_hook():
    """The structural reason the declared pin is inert rather than enforced.

    A distribution with a ``pytest11`` entry point is LOADED by pytest at
    startup, and a version mismatch there is a real incompatibility rather than
    a stale line in a metadata file. hpsv2 has no entry points at all, so pytest
    never touches hpsv2's code and hpsv2 never touches pytest's. If a future
    release adds a plugin hook, --no-deps stops being safe and this fires.
    """
    try:
        dist = md.distribution("hpsv2")
    except md.PackageNotFoundError:
        pytest.skip("hpsv2 is not installed in this environment")

    hooks = [ep for ep in dist.entry_points if ep.group == "pytest11"]
    assert not hooks, (
        f"hpsv2 now registers pytest plugin hooks ({hooks}). Its declared "
        "pytest pin is no longer inert metadata — pytest loads this code. "
        "Re-evaluate the --no-deps install in install-deps.sh."
    )


def test_every_real_hpsv2_dependency_is_satisfied_by_our_own_pins():
    """THE guard for --no-deps, and the one that will actually catch a change.

    Skipping a package's dependency resolution is only safe while every real
    dependency it has is independently present. A future hpsv2 that adds one
    would have it silently omitted at install time — the failure would surface
    much later, inside scoring, as a bare ImportError in front of a user.
    """
    missing = []
    for requirement in _hpsv2_requirements():
        name = _dist_name(requirement)
        if name in HPSV2_LEAKED_TEST_DEPS:
            continue
        try:
            md.version(name)
        except md.PackageNotFoundError:
            missing.append(requirement)

    assert not missing, (
        f"hpsv2 declares dependencies that nothing else installs: {missing}. "
        "It is installed with --no-deps, so pip skipped these silently. Add "
        "them to backend/requirements.txt, or drop the --no-deps treatment."
    )


def test_hpsv2_imports_under_a_runner_its_metadata_forbids():
    """The behavioural half: the package actually works, pin notwithstanding.

    ``hpsv2_model.py`` imports hpsv2 lazily inside its methods, so a broken
    install does not surface at startup — it surfaces when a user scores an
    image. Importing it here moves that discovery into the gate.

    In a SUBPROCESS, for two reasons. The honest one: importing hpsv2 in-process
    pulls `clint`, whose vendored colorama registers an atexit hook that writes
    to `sys.stdout` after pytest has closed its capture — a traceback printed
    below every future gate summary, exit code 0, pure noise in the one report
    that has to stay readable. The better one: a fresh interpreter with no
    pytest imported anywhere is a stronger proof that hpsv2 does not need the
    runner it pins than any import performed from inside pytest could be.
    """
    if not _hpsv2_requirements():  # skips if hpsv2 is absent
        pytest.skip("hpsv2 is not installed in this environment")

    probe = "import hpsv2, sys; sys.exit(0 if hasattr(hpsv2, 'score') else 3)"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        "hpsv2 does not import cleanly in a fresh interpreter "
        f"(exit {result.returncode}). It is installed with --no-deps, so the "
        "first thing to check is whether a dependency was silently skipped.\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )


def test_pytest_split_stays_uninstalled():
    """It was removed deliberately, and re-adding it breaks the runner SILENTLY.

    `pytest-split==0.8.0` — the version hpsv2's metadata names — caps at
    `pytest<8`, so installing it does not fail: pip resolves the conflict by
    DOWNGRADING pytest, which then breaks pytest-asyncio on import. It had no
    callers in this repo (no `--splits`, no `--group`, no durations file), so
    the fix was to drop it rather than to pin a newer one.
    """
    try:
        version = md.version("pytest-split")
    except md.PackageNotFoundError:
        return

    pytest.fail(
        f"pytest-split {version} is installed. It is not in requirements.txt "
        "and has no callers here; 0.8.0 caps at pytest<8 and silently "
        "downgrades the runner. If it is genuinely wanted, pin >=0.11.0 "
        "(which allows pytest<10) and give it a caller."
    )


# ── The five install paths must agree about WHICH packages get --no-deps ─────
#
# This is the drift guard, and it is here because the drift already happened
# twice. `sam3` was added to install-deps.sh, install.sh, install.ps1 and
# install.bat but NOT to the README's manual block, where it sat wrong until
# 2026-08-27. And `hpsv2==1.2.0` sat in requirements.txt with no inline comment
# while its two siblings each carried one, so the next person to read the file
# saw two annotated lines and one bare one.
#
# The failure mode is specific: adding a fourth --no-deps package means editing
# five files plus requirements.txt, and nothing fails if you edit four of them.
# The install still works on the maintainer's machine, because their venv
# already has everything. It breaks for whoever installs from the path you
# forgot.

REPO = pathlib.Path(__file__).resolve().parents[2]

#: Excluded from the bulk resolve for a DIFFERENT reason — the split stack
#: (see requirements.txt) installs these by hand per environment, so they are
#: not re-added with --no-deps and must not be checked as if they were.
TORCH_STACK = frozenset("torch torchvision torchaudio triton triton-windows".split())

#: name -> the file's filter, and how to read the package list out of it.
INSTALL_PATHS = {
    "backend/install-deps.sh": r"\(([a-z0-9|_-]+)\)",
    "backend/install.sh": r"\(([a-z0-9|_-]+)\)",
    "backend/install.ps1": r"\(([a-z0-9|_-]+)\)",
    "README.md": r"\(([a-z0-9|_-]+)\)",
}


def _alternation(path: str) -> set[str]:
    """Package names from the first grep/regex alternation in *path*."""
    text = (REPO / path).read_text(encoding="utf-8")
    for match in re.finditer(INSTALL_PATHS[path], text):
        names = set(match.group(1).split("|"))
        if "scenedetect" in names:  # the filter, not some other parenthesis
            return names
    return set()


def _bat_names() -> set[str]:
    """install.bat uses findstr, not a regex alternation: `^name== ^name==`."""
    text = (REPO / "backend" / "install.bat").read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if "findstr /V" in ln)
    return set(re.findall(r"\^([a-z0-9_-]+)==", line, re.I))


def _no_deps_packages() -> set[str]:
    """The canonical set: install-deps.sh's filter, minus the split stack."""
    if not (REPO / "backend" / "install-deps.sh").is_file():
        pytest.skip("install-deps.sh not in this checkout")
    return _alternation("backend/install-deps.sh") - TORCH_STACK


def test_every_install_path_excludes_the_same_packages():
    """Four regex-filtered paths plus install.bat must name the same set.

    A package excluded in install-deps.sh but not in install.sh is not a
    cosmetic difference: the bulk resolve there still sees the bad metadata and
    aborts, so that install path is simply broken while CI and Docker are green.
    """
    expected = _no_deps_packages() | (
        _alternation("backend/install-deps.sh") & TORCH_STACK
    )
    for path in INSTALL_PATHS:
        got = _alternation(path)
        missing = (expected & got) ^ expected
        # triton is excluded only by install-deps.sh (the container path);
        # compare on the --no-deps set, which every path must carry.
        assert _no_deps_packages() <= got, (
            f"{path} does not exclude {sorted(_no_deps_packages() - got)} from its "
            f"bulk install, but install-deps.sh does. That path's `pip install -r` "
            f"will hit the bad metadata and fail. (diff: {sorted(missing)})"
        )
    assert _no_deps_packages() <= _bat_names(), (
        f"backend/install.bat does not exclude "
        f"{sorted(_no_deps_packages() - _bat_names())}"
    )


def test_install_deps_reinstalls_each_excluded_package_with_no_deps():
    """Excluding a package without adding it back leaves it UNINSTALLED.

    Strictly worse than the resolve failure the exclusion avoids: the install
    succeeds, and the feature fails much later with a bare ImportError.

    Asserted against the shell's actual STRUCTURE - the variable each package is
    extracted into must be the one a ``--no-deps`` install consumes. The first
    draft of this test searched for the bare package name and passed vacuously,
    because that name is already present in the very filter line it was meant to
    be checking against. It also used a POSIX class inside a Python regex, which
    `re` reads as a nested set, so the pattern never matched what it looked like
    it matched - caught by running the file with -W error::FutureWarning.
    """
    text = (REPO / "backend" / "install-deps.sh").read_text(encoding="utf-8")
    for name in sorted(_no_deps_packages()):
        pattern = r"(\w+)=\"\$\(grep -iE '[^']*" + name + r"[^']*=="
        extract = re.search(pattern, text, re.I)
        assert extract, (
            f"install-deps.sh filters {name} out of the bulk install but has no "
            f"block extracting its pinned line - {name} would simply be missing "
            "from every install this script performs"
        )
        var = extract.group(1)
        assert re.search(r'--no-deps "\$' + var + r'"', text), (
            f"install-deps.sh extracts {name} into ${var} but never installs it: "
            f'expected `$PIP --no-deps "${var}"`'
        )


def test_each_no_deps_package_says_why_on_its_own_requirements_line():
    """The annotation is what stops the next person 'tidying' the exclusion.

    A bare pin gives a reader no signal that the package is special. This is
    exactly how the README lost sam3: the line looked ordinary.
    """
    req = (REPO / "backend" / "requirements.txt").read_text(encoding="utf-8")
    for name in sorted(_no_deps_packages()):
        line = next(
            (ln for ln in req.splitlines() if re.match(rf"^\s*{name}\s*==", ln, re.I)),
            None,
        )
        assert line is not None, f"{name} is filtered by install-deps.sh but not pinned"
        assert "--no-deps" in line, (
            f"requirements.txt pins {name} with no inline note that it is installed "
            "--no-deps. Its siblings carry one; an unexplained line invites removal."
        )


#: Constraints a ``--no-deps`` package DECLARES that our own pins deliberately
#: violate. The presence guard above proves the dependency is installed; it says
#: nothing about whether the version we installed is one the package would have
#: accepted. That gap is where a real incompatibility hides, so every violation
#: is listed here with its reason, and an unlisted one fails. A choice nobody
#: wrote down is indistinguishable from an accident.
ACCEPTED_CONSTRAINT_VIOLATIONS = {
    ("hpsv2", "protobuf"): (
        "hpsv2 declares protobuf<4 and imports no protobuf anywhere in its "
        "package, so the constraint is inert. We pin 5.29.6 because 3.20.3 "
        "carried two HIGH advisories, and CVE-2026-0994 is fixed only at "
        ">=5.29.6 -- 4.25.8 clears the other one and leaves that one open."
    ),
    ("hpsv2", "pytest"): (
        "The leaked dev pin. Covered behaviourally by "
        "test_hpsv2_imports_under_a_runner_its_metadata_forbids."
    ),
    ("sam3", "huggingface-hub"): (
        "A stale <1.0 ceiling. Covered behaviourally by "
        "test_sam3_imports_cleanly_despite_declared_hub_pin in "
        "test_transformers5_compat.py."
    ),
}


def _declared_violations() -> dict[tuple[str, str], str]:
    """Every (package, dependency) whose installed version breaks a declaration.

    Requirements gated behind an unsatisfied environment marker are not
    declarations about THIS install and are skipped. A dependency that is
    absent entirely is a presence problem, not a version one, and belongs to
    the guard above -- listing it here would report one defect as two.
    """
    from packaging.requirements import Requirement

    found: dict[tuple[str, str], str] = {}
    for package in sorted(_no_deps_packages()):
        try:
            declared = md.metadata(package).get_all("Requires-Dist") or []
        except md.PackageNotFoundError:
            continue
        for raw in declared:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            if not requirement.specifier:
                continue
            try:
                installed = md.version(requirement.name)
            except md.PackageNotFoundError:
                continue
            if not requirement.specifier.contains(installed, prereleases=True):
                key = (package, requirement.name.lower().replace("_", "-"))
                found[key] = f"declares {requirement.specifier}, installed {installed}"
    return found


def test_every_declared_constraint_we_break_is_one_we_chose_to_break():
    """The second half of the --no-deps claim, and the half that was missing.

    ``--no-deps`` means pip never checked these declarations, so nothing in the
    install fails when one stops holding. The guard above asks "is it there?";
    this one asks "is it a version the package said it could use?". Both
    questions have to be answered or the gap between them is invisible: when
    protobuf moved 3.20.3 -> 5.29.6 for two HIGH advisories, hpsv2's declared
    ``protobuf<4`` went from satisfied to violated and every test in this file
    stayed green, because protobuf was still *present*.
    """
    unexplained = {
        key: detail
        for key, detail in _declared_violations().items()
        if key not in ACCEPTED_CONSTRAINT_VIOLATIONS
    }
    assert not unexplained, (
        "A --no-deps package declares a constraint our pins break, and nobody "
        f"recorded why: {unexplained}. pip did not check it and will not. Either "
        "move the pin back inside the declared range, or add the pair to "
        "ACCEPTED_CONSTRAINT_VIOLATIONS with the reason it is safe."
    )


def test_no_accepted_violation_has_quietly_stopped_being_one():
    """Anti-vacuity: the ledger must not outlive the constraints it excuses.

    An entry that is no longer a violation is a licence nobody needs, and it
    keeps the next real one hidden -- the same failure shape as a test whose
    subject was removed and whose assertion still passes.
    """
    live = _declared_violations()
    stale = sorted(key for key in ACCEPTED_CONSTRAINT_VIOLATIONS if key not in live)
    assert not stale, (
        f"ACCEPTED_CONSTRAINT_VIOLATIONS excuses constraints that now hold: {stale}. "
        "The upstream pin was fixed or our own moved back into range -- drop the "
        "entry so the ledger keeps meaning what it says."
    )
