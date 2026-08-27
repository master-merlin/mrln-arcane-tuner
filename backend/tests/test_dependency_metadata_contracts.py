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
