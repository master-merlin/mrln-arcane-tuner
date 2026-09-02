"""The CI CSP-coherence step installs a hand-picked package set — pin that it
covers the import chain it triggers.

``.github/workflows/gate.yml`` runs ``backend/tests/api/test_csp_policy.py``
inside the *frontend* job, because that is the only job holding the built
``index.html`` the policy applies to. It cannot afford the full backend
requirements there (torch and friends), so it installs a short list by hand.

That list is verified by nothing the local gate can show: the developer venv
has every package, so ``import app.api._security_headers`` always succeeds
here. On CI run 4 (2026-09-03) collection died on ``No module named
'structlog'`` — the test file imports ``app.api._security_headers``, which
runs ``backend/app/__init__.py``, which imports ``app.core.compat``, which
imports ``structlog`` at module level. Nobody had followed the chain.

This test follows the chain statically (module-level imports only, ``try``
blocks excluded because those are the optional ones), collects every
third-party top-level module it reaches, and asserts that the packages named
on the step's ``pip install`` line — plus their transitive dependencies as
the local venv records them — provide each one. A new module-level import
in that chain that the workflow does not install fails here, on Windows,
before it fails on the runner.

Limits, stated: the transitive closure is read from the local venv's
metadata, so a dependency CI would resolve differently (an environment
marker) is out of reach; and the walker sees only literal ``import``
statements, not ``importlib`` calls.
"""

from __future__ import annotations

import ast
import importlib.metadata as md
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).absolute().parents[2]
BACKEND = REPO_ROOT / "backend"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "gate.yml"
CSP_TEST = BACKEND / "tests" / "api" / "test_csp_policy.py"

# The test module itself is a root: whatever IT imports at module level must
# be installed too (pytest, and through `from app.api... import` the app).
ROOTS = (CSP_TEST,)

_STEP_NAME = "CSP coherence against the BUILT bundle"
_PIP_LINE = re.compile(r"python -m pip install\s+(?P<args>.+)$")
_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


# ── the workflow side ───────────────────────────────────────────────────────


def csp_step_packages(workflow_text: str = "") -> set[str]:
    """Package names on the CSP step's ``pip install`` line, flags dropped."""
    text = workflow_text or WORKFLOW.read_text(encoding="utf-8")
    start = text.find(_STEP_NAME)
    assert start >= 0, f"step {_STEP_NAME!r} not found in {WORKFLOW}"
    for line in text[start:].splitlines():
        m = _PIP_LINE.search(line)
        if m:
            return {
                a.strip("'\"") for a in m.group("args").split() if not a.startswith("-")
            }
    raise AssertionError(f"no `pip install` line under step {_STEP_NAME!r}")


# ── the import-chain side ───────────────────────────────────────────────────


def _module_path(dotted: str) -> Path | None:
    """``app.x.y`` → the file that import statement executes, if it is ours."""
    parts = dotted.split(".")
    if parts[0] != "app":
        return None
    base = BACKEND.joinpath(*parts)
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def _package_inits(dotted: str) -> list[Path]:
    """Importing ``app.a.b`` executes ``app/__init__``, ``app/a/__init__`` …"""
    parts = dotted.split(".")
    out: list[Path] = []
    for depth in range(1, len(parts)):
        init = BACKEND.joinpath(*parts[:depth]) / "__init__.py"
        if init.is_file():
            out.append(init)
    return out


def _top_level_imports(path: Path) -> list[tuple[str, list[str]]]:
    """Module-level import statements: ``(module, [names])``.

    Only statements directly in the module body count: an import inside a
    function is deferred, and one inside ``try`` is optional by construction.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[str, list[str]]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name, []))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.module, [a.name for a in node.names]))
    return found


def reachable_third_party(roots: tuple[Path, ...] = ROOTS) -> set[str]:
    """Top-level third-party module names reached from ``roots`` at import."""
    stdlib = sys.stdlib_module_names
    seen: set[Path] = set()
    third_party: set[str] = set()
    stack = list(roots)
    while stack:
        path = stack.pop()
        if path in seen:
            continue
        seen.add(path)
        for module, names in _top_level_imports(path):
            top = module.split(".")[0]
            if top == "__future__" or top in stdlib:
                continue
            if top != "app":
                third_party.add(top)
                continue
            stack.extend(_package_inits(module))
            target = _module_path(module)
            if target is not None:
                stack.append(target)
            # `from app.pkg import submodule` executes the submodule too.
            for name in names:
                sub = _module_path(f"{module}.{name}")
                if sub is not None:
                    stack.append(sub)
    return third_party


# ── what the named packages provide ─────────────────────────────────────────


def _dist_closure(names: set[str]) -> set[str]:
    """Distributions installed by ``pip install <names>``, per local metadata.

    Requirements guarded by an ``extra ==`` marker are excluded — pip does
    not install them without the extra, and neither does the workflow.
    """
    roots = {n.lower() for n in names}
    closure: set[str] = set()
    stack = list(roots)
    while stack:
        name = stack.pop()
        if name in closure:
            continue
        try:
            reqs = md.requires(name) or []
        except md.PackageNotFoundError:
            if name in roots:
                pytest.fail(
                    f"{name!r} is named on the workflow's pip line but is not "
                    f"installed in this venv, so its dependency closure cannot "
                    f"be judged here — install it, or fix the workflow line"
                )
            # A transitive requirement absent from this venv is one whose
            # marker excluded it here (`python_version < "3.11"` and the
            # like) — pip on the runner, same Python, skips it too.
            continue
        closure.add(name)
        for req in reqs:
            if "extra ==" in req:
                continue
            m = _REQ_NAME.match(req)
            if m:
                stack.append(m.group(1).lower())
    return closure


def provided_modules(names: set[str]) -> set[str]:
    """Top-level importable module names the closure of ``names`` installs."""
    closure = _dist_closure(names)
    provided: set[str] = set()
    for module, dists in md.packages_distributions().items():
        if any(d.lower() in closure for d in dists):
            provided.add(module)
    return provided


def missing_modules(pip_packages: set[str]) -> set[str]:
    """Reached third-party modules that ``pip install <pip_packages>`` misses."""
    return reachable_third_party() - provided_modules(pip_packages)


# ── the pins ─────────────────────────────────────────────────────────────────


def test_the_walker_reaches_through_the_app_package_init():
    """Positive control: the chain test → _security_headers → app/__init__
    → app.core.compat → structlog is what run 4 died on; the walker must
    see it, or the pin below is asserting over nothing."""
    reached = reachable_third_party()
    assert "structlog" in reached, reached
    assert "starlette" in reached, reached
    assert "pytest" in reached, reached


def test_a_pip_line_without_structlog_is_caught():
    """Negative control: the exact line that failed on CI run 4 must fail
    here, naming the module the runner could not find."""
    assert missing_modules({"fastapi", "starlette", "httpx", "pytest"}) == {"structlog"}


def test_the_csp_step_installs_every_module_its_import_chain_needs():
    packages = csp_step_packages()
    missing = missing_modules(packages)
    assert not missing, (
        f"the CSP-coherence step in {WORKFLOW.name} installs {sorted(packages)} "
        f"but importing the test module reaches {sorted(missing)} at module "
        f"level — CI collection will die on `No module named ...`; add the "
        f"package to that pip line (or make the import lazy)"
    )
