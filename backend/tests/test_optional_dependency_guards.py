"""An optional dependency that is BROKEN must not take the process down.

ARCHITECTURE D1: nothing imported at startup may raise, ever.

On 2026-09-03 the cu128 container died on its first and only launch:

    RuntimeError: cannot cache function '_make_tree': no locator available
    for /usr/local/lib/python3.12/dist-packages/pymatting/util/kdtree.py

raised while importing `rembg` as uid 10001. numba tried to write its JIT cache
next to the source file inside site-packages, could not, and had no fallback.
The traceback ran uvicorn -> load_app -> import_from_string, so this was the
import of `app.main` itself; the server never started and no restart was
involved.

`rembg.py` DID guard that import. It guarded it with ``except ImportError``,
and a RuntimeError is not an ImportError, so the exception went straight
through a try/except written to prevent exactly this outcome.

The lesson generalises past numba. To this app, a dependency that is ABSENT and
one that is PRESENT BUT CANNOT INITIALISE are the same event: the feature is
unavailable. A CUDA library that will not dlopen, a JIT that cannot write its
cache, a package whose own import has a side effect that fails -- none of them
raise ImportError, and all of them used to be fatal at startup. So these guards
are written about the OUTCOME, not about one exception class.

Two tests, deliberately different in kind. The first reproduces the real crash
through the real import machinery. The second is static and covers the modules
the first cannot reach without importing heavy optional packages, and stops the
narrow pattern being reintroduced somewhere new.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "backend" / "app"

REMBG_MODULE = "app.core.masking.models.rembg"


def _reimport_rembg_with(exc: BaseException):
    """Import the masking wrapper while `import rembg` raises `exc`.

    The failure is injected at the import hook rather than by deleting the
    module, because the incident was not a missing package -- rembg was
    installed and importable, and blew up partway through its own import.
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rembg" or name.startswith("rembg."):
            raise exc
        return real_import(name, *args, **kwargs)

    sys.modules.pop(REMBG_MODULE, None)
    builtins.__import__ = fake_import
    try:
        return importlib.import_module(REMBG_MODULE)
    finally:
        builtins.__import__ = real_import
        sys.modules.pop(REMBG_MODULE, None)
        importlib.import_module(REMBG_MODULE)  # restore the real one for others


def test_a_runtime_error_from_rembg_does_not_escape_the_import():
    """The 2026-09-03 crash, reproduced exactly and then required not to."""
    boom = RuntimeError(
        "cannot cache function '_make_tree': no locator available for "
        "/usr/local/lib/python3.12/dist-packages/pymatting/util/kdtree.py"
    )
    module = _reimport_rembg_with(boom)

    assert module.REMBG_AVAILABLE is False, (
        "importing rembg raised a RuntimeError and the module still reports "
        "itself available"
    )
    assert "RuntimeError" in module.REMBG_UNAVAILABLE_REASON
    assert "_make_tree" in module.REMBG_UNAVAILABLE_REASON, (
        "the reason must survive into the flag, or a container that cannot do "
        "masking gives the user no way to find out why"
    )


def test_a_plain_import_error_still_degrades_the_same_way():
    """The case that always worked must keep working -- widening the catch
    must not have changed the absent-package path."""
    module = _reimport_rembg_with(ImportError("No module named 'rembg'"))
    assert module.REMBG_AVAILABLE is False
    assert "ImportError" in module.REMBG_UNAVAILABLE_REASON


def test_the_reason_reaches_the_user_not_just_the_flag():
    """`load()` is what the API calls; its message is what a user sees."""
    module = _reimport_rembg_with(RuntimeError("numba could not write its cache"))
    with pytest.raises(ImportError, match="numba could not write its cache"):
        module.RemBGModel(service=None).load()


# ── static: the narrow pattern must not come back anywhere ──────────────────


def _module_level_import_guards(path: Path) -> list[int]:
    """Line numbers of module-level `try/except ImportError` blocks.

    Only module scope: a guard inside a function runs when the feature is used,
    which is a different (and acceptable) risk -- the process is already up.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    offenders: list[int] = []
    for node in tree.body:  # module scope ONLY, not ast.walk
        if not isinstance(node, ast.Try):
            continue
        if not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in node.body):
            continue
        for handler in node.handlers:
            names = []
            if isinstance(handler.type, ast.Name):
                names = [handler.type.id]
            elif isinstance(handler.type, ast.Tuple):
                names = [e.id for e in handler.type.elts if isinstance(e, ast.Name)]
            if names and set(names) <= {"ImportError", "ModuleNotFoundError"}:
                offenders.append(handler.lineno)
    return offenders


def test_no_startup_import_is_guarded_by_import_error_alone():
    files = sorted(APP_ROOT.rglob("*.py"))
    assert files, "walked zero application modules -- the glob is wrong"
    offenders = [
        f"{path.relative_to(REPO_ROOT).as_posix()}:{line}"
        for path in files
        for line in _module_level_import_guards(path)
    ]
    assert not offenders, (
        "these module-level import guards catch ImportError only: "
        f"{offenders}. They run during `import app.main`, so a dependency that "
        "is present but cannot initialise (numba unable to write its JIT cache, "
        "a CUDA library that will not dlopen) raises something else, escapes, "
        "and kills the process at startup instead of disabling one feature. "
        "That is ARCHITECTURE D1. Catch `Exception` and record the reason."
    )


def test_the_static_matcher_would_catch_the_original_line():
    """Vacuity check: an AST matcher that never fires proves nothing."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.py"
        bad.write_text(
            "try:\n    from rembg import remove\n    OK = True\nexcept ImportError:\n    OK = False\n",
            encoding="utf-8",
        )
        assert _module_level_import_guards(bad) == [4]

        good = Path(tmp) / "good.py"
        good.write_text(
            "try:\n    from rembg import remove\n    OK = True\nexcept Exception:\n    OK = False\n",
            encoding="utf-8",
        )
        assert _module_level_import_guards(good) == []

        # A guard INSIDE a function is deferred and must not be flagged --
        # otherwise the rule is unusable and gets deleted rather than obeyed.
        deferred = Path(tmp) / "deferred.py"
        deferred.write_text(
            "def probe():\n    try:\n        import torch\n    except ImportError:\n"
            "        return False\n    return True\n",
            encoding="utf-8",
        )
        assert _module_level_import_guards(deferred) == []
