"""Bridge to the harness-owned tests — the ONE tracked file that never needs to change.

The dev harness (`_harness/`, gitignored) is junction-linked into every worktree, so a
harness tool is global and always newest, while a test of it that lives in `tests/` is
tracked and forks with the branch. Those two halves of one contract cannot be kept in
step by git (a fresh fork off the main branch was born red for a lint change it had never
seen). Hence the tools' own tests live BESIDE the tools in `_harness/tests/` — same storage,
same junction, always the version that matches — and this module only *collects* them:
every `_harness/tests/test_*.py` is loaded, its test functions are re-exported here under a
`<module>__` prefix (that is their pytest id) and its fixtures under their own names. Public
CI has no `_harness/` and skips the lot.

The checkout a harness test is ABOUT is THIS file's checkout (`tests/` is real even in a
worktree), handed over as `HARNESS_CHECKOUT` (legacy name `MRLN_CHECKOUT` is set too); a
harness test must never resolve it through its own `__file__`, which walks the junction into
the main checkout.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

CHECKOUT = Path(__file__).resolve().parents[1]
SUITE = CHECKOUT / "_harness" / "tests"

pytestmark = pytest.mark.skipif(
    not SUITE.is_dir(), reason="_harness/tests not present (dev harness only)"
)


def _is_fixture(obj) -> bool:
    # pytest < 8.4 marks the function itself; >= 8.4 wraps it in a definition object
    return type(obj).__name__ == "FixtureFunctionDefinition" or hasattr(
        obj, "_pytestfixturefunction"
    )


def _collect() -> dict[str, object]:
    exported: dict[str, object] = {}
    if not SUITE.is_dir():
        return exported
    os.environ["HARNESS_CHECKOUT"] = str(CHECKOUT)
    os.environ["MRLN_CHECKOUT"] = str(CHECKOUT)
    for path in sorted(SUITE.glob("test_*.py")):
        name = f"harness_tests_{path.stem}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except pytest.skip.Exception:  # module-level skip (its tool is absent): that module only
            continue
        for attr, obj in vars(mod).items():
            if attr.startswith("test_") and callable(obj):
                exported[f"{path.stem}__{attr}"] = obj
            elif _is_fixture(obj):
                exported[attr] = obj  # fixtures are looked up by their own name
    return exported


globals().update(_collect())


def test_the_bridge_collected_something_when_the_harness_is_present():
    collected = [k for k in globals() if "__test_" in k]
    assert collected, "_harness/tests exists but nothing was collected — a module raised at import?"
