"""RULE-20 guard for LANE-40 — a legacy cleanup needs a way to be RUN.

**The defect this exists to stop from recurring.** `a5003618` changed a
derived-path layout, shipped `purge_legacy_layout()` to clean up what the old
layout left on disk, and wired it into exactly one caller: the dataset scan.
Every line of it was correct. It just never ran, because nothing makes a user
rescan a dataset they are happy with — so data already on disk stayed
un-migrated indefinitely and nothing in the product said so.

The transferable shape is *"the cleanup exists but only an operation the user
has no reason to perform can trigger it"*. The mechanical form of that is
cheap to check: a cleanup routine whose only callers live in the same layer it
was written for has no user-reachable path. So this asserts that every
``purge_legacy_*`` / ``migrate_legacy_*`` function in ``app/core`` is called
from ``app/api`` — the layer the user can actually reach.

This is an offender-collecting scan, which is the shape that can go green by
breaking rather than by being satisfied (CONVENTIONS "Tests" 11), so it
carries a positive control: a synthetic core module with an uncalled cleanup
must be caught by the same code that produces the real verdict.
"""

from __future__ import annotations

import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
CORE = BACKEND / "app" / "core"
API = BACKEND / "app" / "api"

#: A cleanup of data written by a superseded layout/format. The naming
#: convention IS the contract here — a routine that does this and is named
#: something else is invisible to this guard, which is why the convention is
#: stated in the docstring above rather than assumed.
_CLEANUP_DEF = re.compile(r"^def ((?:purge|migrate)_legacy_\w+)", re.MULTILINE)


def _cleanup_functions(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in root.rglob("*.py"):
        for name in _CLEANUP_DEF.findall(path.read_text(encoding="utf-8")):
            found[name] = path
    return found


def _unreachable(core_root: Path, api_root: Path) -> list[str]:
    """Return cleanup functions with no call site under *api_root*."""
    api_text = "\n".join(
        p.read_text(encoding="utf-8") for p in api_root.rglob("*.py")
        if "tests" not in p.parts
    )
    return sorted(
        name for name in _cleanup_functions(core_root)
        if not re.search(rf"\b{re.escape(name)}\s*\(", api_text)
    )


def test_every_legacy_cleanup_is_reachable_from_the_api_layer():
    """A cleanup only the scan can trigger is a cleanup that does not happen.

    Fix by giving it a user-triggerable route (see
    ``app/api/dataset/thumbnail_routes.py``), not by renaming the function out
    of this guard's sight.
    """
    offenders = _unreachable(CORE, API)
    assert offenders == [], (
        f"legacy cleanups with no user-reachable caller in app/api: {offenders}. "
        "Shipping one of these is how the thumbnail relayout left every "
        "un-rescanned dataset holding orphan renditions forever."
    )


def test_the_reachability_scan_actually_catches_an_unreachable_cleanup(tmp_path):
    """Positive control.

    Without this, a drifted regex or a moved package would make the assertion
    above collect zero offenders and pass permanently — the same object as
    'nothing is wrong', and indistinguishable from it.
    """
    fake_core = tmp_path / "core"
    fake_api = tmp_path / "api"
    fake_core.mkdir()
    fake_api.mkdir()
    (fake_core / "widgets.py").write_text(
        "def purge_legacy_widgets(p):\n    return 0\n"
        "def migrate_legacy_shapes(p):\n    return 0\n",
        encoding="utf-8",
    )
    (fake_api / "routes.py").write_text(
        "from x import migrate_legacy_shapes\n"
        "def r():\n    return migrate_legacy_shapes('p')\n",
        encoding="utf-8",
    )

    assert _unreachable(fake_core, fake_api) == ["purge_legacy_widgets"]


def test_the_real_thumbnail_purge_is_what_this_guard_is_watching():
    """Reachability precondition (CONVENTIONS 'Tests' 10): the guard must have
    a real subject, or it is a tick. If this name disappears, the guard above
    is scanning an empty set and needs a new subject, not a green tick."""
    assert "purge_legacy_layout" in _cleanup_functions(CORE)
