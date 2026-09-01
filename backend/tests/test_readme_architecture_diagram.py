"""The README architecture diagram's counts, loosely pinned to the tree.

RELEASE AUDIT 0.7.9-beta..HEAD (2026-09-01) found the diagram claiming "43
Standalone Components" and "9 Route Domains" against a measured 153 component
files and 31 route modules — over 3x stale, unnoticed because nothing read
the tree against it. `_harness/DOCS_MANIFEST.json` already marks these counts
as DECAYS ("neither is currently pinned by a test"); this is that pin.

Not an exact match — a "150+" / "30+" hedge is intentionally imprecise
because these numbers grow between doc passes and README churn is not worth
it for a diagram, not the family table. What the test guards against is
*magnitude* drift: the claimed number falling below half the tree's measured
count, which is what "wrong by 3.5x" looks like from either direction.
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
README = REPO / "README.md"
FRONTEND_APP = REPO / "frontend" / "src" / "app"
API_DIR = REPO / "backend" / "app" / "api"


def _tree_component_count() -> int:
    return sum(
        1
        for f in FRONTEND_APP.rglob("*.ts")
        if "__tests__" not in f.parts and ".spec." not in f.name and "@Component" in f.read_text(encoding="utf-8", errors="ignore")
    )


def _tree_route_module_count() -> int:
    return sum(1 for f in API_DIR.rglob("*routes*.py") if "__pycache__" not in f.parts)


def _diagram_number(label: str) -> int:
    text = README.read_text(encoding="utf-8")
    match = re.search(rf"(\d+)\+?\s*{re.escape(label)}", text)
    assert match, f"README architecture diagram is missing a '{label}' count line"
    return int(match.group(1))


def test_diagram_component_count_not_grossly_stale() -> None:
    claimed = _diagram_number("Standalone Components")
    measured = _tree_component_count()
    assert claimed >= measured // 2, (
        f"README diagram claims {claimed} Standalone Components; the tree has "
        f"{measured} — more than 2x stale. Re-measure and update the diagram."
    )


def test_diagram_route_module_count_not_grossly_stale() -> None:
    claimed = _diagram_number("Route Modules")
    measured = _tree_route_module_count()
    assert claimed >= measured // 2, (
        f"README diagram claims {claimed} Route Modules; the tree has "
        f"{measured} — more than 2x stale. Re-measure and update the diagram."
    )
