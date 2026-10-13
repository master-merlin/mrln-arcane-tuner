"""The README architecture diagram's counts, pinned to the tree.

RELEASE AUDIT 0.7.9-beta..HEAD (2026-09-01) found the diagram claiming "43
Standalone Components" and "9 Route Domains" against a measured 153 component
files and 31 route modules -- over 3x stale, unnoticed because nothing read
the tree against it. `_harness/DOCS_MANIFEST.json` marked these counts as
DECAYS; this is the pin.

Definitions (the same in the README and docs/ARCHITECTURE.md):

* Standalone Component: a `*.ts` file under `frontend/src/app` (not
  `*.spec.ts`, not under `__tests__`) whose text contains `@Component`.
* Route Module: a `*routes*.py` file under `backend/app/api` at any depth
  (`__pycache__` and `tests` excluded). `websocket.py` is a row of the "Other
  Route Modules" table but is NOT a Route Module by this definition; the
  membership test excludes it by name.
* Route Domain: one `####` group under `### API Route Domains` in
  docs/ARCHITECTURE.md.

Tolerance: the Component and Module counts are checked for *magnitude*, not
churn, on both sides: `measured // 2 <= claimed <= measured * 2` (an "N+"
hedge reads as N). The Domain count is small, so it is exact. Membership is
exact: every Route Module on disk has a row in exactly one domain table, and
every table row naming a `*routes*.py` file exists on disk.
"""

from __future__ import annotations

import fnmatch
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
README = REPO / "README.md"
ARCHITECTURE_DOC = REPO / "docs" / "ARCHITECTURE.md"
FRONTEND_APP = REPO / "frontend" / "src" / "app"
API_DIR = REPO / "backend" / "app" / "api"

ROUTE_PATTERN = "*routes*.py"
# Documented as a row of the "Other" table, but not a *routes*.py module.
NOT_A_MODULE = {"websocket.py"}


def _tree_component_count() -> int:
    return sum(
        1
        for f in FRONTEND_APP.rglob("*.ts")
        if "__tests__" not in f.parts and ".spec." not in f.name and "@Component" in f.read_text(encoding="utf-8", errors="ignore")
    )


def _tree_route_modules() -> set[str]:
    return {f.name for f in API_DIR.rglob(ROUTE_PATTERN) if not ({"__pycache__", "tests"} & set(f.parts))}


def _doc_route_domains(doc_path: pathlib.Path) -> dict[str, set[str]]:
    """Each `####` heading of the API Route Domains section -> its backticked `*.py` file names."""
    text = doc_path.read_text(encoding="utf-8")
    start = text.index("### API Route Domains")
    rest = text[start + len("### API Route Domains"):]
    end = re.search(r"^#{1,3} ", rest, re.MULTILINE)
    section = rest[: end.start()] if end else rest
    domains: dict[str, set[str]] = {}
    current: set[str] | None = None
    for line in section.splitlines():
        if line.startswith("#### "):
            current = domains.setdefault(line[5:].strip(), set())
        elif current is not None:
            row = re.match(r"\|\s*`([\w]+\.py)`", line)
            if row and row.group(1) not in NOT_A_MODULE:
                current.add(row.group(1))
    return domains


def _diagram_number(label: str) -> int:
    text = README.read_text(encoding="utf-8")
    match = re.search(rf"(\d+)\+?\s*{re.escape(label)}", text)
    assert match, f"README architecture diagram is missing a '{label}' count line"
    return int(match.group(1))


def _assert_within_2x(label: str, definition: str) -> None:
    claimed = _diagram_number(label)
    measured = {"Standalone Components": _tree_component_count, "Route Modules": lambda: len(_tree_route_modules())}[label]()
    assert measured // 2 <= claimed <= measured * 2, (
        f"README diagram claims {claimed} {label}; the tree has {measured} ({definition}) "
        f"-- outside the 2x tolerance [{measured // 2}, {measured * 2}]. Re-measure and update the README diagram line."
    )


def test_diagram_component_count_not_grossly_stale() -> None:
    _assert_within_2x("Standalone Components", "`*.ts` under frontend/src/app with @Component, no specs")


def test_diagram_route_domain_count_matches_architecture_doc() -> None:
    claimed = _diagram_number("Route Domains")
    groups = _doc_route_domains(ARCHITECTURE_DOC)
    assert claimed == len(groups), (
        f"README diagram claims {claimed} Route Domains; docs/ARCHITECTURE.md 'API Route Domains' has "
        f"{len(groups)} #### groups ({sorted(groups)})."
    )


def test_diagram_route_module_count_not_grossly_stale() -> None:
    _assert_within_2x("Route Modules", "`*routes*.py` under backend/app/api")


def test_every_route_module_belongs_to_one_documented_domain() -> None:
    disk = _tree_route_modules()
    groups = _doc_route_domains(ARCHITECTURE_DOC)
    problems: list[str] = []
    for name in sorted(disk):
        owners = [g for g, names in groups.items() if name in names]
        if not owners:
            problems.append(f"{name}: undocumented (on disk, no row in any domain table)")
        elif len(owners) > 1:
            problems.append(f"{name}: in two domains {owners}")
    documented = set().union(*groups.values()) if groups else set()
    for name in sorted(documented):
        if fnmatch.fnmatch(name, ROUTE_PATTERN) and name not in disk:
            problems.append(f"{name}: stale row (documented, not on disk)")
    assert not problems, "route module membership drifted ({} excluded by name):\n  ".format(sorted(NOT_A_MODULE)) + "\n  ".join(problems)
