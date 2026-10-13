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
import sys

import pytest

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


def _tree_route_modules() -> list[str]:
    """Route Modules as POSIX paths relative to API_DIR (file identity, not basename)."""
    return sorted(
        f.relative_to(API_DIR).as_posix()
        for f in API_DIR.rglob(ROUTE_PATTERN)
        if not ({"__pycache__", "tests"} & set(f.relative_to(API_DIR).parts))
    )


def _doc_route_domains(doc_path: pathlib.Path) -> list[tuple[str, list[str]]]:
    """Each `####` heading OCCURRENCE of the API Route Domains section -> (heading, rows as paths relative to api/).

    One entry per occurrence, in document order: a repeated heading is two groups, never merged.

    The directory a heading names in backticks (`api/dataset/`) prefixes its rows; a heading naming
    none is the top level. Repeated rows are kept (multiplicity is asserted); `websocket.py` is dropped.
    """
    text = doc_path.read_text(encoding="utf-8")
    start = text.index("### API Route Domains")
    rest = text[start + len("### API Route Domains"):]
    end = re.search(r"^#{1,3} ", rest, re.MULTILINE)
    section = rest[: end.start()] if end else rest
    domains: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    prefix = ""
    for line in section.splitlines():
        if line.startswith("#### "):
            heading = line[5:].strip()
            current = []
            domains.append((heading, current))
            named = re.search(r"`api/([\w/]+/)`", heading)
            prefix = named.group(1) if named else ""
        elif current is not None:
            row = re.match(r"\|\s*`([\w]+\.py)`", line)
            if row and row.group(1) not in NOT_A_MODULE:
                current.append(prefix + row.group(1))
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
    names = [heading for heading, _ in groups]
    repeated = sorted({n for n in names if names.count(n) > 1})
    assert not repeated, (
        f"docs/ARCHITECTURE.md 'API Route Domains' repeats the #### heading {repeated}; "
        f"each heading is one group ({len(groups)} groups, README claims {claimed})."
    )
    assert claimed == len(groups), (
        f"README diagram claims {claimed} Route Domains; docs/ARCHITECTURE.md 'API Route Domains' has "
        f"{len(groups)} #### groups ({names})."
    )


def test_diagram_route_module_count_not_grossly_stale() -> None:
    _assert_within_2x("Route Modules", "`*routes*.py` under backend/app/api")


def test_every_route_module_belongs_to_one_documented_domain() -> None:
    disk = _tree_route_modules()
    groups = _doc_route_domains(ARCHITECTURE_DOC)
    rows = [path for _, paths in groups for path in paths]
    problems: list[str] = []
    for path in disk:
        count = rows.count(path)
        if count == 0:
            problems.append(f"{path}: undocumented (on disk, no row in any domain table)")
        elif count > 1:
            problems.append(f"{path}: documented {count} times (exactly one row expected)")
    for path in sorted(set(rows)):
        if fnmatch.fnmatch(path.rsplit("/", 1)[-1], ROUTE_PATTERN) and path not in disk:
            problems.append(f"{path}: stale row (documented, not on disk)")
    header = f"route module membership drifted ({sorted(NOT_A_MODULE)} excluded by name):"
    assert not problems, header + "\n  " + "\n  ".join(sorted(problems))


def test_domain_count_rejects_a_repeated_heading(tmp_path, monkeypatch) -> None:
    text = ARCHITECTURE_DOC.read_text(encoding="utf-8")
    table_head = "| `caption_context_routes.py`"
    assert table_head in text
    split = "\n#### Other Route Modules\n\n| Module | Prefix | Purpose |\n| --- | --- | --- |\n" + table_head
    doc = tmp_path / "ARCHITECTURE.md"
    doc.write_text(text.replace("\n" + table_head, split, 1), encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "ARCHITECTURE_DOC", doc)
    with pytest.raises(AssertionError, match=r"repeats the #### heading \['Other Route Modules'\].*4 groups, README claims 3"):
        test_diagram_route_domain_count_matches_architecture_doc()


def test_membership_rejects_a_duplicated_row(tmp_path, monkeypatch) -> None:
    text = ARCHITECTURE_DOC.read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| `thumbnail_routes.py`"))
    doc = tmp_path / "ARCHITECTURE.md"
    doc.write_text(text.replace(row, row + "\n" + row, 1), encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "ARCHITECTURE_DOC", doc)
    with pytest.raises(AssertionError, match=r"dataset/thumbnail_routes\.py: documented 2 times"):
        test_every_route_module_belongs_to_one_documented_domain()


def test_membership_rejects_a_same_named_file_in_another_directory(tmp_path, monkeypatch) -> None:
    api = tmp_path / "api"
    for module in _tree_route_modules():
        (api / module).parent.mkdir(parents=True, exist_ok=True)
        (api / module).write_text("", encoding="utf-8")
    (api / "training" / "thumbnail_routes.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "API_DIR", api)
    with pytest.raises(AssertionError, match=r"training/thumbnail_routes\.py: undocumented"):
        test_every_route_module_belongs_to_one_documented_domain()
