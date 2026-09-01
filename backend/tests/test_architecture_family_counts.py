"""ARCHITECTURE.md's family surface, pinned to the tree it describes.

That section said "13 families + wan_shared" while the tree held 28 across 50
definitions. Fifteen families were added, each by someone who had no reason to
scroll to a prose paragraph in a document at another path, and the number was
true when typed and then stopped being true fifteen separate times. Nothing ever
compared the two, so nothing ever said so.

The load-bearing assertion here is NOT the count. A count is one integer and a
future drift can satisfy it by accident -- add one family, delete another, the
number holds and the table is wrong in two places. The load-bearing assertion is
``test_the_table_lists_exactly_the_families_that_exist``: set equality between
the tree and the table's first column, which fails on an addition, a removal,
a rename and a typo, and names which.

Deliberately NOT asserted: the prose in each row. A row's model description is
editorial and asserting it would make this guard fire on every wording change
until someone switched it off -- and took the set check with it.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
ARCHITECTURE = REPO / "docs" / "ARCHITECTURE.md"  # moved from documentation/ 2026-09-01, LANE-62; the old path is gone
FAMILIES_DIR = REPO / "backend" / "app" / "engine" / "models" / "families"


def _tree_families() -> set[str]:
    """A family is a package that ships definitions. The rest are support code."""
    return {
        d.name
        for d in FAMILIES_DIR.iterdir()
        if d.is_dir() and d.name != "__pycache__" and any((d / "definitions").glob("*.yaml"))
    }


def _tree_support_packages() -> set[str]:
    """A support package ships no definitions but IS a package: it has __init__.py.

    That clause is not pedantry. The first version of this test counted any
    directory without definitions, so an empty, untracked `families/definitions/`
    sitting in one working tree made the count 3 while a clean checkout saw 2 --
    and the documentation written from that count claimed a component the
    repository does not contain. The guard measured a working tree and the prose
    believed it. Requiring __init__.py measures the repository instead, and
    excludes __pycache__ for the same reason rather than by name.
    """
    return {
        d.name
        for d in FAMILIES_DIR.iterdir()
        if d.is_dir()
        and (d / "__init__.py").is_file()
        and not any((d / "definitions").glob("*.yaml"))
    }


def _tree_definition_count() -> int:
    return len(list(FAMILIES_DIR.glob("*/definitions/*.yaml")))


def _doc() -> str:
    return ARCHITECTURE.read_text(encoding="utf-8")


def _documented_families() -> set[str]:
    """First column of the family table: rows whose leading cell is a `backtick` id."""
    found = set()
    for line in _doc().splitlines():
        m = re.match(r"^\|\s*`([a-z0-9_]+)`\s*\|.*\|.*\|.*\|\s*$", line)
        if m:
            found.add(m.group(1))
    return found


def test_the_extractor_finds_a_table_at_all():
    """Anti-vacuity. Every assertion below is satisfied by an empty set."""
    assert len(_documented_families()) > 20, (
        "The family-table extractor found almost nothing. The table's shape changed "
        "and every check in this file is now passing on an empty set."
    )
    assert _tree_families(), "No family directory ships definitions -- the tree scan is wrong, not the doc."


def test_the_table_lists_exactly_the_families_that_exist():
    """THE guard: set equality, so it fails on add, remove, rename and typo alike."""
    tree, documented = _tree_families(), _documented_families()
    missing = sorted(tree - documented)
    ghosts = sorted(documented - tree)
    assert not missing and not ghosts, (
        f"ARCHITECTURE.md's family table does not match the tree.\n"
        f"  in the tree, absent from the table: {missing}\n"
        f"  in the table, absent from the tree: {ghosts}\n"
        "A family added to every ENUMERATING surface except this one is the exact "
        "failure CLAUDE.md warns about; add the row."
    )


def test_the_stated_family_and_definition_counts_are_the_real_ones():
    doc = _doc()
    families, definitions = len(_tree_families()), _tree_definition_count()
    assert f"**{families} shipped families across {definitions} definitions**" in doc, (
        f"The prose does not state the measured counts ({families} families, "
        f"{definitions} definitions). Update it, or explain why it disagrees."
    )
    assert f"# {families} families + {len(_tree_support_packages())} support packages" in doc, (
        "The directory tree's comment above the table disagrees with the tree. "
        "Two claims about one number, in one file, is how the last drift survived."
    )


def test_every_support_package_is_named_and_no_invented_one_is():
    """Support packages are the ones a reader will otherwise miscount as families."""
    doc = _doc()
    for name in _tree_support_packages():
        # `name` or `name/` -- a directory is naturally written with its slash.
        assert f"`{name}`" in doc or f"`{name}/`" in doc, (
            f"`{name}` ships no definitions, so it is a support package rather than a "
            "family. The prose enumerates them; it does not mention this one, so a "
            "reader counting directories gets a different answer than the table gives."
        )


@pytest.mark.parametrize(
    "archetype", ["latent_diffusion", "unified_transformer", "pixel_transformer"]
)
def test_each_archetype_in_use_is_named_in_the_prose(archetype: str):
    """The prose used to promise two archetypes while the tree had three."""
    in_tree = any(
        re.search(rf'archetype\s*=\s*"{archetype}"', (FAMILIES_DIR / f / "family.py").read_text(encoding="utf-8"))
        for f in _tree_families()
        if (FAMILIES_DIR / f / "family.py").is_file()
    )
    if not in_tree:
        pytest.skip(f"no family declares {archetype}")
    assert f"`{archetype}`" in _doc(), (
        f"A family declares archetype {archetype!r} and ARCHITECTURE.md never names it. "
        "The prose enumerates the archetypes as a closed set, so an unnamed one reads "
        "as impossible rather than as undocumented."
    )
