"""The README's family table, pinned to the tree — and its link pinned to a real anchor.

It listed three families of twenty-eight. Not stale in the ordinary sense: it
was written when three existed and then twenty-five arrived, each added to the
registry, the definition loader and the coverage tables, and none of them to the
document a stranger reads first. LANE-8 measured it and deliberately did not
guess at a fix; UAT round 2 answered "fix the counts and update everything to
as-is".

Set equality, not a count, for the reason given in
``test_architecture_family_counts.py``: one integer can stay right while the
table goes wrong in two places at once.

The anchor check earns its place separately. This table's intro sends the reader
to ARCHITECTURE.md for archetypes and capability flags, and the first draft of
that link pointed at ``#model-families`` when the document had no such heading —
bold prose is not a heading and GitHub renders no anchor for it. A cross-doc
link that silently lands at the top of the page is the same defect class as a
doc reference to a missing image, and it is invisible to every check that reads
only one file at a time.
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
README = REPO / "README.md"
ARCHITECTURE = REPO / "docs" / "ARCHITECTURE.md"  # moved from documentation/ 2026-09-01, LANE-62; a redirect stub holds the old path
FAMILIES_DIR = REPO / "backend" / "app" / "engine" / "models" / "families"

#: The heading the README's families intro links to.
LINKED_ANCHOR = "model-families"


def _tree_families() -> set[str]:
    return {
        d.name
        for d in FAMILIES_DIR.iterdir()
        if d.is_dir() and d.name != "__pycache__" and any((d / "definitions").glob("*.yaml"))
    }


def _tree_definition_count() -> int:
    return len(list(FAMILIES_DIR.glob("*/definitions/*.yaml")))


def _readme_section() -> str:
    """Only the families section: a `backtick` row elsewhere must not count."""
    text = README.read_text(encoding="utf-8")
    start = text.index("#### Supported Model Families")
    rest = text[start + 1 :]
    end = rest.find("\n#### ")
    return rest if end == -1 else rest[:end]


def _documented_families() -> set[str]:
    return set(re.findall(r"^\|\s*`([a-z0-9_]+)`\s*\|", _readme_section(), re.M))


def _github_anchor(heading: str) -> str:
    """GitHub's slug: lowercase, drop punctuation, spaces to hyphens."""
    slug = heading.strip().lstrip("#").strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    return re.sub(r"\s+", "-", slug)


def test_the_extractor_finds_the_section_and_rows():
    """Anti-vacuity: every assertion below passes on an empty set."""
    assert len(_documented_families()) > 20, (
        "The README families extractor found almost nothing. The section moved or "
        "its table shape changed, and the checks below are now vacuous."
    )


def test_the_readme_lists_exactly_the_families_that_exist():
    tree, documented = _tree_families(), _documented_families()
    missing = sorted(tree - documented)
    ghosts = sorted(documented - tree)
    assert not missing and not ghosts, (
        "README's family table does not match the tree.\n"
        f"  in the tree, absent from the README: {missing}\n"
        f"  in the README, absent from the tree: {ghosts}\n"
        "This is the document a stranger reads first; a family missing here is "
        "invisible to everyone who has not read the registry."
    )


def test_the_stated_totals_are_the_real_ones():
    families, definitions = len(_tree_families()), _tree_definition_count()
    assert f"**{families} families, {definitions} shipped definitions.**" in _readme_section(), (
        f"The README's headline totals are not the measured ones ({families} families, "
        f"{definitions} definitions)."
    )


def test_the_modality_blocks_partition_the_families():
    """Every family sits in exactly one block, and the block sizes are stated.

    Without this, a family can be listed twice — once under Image and once under
    Video — and the set check above still passes, because a set does not count.
    """
    section = _readme_section()
    listed = re.findall(r"^\|\s*`([a-z0-9_]+)`\s*\|", section, re.M)
    duplicated = sorted({f for f in listed if listed.count(f) > 1})
    assert not duplicated, f"listed in more than one modality block: {duplicated}"

    stated = sum(int(n) for n in re.findall(r"^\*\*\w+ — (\d+) famil", section, re.M))
    assert stated == len(listed) == len(_tree_families()), (
        f"The modality headings claim {stated} families, the table has {len(listed)} "
        f"rows, and the tree has {len(_tree_families())}. All three must agree."
    )


def test_the_architecture_link_points_at_a_heading_that_exists():
    section = _readme_section()
    assert f"ARCHITECTURE.md#{LINKED_ANCHOR}" in section, (
        f"The families intro no longer links to ARCHITECTURE.md#{LINKED_ANCHOR}. If the "
        "link was moved or removed, move this check with it rather than deleting it."
    )
    anchors = {
        _github_anchor(line)
        for line in ARCHITECTURE.read_text(encoding="utf-8").splitlines()
        if line.startswith("#")
    }
    assert LINKED_ANCHOR in anchors, (
        f"README links to ARCHITECTURE.md#{LINKED_ANCHOR} and that document has no heading "
        f"with that slug. Its headings resolve to: {sorted(anchors)}. A link to a missing "
        "anchor does not 404 — it lands silently at the top of the page, which is why "
        "nobody reports it."
    )
