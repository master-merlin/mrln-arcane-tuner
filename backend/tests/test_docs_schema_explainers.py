"""`backend/scripts/docs_schema_explainers.py` keeps `docs/training-guide.md` in step
with the training schema and `frontend/public/config_help.json` (LANE-72).

These read only committed files, so they run everywhere `pytest backend` runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import docs_schema_explainers as dse  # noqa: E402


def test_check_is_clean_on_the_real_committed_guide():
    """The COMMITTED `docs/training-guide.md` is not stale against the real
    schema + real config_help.json (adversary LANE-72 round 1, MAJOR 2)."""
    schema = dse._load_schema()
    config_help = dse._load_config_help()
    blocks, _, _ = dse.render_all(schema, config_help)
    text = dse._GUIDE.read_text(encoding="utf-8")
    assert dse.check(text, blocks) == []


def test_a_renamed_heading_is_reported_as_a_MISSING_ANCHOR_not_as_stale_blocks(
    tmp_path, capsys, monkeypatch
):
    """The heading is a machine anchor, and the tool used to hide that.

    2026-09-06: a docs pass renamed `## Field reference (generated from the
    schema)` to `## Base schema field reference`. `--check` then reported all
    NINE groups as stale while every generated block was byte-identical, and
    told the reader to run `--write`, which would have rebuilt the section
    instead of restoring one line. This pins the honest diagnosis.
    """
    schema = dse._load_schema()
    config_help = dse._load_config_help()
    blocks, _, _ = dse.render_all(schema, config_help)

    generated, _ = dse.apply("# Training\n\n## Where things live\n\nDetails.\n", blocks)
    renamed = generated.replace(dse._SECTION_HEADING, "## Base schema field reference")
    assert renamed != generated, "the heading must actually have been replaced"

    guide = tmp_path / "training-guide.md"
    guide.write_text(renamed, encoding="utf-8")
    monkeypatch.setattr(dse, "_GUIDE", guide)
    monkeypatch.setattr(sys, "argv", ["docs_schema_explainers.py", "--check"])

    assert dse.main() == 1
    out = capsys.readouterr().out

    # Says what is actually wrong, and quotes the line it wants back.
    assert "section heading not found" in out
    assert dse._SECTION_HEADING in out
    # Does NOT send the reader to --write, which would rebuild the section.
    assert dse._WRITE_ADVICE not in out
    # And does not claim the blocks are stale, because they are not.
    assert "stale/missing blocks for groups" not in out


def test_genuinely_stale_blocks_still_report_as_stale_and_still_advise_write(
    tmp_path, capsys, monkeypatch
):
    """The negative control for the test above: heading intact, one block
    hand-edited -- the stale message and the `--write` advice must both survive,
    or the fix could be 'never report stale blocks'."""
    schema = dse._load_schema()
    config_help = dse._load_config_help()
    blocks, _, _ = dse.render_all(schema, config_help)

    generated, _ = dse.apply("# Training\n\n## Where things live\n\nDetails.\n", blocks)
    tampered = generated.replace(dse._SECTION_INTRO, "A hand-edited intro paragraph.")
    assert tampered != generated

    guide = tmp_path / "training-guide.md"
    guide.write_text(tampered, encoding="utf-8")
    monkeypatch.setattr(dse, "_GUIDE", guide)
    monkeypatch.setattr(sys, "argv", ["docs_schema_explainers.py", "--check"])

    assert dse.main() == 1
    out = capsys.readouterr().out
    assert "stale/missing blocks for groups" in out
    assert "Run: python backend/scripts/docs_schema_explainers.py --write" in out
    assert "section heading not found" not in out
