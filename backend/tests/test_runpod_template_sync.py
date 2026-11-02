"""`.agent/scripts/sync_runpod_templates.py` keeps the RunPod templates in step with a release.

The script is gitignored, so public CI has none: every test here SKIPS when it is absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUNPOD_SCRIPT = REPO / ".agent" / "scripts" / "sync_runpod_templates.py"


def _need(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"{path.relative_to(REPO).as_posix()} not present (gitignored; public CI has none)")


# `.agent/scripts/sync_runpod_templates.py` must refresh every field it claims to
# refresh (the size was never rewritten: 14.8 GB stayed in two places for 14.4 GB),
# and must not duplicate a history entry when run twice (found 2026-09-07,
# publishing v0.8.0-beta.3). Pure-text functions, pinned without docker.

#: The wording as it actually appears in both templates.
HEADER = (
    "This image is version **0.8.0-beta.2**, commit `deadbeef`, digest "
    "`sha256:994255cb…` — CUDA 12.6, torch 2.11.0+cu126, 14.8 GB unpacked."
)
DISK = (
    "**Container Disk: at least 25 GB.** The image alone is 14.8 GB unpacked "
    "and the writable layer needs room on top."
)
HISTORY = "\n## Version History\n\n* **0.8.0-beta.2** — the previous release.\n"


@pytest.fixture(scope="module")
def runpod_mod():
    _need(RUNPOD_SCRIPT)
    spec = importlib.util.spec_from_file_location("_sync_runpod_under_test", RUNPOD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: the module defines a dataclass, and dataclasses
    # resolves annotations through sys.modules[cls.__module__] at class-creation
    # time. Without this the import dies inside dataclasses, not in our code.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def test_rewrite_replaces_every_size_claim(runpod_mod):
    """The defect: the size stayed at the previous release's number."""
    text = f"{HEADER}\n\n{DISK}\n{HISTORY}"
    out = runpod_mod._rewrite(
        text,
        version="0.8.0-beta.3",
        commit="01efaf5f",
        digest="sha256:994255cb522dfdfb669c459168d6715ef73fcd732d383152b350e6491eb022c3",
        old_version="0.8.0-beta.2",
        size="14.4 GB unpacked",
    )
    assert "14.8 GB" not in out, (
        "a size from the previous release survived the rewrite:\n" + out
    )
    # BOTH occurrences, not just the header — the disk advice is the one a user
    # acts on, and it is the one that reads as prose rather than as a field.
    assert out.count("14.4 GB unpacked") == 2, out


def test_the_size_pattern_matches_the_wording_that_actually_ships(runpod_mod):
    """Positive control. `not in` proves nothing unless the pattern is shown to
    fire on the real strings — an over-narrow regex would pass the test above
    by matching nothing and replacing nothing."""
    assert runpod_mod._SIZE_RE.findall(HEADER) == ["14.8 GB unpacked"]
    assert runpod_mod._SIZE_RE.findall(DISK) == ["14.8 GB unpacked"]
    assert runpod_mod._SIZE_RE.findall("16.9 GB unpacked") == ["16.9 GB unpacked"]
    # …and must not swallow neighbouring prose or an unrelated size.
    assert runpod_mod._SIZE_RE.findall("at least 25 GB.") == []


def test_rewrite_demands_a_size(runpod_mod):
    """Deleting the parameter must break loudly, not silently stop updating it."""
    with pytest.raises(TypeError):
        runpod_mod._rewrite(
            "x", version="1.0.0", commit="abcdef12",
            digest="sha256:" + "0" * 64, old_version="0.9.0",
        )


def test_insert_todo_is_idempotent_once_the_note_is_written(runpod_mod):
    """The marker is GONE after the note is written; that is not permission to
    add another bullet for the same release."""
    written = (
        "## Version History\n\n"
        "* **0.8.0-beta.3** — a real release note, written by hand.\n"
        "* **0.8.0-beta.2** — the previous release.\n"
    )
    assert runpod_mod._insert_todo(written, "0.8.0-beta.3") == written


def test_insert_todo_still_inserts_for_a_genuinely_new_version(runpod_mod):
    """Negative control: the fix must not degenerate into 'never insert'."""
    written = (
        "## Version History\n\n"
        "* **0.8.0-beta.3** — a real release note, written by hand.\n"
    )
    out = runpod_mod._insert_todo(written, "0.8.0-beta.4")
    assert out != written
    assert out.count("* **0.8.0-beta.4**") == 1
    assert runpod_mod._TODO in out
    # The new entry goes on TOP, where the history is read from.
    assert out.index("0.8.0-beta.4") < out.index("0.8.0-beta.3")


def test_insert_todo_still_skips_while_a_marker_survives(runpod_mod):
    """The original guard must keep working."""
    pending = f"## Version History\n\n* **9.9.9** — {runpod_mod._TODO}: one sentence.\n"
    assert runpod_mod._insert_todo(pending, "9.9.9") == pending
