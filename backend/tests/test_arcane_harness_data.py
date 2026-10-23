"""Checks on this project's own harness data and release tooling.

`_harness/`, `.agent/` and `.claude/skills/` are gitignored, so public CI has
none of them: every test here SKIPS when the file it reads is absent.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_VENV_PY = REPO / "backend" / "venv" / "Scripts" / "python.exe"
PY = str(_VENV_PY) if _VENV_PY.is_file() else sys.executable
TOOLS = REPO / "_harness" / "tools"
HARNESS_JSON = REPO / "_harness" / "harness.json"
SKILL = REPO / ".claude" / "skills" / "bump-version" / "SKILL.md"
RUNPOD_SCRIPT = REPO / ".agent" / "scripts" / "sync_runpod_templates.py"


def _need(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"{path.relative_to(REPO).as_posix()} not present (gitignored; public CI has none)")


def _maintain(*args: str, timeout: int) -> tuple[subprocess.CompletedProcess, dict]:
    """Run `_harness/tools/maintain.py` on this checkout; return the run and its last JSON line."""
    _need(TOOLS)
    env = dict(os.environ, CLAUDE_PROJECT_DIR=str(REPO), HARNESS_GENERATION="live", PYTHONIOENCODING="utf-8")
    r = subprocess.run(
        [PY, str(TOOLS / "maintain.py"), "--root", str(REPO), "--generation", "live", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, cwd=str(REPO), timeout=timeout, check=False,
    )
    last = [ln for ln in (r.stdout or "").splitlines() if ln.strip().startswith("{")]
    return r, (json.loads(last[-1]) if last else {})


# ------------------------------------------------------------------ 1. moved-rule coverage


def test_the_coverage_check_passes_and_resolves_root_destinations():
    """Every rule that left a loaded surface (CLAUDE.md / AGENTS.md) is still in the
    surface it moved to, and that surface is named from the loaded text. `root/X`
    is the repo's own X, never `_harness/root/X`."""
    r, res = _maintain("budget", "--coverage", timeout=600)
    assert res.get("status") == "PASS" and r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    assert not any("_harness/root/" in f for f in res.get("findings") or []), res


# ------------------------------------------------------------------ 2. lessons essence headroom

ESSENCE_HEADROOM_LINES = 15


def test_the_lessons_essence_renders_under_its_budget_with_headroom():
    r, res = _maintain("essence", "--check", "--source", "lessons", timeout=300)
    assert res, (r.stdout, r.stderr)
    ev = res["evidence"]
    lines, limit = ev.get("lines"), ev.get("limit")
    assert isinstance(lines, int) and isinstance(limit, int), ev
    assert lines <= limit - ESSENCE_HEADROOM_LINES, (
        f"LESSONS-ESSENCE.md renders {lines}/{limit} lines, less than {ESSENCE_HEADROOM_LINES} from the budget: "
        "re-curate LESSONS-ESSENCE-MAP.json (fold overlapping rules; `maintain.py essence --rules` lists the keys). "
        "Raising `budgets.essence_lines` is not the remedy."
    )


# ------------------------------------------------------------------ 3. no bare `python` command value


def _bare_python_commands(cfg: dict) -> list[str]:
    """The harness.json keys (and `gate[i]` entries) whose command starts with the PATH interpreter."""
    bad = [k for k, v in cfg.items() if isinstance(v, str) and re.match(r"python(\.exe)?\s", v)]
    bad += [f"gate[{i}]" for i, v in enumerate(cfg.get("gate") or []) if re.match(r"python(\.exe)?\s", str(v))]
    return bad


def test_no_harness_json_command_value_starts_with_bare_python():
    """Every command value uses the venv interpreter (`./backend/venv/Scripts/python.exe ...`),
    never the PATH interpreter `python ...`."""
    _need(HARNESS_JSON)
    cfg = json.loads(HARNESS_JSON.read_text(encoding="utf-8-sig"))
    bad = _bare_python_commands(cfg)
    assert not bad, bad


def test_the_bare_python_check_lists_a_bare_python_command():
    """Positive control: the check above asserts an EMPTY list, which a dead pattern
    also yields. Reads no file, so it runs in public CI too."""
    cfg = {
        "x_tool": "python -m pytest",
        "y_tool": "./backend/venv/Scripts/python.exe -m pytest",
        "gate": ["python.exe -m ruff check .", "./backend/venv/Scripts/python.exe -m ruff check ."],
    }
    assert _bare_python_commands(cfg) == ["x_tool", "gate[0]"]


# ------------------------------------------------------------------ 4. the bump-version skill

_SHIPS_THROUGH_THE_TOOL = re.compile(r"gate\.py\s+release-ship\b")
#: auto mode finds the previous bump by grepping for exactly this phrase in the subject.
_MANDATED_SUBJECT = re.compile(r"commit subject MUST contain the literal words\s+`([^`]+)`", re.IGNORECASE)


def test_the_bump_version_skill_states_the_release_procedure():
    """The installed skill still releases through `gate.py release-ship` and still
    mandates the commit subject (that part comes from the project extension)."""
    _need(SKILL)
    text = SKILL.read_text(encoding="utf-8")
    for name, rx in (
        ("the release through `gate.py release-ship`", _SHIPS_THROUGH_THE_TOOL),
        ("the mandated commit subject", _MANDATED_SUBJECT),
    ):
        assert rx.search(text), f"{name} is not in {SKILL.relative_to(REPO).as_posix()}"


# ------------------------------------------------------------------ 5. RunPod template sync
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
