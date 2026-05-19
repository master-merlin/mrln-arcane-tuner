"""Regression guard: every `print(` in in-scope files must be annotated.

Enforces the safety-net annotation convention defined in
`docs/LOGGING.md` `<the_golden_rules>` `[THE PRINT BOUNDARY]`: any
`print()` call in production code must either be removed or carry an
inline comment of the form `# safety-net print: <reason>` either on
the same line or within the 3 lines immediately preceding the call.

If this test fails, either:
  1. Remove the new print() (route through JobLogWriter / logger), or
  2. If it is genuinely a safety net (bootstrap or fail-safe), annotate
     it with `# safety-net print: <reason>` and update docs/LOGGING.md
     if the reason class is new.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Files in scope for R-LOG-03 + this regression guard. New production
# Python files added under backend/app/engine/ or backend/run_trainer.py
# are automatically covered by the glob; add others here explicitly if
# they should also be covered.
IN_SCOPE = [
    REPO_ROOT / "backend" / "run_trainer.py",
    *(REPO_ROOT / "backend" / "app" / "engine").rglob("*.py"),
]

PRINT_PATTERN = re.compile(r"\bprint\s*\(")
ANNOTATION_PATTERN = re.compile(r"#\s*safety-net print:")


def _violating_lines(path: Path) -> list[tuple[int, str]]:
    """Return (line_number, line_text) for unannotated print() calls."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    violations: list[tuple[int, str]] = []

    for i, line in enumerate(lines):
        if not PRINT_PATTERN.search(line):
            continue
        # Skip comments (line starts with # after leading whitespace)
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        # Annotation may be on the same line or on the 3 preceding lines
        same_line_ok = ANNOTATION_PATTERN.search(line) is not None
        preceding = lines[max(0, i - 3):i]
        preceding_ok = any(ANNOTATION_PATTERN.search(p) for p in preceding)
        if not (same_line_ok or preceding_ok):
            violations.append((i + 1, line))

    return violations


@pytest.mark.parametrize(
    "path",
    IN_SCOPE,
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_no_unannotated_prints(path: Path) -> None:
    """Every print() call in `path` must be annotated as a safety net."""
    if not path.exists() or not path.is_file():
        pytest.skip(f"{path} does not exist (filter glob mismatch)")

    violations = _violating_lines(path)
    if violations:
        formatted = "\n".join(
            f"  {path.relative_to(REPO_ROOT)}:{lineno}: {text.strip()}"
            for lineno, text in violations
        )
        pytest.fail(
            f"Unannotated print() call(s) found in {path.relative_to(REPO_ROOT)}:\n"
            f"{formatted}\n\n"
            "Either remove the print (route through JobLogWriter / logger), "
            "or annotate it with `# safety-net print: <reason>` "
            "(see docs/LOGGING.md THE PRINT BOUNDARY)."
        )
