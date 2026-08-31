"""A process we spawn may not have its output thrown away unlisted (LANE-51).

``/api/system/restart`` sent the replacement server's stdout and stderr to
DEVNULL to fix a real freeze (a dead inherited pipe blocking the logging lock,
``e2e3cfc8``). The trade was never revisited, and for six weeks a restart that
failed produced no trace on the console, in ``server.log`` or in the app.

DEVNULL is not banned — sometimes the output genuinely lives elsewhere. What is
banned is doing it **silently**: every site must appear below with the place its
output actually goes, so the next person choosing DEVNULL has to answer the
question rather than inherit the answer.
"""

from __future__ import annotations

import ast
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]

#: (module relative to backend/, enclosing function) -> where the output goes.
#: A row here turns the check OFF for one call site and carries its evidence.
ALLOWED: dict[tuple[str, str], str] = {
    # Evidence: app/engine/models/training_plugin.py:113-123 — the boot output
    # goes to <output_dir>/trainer_stdout.log and the run itself to
    # job_log.jsonl via JobLogWriter; DEVNULL is only the fallback for when
    # that file cannot be opened.
    ("app/engine/models/training_plugin.py", "start_training"):
        "trainer_stdout.log + job_log.jsonl (DEVNULL is the open() fallback)",
    # KNOWN GAP, listed rather than hidden: a frontend autostart that fails
    # (`ng serve` missing, port taken, node absent) is silent exactly the way
    # LANE-51's restart was. Out of that lane's scope; whoever touches this
    # next owes it a log target, not another DEVNULL.
    ("app/main.py", "_maybe_start_frontend"):
        "KNOWN GAP - a failed `ng serve` is invisible; see LESSONS 2026-08-31",
}

_SPAWNERS = {"Popen", "run", "call", "check_call", "check_output"}


def _devnull_aliases(node: ast.AST) -> set[str]:
    """Local names bound to DEVNULL in this scope.

    Without this the check is trivially defeated by one assignment — and not
    hypothetically: ``training_plugin.start_training`` does exactly that as a
    fallback. An alias is how a scan of this shape goes quietly blind.
    """
    names: set[str] = set()

    def scan(current: ast.AST) -> None:
        for child in ast.iter_child_nodes(current):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # its own scope, collected when we descend into it
            if isinstance(child, ast.Assign) and isinstance(child.value, ast.Attribute) \
                    and child.value.attr == "DEVNULL":
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            scan(child)

    scan(node)
    return names


def _discards_output(call: ast.Call, aliases: set[str]) -> bool:
    """True for ``subprocess.<spawner>(…, stdout|stderr=DEVNULL)``, alias or not."""
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr not in _SPAWNERS:
        return False
    for kw in call.keywords:
        if kw.arg not in ("stdout", "stderr"):
            continue
        if isinstance(kw.value, ast.Attribute) and kw.value.attr == "DEVNULL":
            return True
        if isinstance(kw.value, ast.Name) and kw.value.id in aliases:
            return True
    return False


def _discarding_calls(tree: ast.AST) -> list[tuple[str, int]]:
    """(INNERMOST enclosing function, line) for every spawn that discards.

    An explicit descent, not ``ast.walk``: walk flattens the tree, so a call
    inside a nested ``def`` would be attributed to whichever ancestor happened
    to be visited — and an allow-list keyed on the wrong function silences the
    wrong call site.
    """
    found: list[tuple[str, int]] = []

    def visit(node: ast.AST, enclosing: str, aliases: set[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child.name, aliases | _devnull_aliases(child))
                continue
            if isinstance(child, ast.Call) and _discards_output(child, aliases):
                found.append((enclosing, child.lineno))
            visit(child, enclosing, aliases)

    visit(tree, "<module>", _devnull_aliases(tree))
    return found


def _first_party_modules() -> list[Path]:
    skip = {"venv", "tests", "vendor", "__pycache__", "node_modules"}
    out = []
    for path in list((BACKEND / "app").rglob("*.py")) + list(BACKEND.glob("*.py")):
        if any(part in skip for part in path.parts):
            continue
        out.append(path)
    return out


def test_every_discarded_output_names_where_the_output_went():
    offenders = []
    for path in _first_party_modules():
        rel = path.relative_to(BACKEND).as_posix()
        # utf-8-sig: a BOM makes ast.parse raise, and a scan that ABORTS reads
        # exactly like a scan that found nothing (LESSONS 2026-08-25).
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for func, line in _discarding_calls(tree):
            if (rel, func) not in ALLOWED:
                offenders.append(f"{rel}:{line} in {func}()")
    assert not offenders, (
        "these spawns discard their child's output with no record of where it "
        "goes — give the child a log file, or add the site to ALLOWED with the "
        "place its output actually lands:\n  " + "\n  ".join(offenders))


def test_the_scanner_catches_a_new_offender():
    """Positive control. Without it, a drifted AST shape or a renamed directory
    would leave this test permanently green with nobody the wiser."""
    source = """
import subprocess

def spawn_something():
    subprocess.Popen(["x"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
"""
    found = _discarding_calls(ast.parse(source))
    assert found == [("spawn_something", 5)], found


def test_the_scanner_catches_an_offender_hiding_behind_an_alias():
    """The exact evasion that would have made this guard vacuous."""
    source = """
import subprocess

def spawn_something():
    sink = subprocess.DEVNULL
    subprocess.Popen(["x"], stdout=sink, stderr=sink)
"""
    assert _discarding_calls(ast.parse(source)) == [("spawn_something", 6)]


def test_an_alias_does_not_leak_between_functions():
    source = """
import subprocess

def one():
    sink = subprocess.DEVNULL

def two(sink):
    subprocess.Popen(["x"], stdout=sink)
"""
    assert _discarding_calls(ast.parse(source)) == []


def test_the_scanner_ignores_a_spawn_that_keeps_its_output():
    source = """
import subprocess

def spawn_something(handle):
    subprocess.Popen(["x"], stdin=subprocess.DEVNULL, stdout=handle, stderr=handle)
"""
    assert _discarding_calls(ast.parse(source)) == []


def test_the_allow_list_has_no_stale_rows():
    """A row that no longer matches a call site is a comment pretending to be a
    check; it must be deleted when its site is fixed."""
    live = set()
    for path in _first_party_modules():
        rel = path.relative_to(BACKEND).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for func, _line in _discarding_calls(tree):
            live.add((rel, func))
    stale = [key for key in ALLOWED if key not in live]
    assert not stale, f"ALLOWED rows with no matching call site any more: {stale}"
