"""Static lint: no heavy sync I/O inside `async def` handlers under app/api/.

This guards against future regressions where a contributor adds sync DB or
filesystem I/O directly in a FastAPI route handler, which would block the
event loop and cascade latency to every concurrent request.

Patterns are listed in tests/_lint_data/sync_io_forbidden.txt. Known-accepted
exceptions live in sync_io_allowed.txt.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent / "app" / "api"
DATA_DIR = Path(__file__).parent / "_lint_data"
FORBIDDEN_FILE = DATA_DIR / "sync_io_forbidden.txt"
ALLOWED_FILE = DATA_DIR / "sync_io_allowed.txt"


def _load_lines(p: Path) -> list[str]:
    return [
        line for line in p.read_text(encoding="utf-8").splitlines()
        if line and not line.lstrip().startswith("#")
    ]


def _load_patterns() -> list[re.Pattern[str]]:
    return [re.compile(line) for line in _load_lines(FORBIDDEN_FILE)]


def _load_allowlist() -> set[str]:
    """Returns the set of '<rel-path>:<line>' allowlisted entries."""
    entries = _load_lines(ALLOWED_FILE)
    keys: set[str] = set()
    for entry in entries:
        # Format: "<path>:<line>: <comment>" — strip the comment
        if ":" not in entry:
            continue
        head = entry.split(": ", 1)[0] if ": " in entry else entry
        keys.add(head.strip())
    return keys


def _strip_nested_defs(source: str) -> str:
    """Replace lines inside nested `def _foo(...):` blocks with blank lines.

    Tracks indentation: any line indented MORE than the `def` line is part of
    that nested function (which runs in `asyncio.to_thread`, so it's exempt).
    Outer `async def` handler lines are kept.
    """
    lines = source.splitlines()
    out: list[str] = []
    in_nested = False
    nested_indent = 0
    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if in_nested:
            if not stripped or indent > nested_indent:
                out.append("")  # blank out content of nested def
                continue
            else:
                in_nested = False
        if stripped.startswith("def ") and not stripped.startswith("def __init__"):
            # nested def inside an async handler (we only run this on bodies of async handlers)
            out.append("")
            in_nested = True
            nested_indent = indent
            continue
        out.append(line)
    return "\n".join(out)


def _async_handler_bodies(source: str):
    """Yield (handler_name, body_with_line_offsets) for each `async def` block."""
    matches = list(re.finditer(r"(?ms)^async def (\w+).*?(?=^(?:async )?def |\Z)", source))
    for m in matches:
        base_line = source[: m.start()].count("\n") + 1
        yield m.group(1), m.group(0), base_line


def test_no_sync_io_in_async_handlers() -> None:
    patterns = _load_patterns()
    allowlist = _load_allowlist()
    repo_root = ROOT.parent.parent.parent  # backend/app/api -> backend/app -> backend -> repo

    violations: list[str] = []

    for py in ROOT.rglob("*.py"):
        src = py.read_text(encoding="utf-8")
        for name, body, base_line in _async_handler_bodies(src):
            cleaned = _strip_nested_defs(body)
            for line_idx, line in enumerate(cleaned.splitlines()):
                if "to_thread" in line or "aiofiles" in line:
                    continue
                for pat in patterns:
                    if pat.search(line):
                        absolute_line = base_line + line_idx
                        rel_path = py.relative_to(repo_root).as_posix()
                        key = f"{rel_path}:{absolute_line}"
                        if key in allowlist:
                            continue
                        violations.append(
                            f"{rel_path}:{absolute_line}: in async def {name}: {line.strip()}"
                        )

    assert not violations, (
        "Sync I/O detected inside async handlers. Wrap with asyncio.to_thread "
        "or aiofiles. If genuinely intentional, add the file:line key to "
        "backend/tests/_lint_data/sync_io_allowed.txt with a comment.\n\n"
        + "\n".join(violations)
    )
