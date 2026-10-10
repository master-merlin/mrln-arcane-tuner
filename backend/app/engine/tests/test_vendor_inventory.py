"""The vendor inventory binds to the tree and to the installed diffusers (LANE-149).

``backend/app/engine/models/families/VENDOR_INVENTORY.md`` is LANE-150's
retirement list: one row per vendored ``.py`` file under
``families/<family>/vendor/``, each with a verdict -- ``shipped`` (the pinned
diffusers ships an equivalent of the whole module), ``partial`` (part of it) or
``none`` -- and, for ``shipped``/``partial``, the dotted ``diffusers....``
symbol(s) of that equivalent.

What this file holds the table to (each failure names the row or the module):

* the row set equals the vendored-file set, both directions, no duplicates;
* every row's ``lines`` cell is the file's current line count (a refreshed or
  edited vendor file invalidates the comparison the row records);
* every vendored module imports on the installed release;
* every ``shipped``/``partial`` row names at least one symbol and every symbol
  resolves by import on the installed diffusers; a ``none`` row names none;
* every verdict is one of the three words.

The verdicts are checked here by import and never copied into a second list
(RULE-21): the table is the one producer.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

FAMILIES_DIR = Path(__file__).resolve().parents[1] / "models" / "families"
INVENTORY = FAMILIES_DIR / "VENDOR_INVENTORY.md"
HEADER = (
    "family",
    "module",
    "lines",
    "verdict",
    "0.41 symbol(s)",
    "known differences",
    "LANE-150 action",
)
VERDICTS = frozenset({"shipped", "partial", "none"})
_SYMBOL = re.compile(r"`(diffusers(?:\.\w+)+)`")


def _vendor_files() -> list[str]:
    """Every vendored ``.py`` file, as a posix path relative to ``families/``."""
    return sorted(
        p.relative_to(FAMILIES_DIR).as_posix()
        for p in FAMILIES_DIR.glob("*/vendor/**/*.py")
        if "__pycache__" not in p.parts
    )


VENDOR_FILES = _vendor_files()


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _rows() -> list[dict[str, str]]:
    """The inventory's data rows, keyed by HEADER; the table's header must match."""
    if not INVENTORY.is_file():
        pytest.fail(f"LANE-149: the vendor inventory {INVENTORY} does not exist")
    lines = INVENTORY.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("|") and tuple(_cells(line)) == HEADER:
            start = i + 2  # skip the |---| separator
            break
    else:
        pytest.fail(f"LANE-149: no table with the header {HEADER} in {INVENTORY.name}")
    rows = []
    for line in lines[start:]:
        if not line.lstrip().startswith("|"):
            break
        cells = _cells(line)
        assert len(cells) == len(HEADER), (
            f"row has {len(cells)} cells, wants {len(HEADER)}: {line}"
        )
        row = dict(zip(HEADER, cells))
        row["module"] = row["module"].strip("`")
        rows.append(row)
    return rows


def _dotted(rel: str) -> str:
    return "app.engine.models.families." + rel[: -len(".py")].replace(
        "/", "."
    ).removesuffix(".__init__")


def _resolve(dotted: str) -> object:
    """Import the longest importable module prefix of ``dotted``, getattr the rest."""
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for attr in parts[cut:]:
            obj = getattr(obj, attr)  # AttributeError names the missing part
        return obj
    raise ImportError(f"no importable prefix of {dotted}")


def test_every_vendored_module_has_one_row():
    rows = _rows()
    modules = [r["module"] for r in rows]
    files = set(VENDOR_FILES)
    without_row = sorted(files - set(modules))
    without_module = sorted(set(modules) - files)
    duplicates = sorted({m for m in modules if modules.count(m) > 1})
    problems = (
        [f"vendored module without a row: {m}" for m in without_row]
        + [f"row without a vendored module: {m}" for m in without_module]
        + [f"duplicate row: {m}" for m in duplicates]
    )
    assert not problems, "LANE-149: " + "; ".join(problems)


def test_every_row_states_the_current_line_count():
    problems = []
    for row in _rows():
        path = FAMILIES_DIR / row["module"]
        if not path.is_file():
            continue  # test_every_vendored_module_has_one_row names it
        have = len(path.read_text(encoding="utf-8").splitlines())
        if row["lines"] != str(have):
            problems.append(
                f"{row['module']}: row says {row['lines']} lines, file has {have}"
            )
    assert not problems, "LANE-149: " + "; ".join(problems)


@pytest.mark.parametrize("rel", VENDOR_FILES)
def test_every_vendored_module_imports_on_the_pinned_release(rel):
    import diffusers

    try:
        importlib.import_module(_dotted(rel))
    except Exception as exc:  # noqa: BLE001 - the finding is the exception itself
        pytest.fail(
            f"LANE-149: vendored module {rel} does not import on diffusers "
            f"{diffusers.__version__}: {type(exc).__name__}: {exc}"
        )


def test_every_claimed_symbol_resolves():
    import diffusers

    problems = []
    for row in _rows():
        symbols = _SYMBOL.findall(row["0.41 symbol(s)"])
        if row["verdict"] in ("shipped", "partial"):
            if not symbols:
                problems.append(
                    f"{row['module']}: verdict {row['verdict']} names no diffusers symbol"
                )
            for sym in symbols:
                try:
                    _resolve(sym)
                except (ImportError, AttributeError) as exc:
                    problems.append(f"{row['module']}: {sym} does not resolve ({exc})")
        elif symbols:
            problems.append(
                f"{row['module']}: verdict {row['verdict']} names symbols {symbols}"
            )
    assert not problems, f"LANE-149 (diffusers {diffusers.__version__}): " + "; ".join(
        problems
    )


def test_verdicts_are_the_three_words():
    bad = [
        f"{r['module']}: {r['verdict']!r}"
        for r in _rows()
        if r["verdict"] not in VERDICTS
    ]
    assert not bad, f"LANE-149: verdicts outside {sorted(VERDICTS)}: " + "; ".join(bad)
