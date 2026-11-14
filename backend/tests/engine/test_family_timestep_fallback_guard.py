"""No family module carries its own fallback for a timestep setting.

A family's timestep default lives in its definition (``timestep.*`` keys or its ``defaults:``
block) and reaches the job's config in the shared ``setup()``; a value the config lacks after that
is read through ``timestep_setting``. A literal or constant fallback in family code is a second
copy of a default that the form never sees and that drifts from the definition, so this scan
fails on any ``.get(<timestep key>, <fallback>)`` with a fallback other than ``None`` and on any
``.setdefault(<timestep key>, ...)`` in a module under ``app/engine/models/families/``.
"""

from __future__ import annotations

import ast
from pathlib import Path

from app.engine.strategies.timestep_sampling import TIMESTEP_PARAMETERS

FAMILIES_DIR = Path(__file__).resolve().parents[2] / "app" / "engine" / "models" / "families"

TIMESTEP_KEYS = frozenset(
    {"timestep_sampling", "logit_normal_mean", "logit_normal_std", *TIMESTEP_PARAMETERS}
)


def _is_timestep_key(node: ast.AST) -> bool:
    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
        return False
    return node.value in TIMESTEP_KEYS or node.value.startswith("timestep.")


def fallbacks(source: str, filename: str = "<fixture>") -> list[str]:
    """Each ``file:line: call`` that reads a timestep key with its own fallback."""
    found = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args):
            continue
        if not _is_timestep_key(node.args[0]):
            continue
        name = node.func.attr
        if name == "setdefault" or (
            name == "get"
            and len(node.args) >= 2
            and not (isinstance(node.args[1], ast.Constant) and node.args[1].value is None)
        ):
            found.append(f"{filename}:{node.lineno}: {ast.unparse(node)}")
    return found


def test_no_family_module_has_its_own_timestep_fallback():
    found = []
    for path in sorted(FAMILIES_DIR.rglob("*.py")):
        found += fallbacks(path.read_text(encoding="utf-8"), str(path.relative_to(FAMILIES_DIR)))
    assert not found, "family code with its own timestep fallback:\n" + "\n".join(found)


FIXTURE = '''
class Trainer:
    DEFAULT_MODE = "mode"

    def draw(self, config, arch):
        a = config.get("timestep_sampling", "uniform")
        b = config.get("mode_scale", self.DEFAULT_MODE)
        c = arch.get("timestep.logit_normal_mu", 1.09)
        config.setdefault("flux_shift_max", 1.15)
        d = config.get("timestep_sampling")
        e = config.get("timestep_sampling", None)
        f = config.get("learning_rate", 1e-4)
        return a, b, c, d, e, f
'''


def test_the_scan_finds_each_kind_of_fallback_and_nothing_else():
    lines = [entry.split(":")[1] for entry in fallbacks(FIXTURE)]
    assert lines == ["6", "7", "8", "9"]
