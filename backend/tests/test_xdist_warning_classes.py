"""No warning category may be defined inside a test module (LANE-63, xdist).

pytest-xdist serialises a warning as ``(message_module, message_class_name)``
and the CONTROLLER re-imports ``message_module`` by name to rebuild it
(``xdist/workermanage.py`` ``unserialize_warning_message`` — an
``importlib.import_module`` with no fallback). Test modules are imported by
pytest under ``--import-mode=importlib`` names that only the importing worker
knows (``api.test_csp_policy``, because ``tests/api`` has an ``__init__.py``
and ``tests/`` does not); the controller's import fails, the worker is marked
down, and the loadgroup scheduler crashes with ``KeyError`` — measured
2026-09-02: ``-n 8`` died at 15% over ONE ``SourceFallbackWarning``.

A warning class belongs in an importable module (``app/...`` or
``tests/support/...``). This scan is AST-based, so it costs nothing to run and
cannot be fooled by an import that happens to work in the scanning process.
"""
import ast
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
ROOTS = [BACKEND / "tests", BACKEND / "app" / "engine" / "tests", BACKEND / "app" / "api" / "tests"]


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def warning_classes_in_test_modules(roots=ROOTS) -> list[str]:
    hits = []
    for root in roots:
        for path in sorted(root.rglob("test_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and any(
                    _base_name(b).endswith("Warning") for b in node.bases
                ):
                    shown = path.relative_to(BACKEND) if path.is_relative_to(BACKEND) else path
                    hits.append(f"{shown}:{node.lineno} {node.name}")
    return hits


def test_no_warning_category_is_defined_in_a_test_module():
    hits = warning_classes_in_test_modules()
    assert not hits, (
        "warning classes defined in test modules (xdist's controller cannot re-import "
        "them; move each to app/ or tests/support/):\n  " + "\n  ".join(hits)
    )


def test_the_scan_sees_a_warning_subclass(tmp_path):
    """Mutation control: a planted subclass IS found, so an empty result means
    absence, not a blind scanner."""
    (tmp_path / "test_planted.py").write_text(
        "import warnings\nclass Planted(UserWarning):\n    pass\n"
        "class Deeper(warnings.DeprecationWarning):\n    pass\nclass TestNotAWarning:\n    pass\n"
    )
    hits = warning_classes_in_test_modules([tmp_path])
    assert [h.split()[-1] for h in hits] == ["Planted", "Deeper"], hits


@pytest.mark.parametrize("root", ROOTS, ids=lambda p: p.name)
def test_every_root_has_test_modules(root):
    assert any(root.rglob("test_*.py")), f"{root} has no test modules — the scan would be vacuous"
