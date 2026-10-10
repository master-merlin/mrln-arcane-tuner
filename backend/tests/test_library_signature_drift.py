"""Library-signature drift guard (LANE-154, RULE-20 class T).

A transformers/diffusers/peft pin once renamed a keyword (``create_causal_mask``
``input_embeds`` -> ``inputs_embeds``, ``cache_position`` removed) and two
families became untrainable while the suite stayed green: the tests stubbed the
seam with ``lambda **kw``.  This test walks ``backend/app`` (never tests) and
checks, against the INSTALLED libraries:

* every ``from transformers|diffusers|peft.<submodule> import X`` resolves;
* every call of an imported FUNCTION with explicit keywords uses only keywords
  that are parameters of the installed signature (``**kwargs`` functions skipped).

Out of reach, by design: ``from_pretrained(**kwargs)``, dynamic registries,
methods reached through instances, classes, and positional-argument drift.
"""
from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1] / "app"
LIBS = ("transformers", "diffusers", "peft")


def _is_lib_submodule(mod: str) -> bool:
    parts = mod.split(".")
    return parts[0] in LIBS and len(parts) > 1


def scan(root: Path) -> list[str]:
    """Return one finding line per unresolved import or stale keyword."""
    findings: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts or "tests" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        funcs: dict[str, object] = {}
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.level == 0
                and _is_lib_submodule(node.module)
            ):
                continue
            for a in node.names:
                try:
                    obj = getattr(importlib.import_module(node.module), a.name)
                except BaseException as exc:  # noqa: BLE001 - any import failure is a finding
                    findings.append(
                        f"{rel}:{node.lineno} -> cannot import "
                        f"{node.module}.{a.name}: {type(exc).__name__}"
                    )
                    continue
                if inspect.isfunction(obj) or inspect.isbuiltin(obj):
                    funcs[a.asname or a.name] = obj
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in funcs
            ):
                continue
            kws = [k.arg for k in node.keywords if k.arg is not None]
            if not kws:
                continue
            try:
                params = inspect.signature(funcs[node.func.id]).parameters
            except (TypeError, ValueError) as exc:
                # Failure is never silent: an unreadable signature is a finding.
                findings.append(
                    f"{rel}:{node.lineno} -> {node.func.id}: signature not "
                    f"inspectable ({type(exc).__name__}); verify its keywords by hand"
                )
                continue
            if any(p.kind is p.VAR_KEYWORD for p in params.values()):
                continue
            for kw in kws:
                if kw not in params:
                    findings.append(
                        f"{rel}:{node.lineno} -> {node.func.id}: keyword '{kw}' "
                        f"not in signature{tuple(params)}"
                    )
    return findings


def test_library_internal_imports_and_keyword_calls_match_the_installed_versions():
    findings = scan(APP_ROOT)
    assert not findings, "\n".join(findings)


def test_drift_guard_catches_a_stale_keyword(tmp_path):
    (tmp_path / "stale.py").write_text(
        "from transformers.masking_utils import create_causal_mask\n"
        "create_causal_mask(config=None, cache_position=1)\n",
        encoding="utf-8",
    )
    findings = scan(tmp_path)
    assert len(findings) == 1
    assert "cache_position" in findings[0]


def test_drift_guard_reports_an_uninspectable_signature(tmp_path, monkeypatch):
    (tmp_path / "opaque.py").write_text(
        "from transformers.masking_utils import create_causal_mask\n"
        "create_causal_mask(config=None, cache_position=1)\n",
        encoding="utf-8",
    )
    # Import first so only the guard's own signature read is affected.
    importlib.import_module("transformers.masking_utils")

    def _no_signature(*_a, **_k):
        raise ValueError("no signature found")

    monkeypatch.setattr(inspect, "signature", _no_signature)
    findings = scan(tmp_path)
    monkeypatch.undo()
    assert len(findings) == 1
    assert "signature not inspectable" in findings[0]
    assert "create_causal_mask" in findings[0]
