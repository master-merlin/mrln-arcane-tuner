"""No code sets a whitespace-split pip variable from a bare path.

pip splits its list-valued environment settings at whitespace, so a directory
whose path has a space reaches it as two locations, both ignored. A test serving
its wheels that way sees pip find nothing, and an install it expects to fail
fails for that reason instead (see ``tests/support/pip_env.py``).

The scan reads each file's syntax tree. A value is allowed when it is a call of
``find_links_value(...)``, or a string constant with no whitespace that is a URI
(``scheme://``) or holds no path separator (a host name). Anything else set into
one of these variables is reported: ``str(path)``, a name, an f-string, a joined
string. Reading and deleting are not setting, and are never reported.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
SCANNED = (BACKEND / "tests", BACKEND / "app")

# pip's settings that it splits at whitespace when read from the environment.
_LIST_VARS = frozenset({
    "PIP_FIND_LINKS", "PIP_EXTRA_INDEX_URL", "PIP_INDEX_URL", "PIP_CONSTRAINT",
    "PIP_BUILD_CONSTRAINT", "PIP_REQUIREMENT", "PIP_TRUSTED_HOST",
})
_URI = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_SETTERS = frozenset({"setenv", "putenv", "setdefault"})


def _is_list_var(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value in _LIST_VARS


def _allowed(value: ast.expr) -> bool:
    if isinstance(value, ast.Call):
        func = value.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        return name == "find_links_value"
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        text = value.value
        if any(c.isspace() for c in text):
            return False
        return bool(_URI.match(text)) or ("/" not in text and "\\" not in text)
    return False


def _set_values(tree: ast.AST):
    """(line, value) for every value set into one of the list variables."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Subscript) and _is_list_var(target.slice) and node.value is not None:
                    yield node.lineno, node.value
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if _is_list_var(key):
                    yield value.lineno, value
        elif isinstance(node, ast.Call):
            if (getattr(node.func, "attr", None) in _SETTERS and len(node.args) >= 2
                    and _is_list_var(node.args[0])):
                yield node.lineno, node.args[1]
            for kw in node.keywords:
                if kw.arg in _LIST_VARS:
                    yield kw.value.lineno, kw.value


def _bare_pip_list_assignments(source: str) -> list[str]:
    """``"<line>: <code>"`` for each line that sets a whitespace-split pip variable from
    anything but ``find_links_value(...)`` or a constant without a path."""
    lines = source.splitlines()
    found = sorted({lineno for lineno, value in _set_values(ast.parse(source)) if not _allowed(value)})
    return [f"{n}: {lines[n - 1].strip()}" for n in found]


def test_no_test_sets_a_whitespace_split_pip_variable_from_a_bare_path():
    offenders = []
    for root in SCANNED:
        for path in sorted(root.rglob("*.py")):
            source = path.read_text(encoding="utf-8-sig")  # ast.parse rejects a byte order mark
            if "PIP_" not in source:
                continue
            rel = path.relative_to(BACKEND).as_posix()
            offenders += [f"{rel}:{hit}" for hit in _bare_pip_list_assignments(source)]
    assert offenders == [], (
        "pip splits these variables at whitespace, so a path with a space reaches it as two "
        "ignored locations. Set the value through tests.support.pip_env.find_links_value "
        "(a file URI):\n" + "\n".join(offenders)
    )


def _reports(code: str) -> bool:
    return bool(_bare_pip_list_assignments(code))


def test_the_guard_reports_the_two_lines_this_fix_replaced():
    # The two lines the fix replaced, each reported.
    assert _reports('env["PIP_FIND_LINKS"] = str(sandbox["wheels"])\n')
    assert _reports('monkeypatch.setenv("PIP_FIND_LINKS", str(wheels))\n')
    # Other bare forms.
    assert _reports('env["PIP_FIND_LINKS"] = str(wheels)\n')
    assert _reports('monkeypatch.setenv("PIP_FIND_LINKS", str(p))\n')
    assert _reports('env = {"PIP_CONSTRAINT": f"{root}/c.txt"}\n')
    assert _reports('env["PIP_FIND_LINKS"] = "C:/w h"\n')
    assert _reports('env.update(PIP_EXTRA_INDEX_URL=wheels.as_posix())\n')
    # Allowed: the helper, a URI, a host name, and reading or deleting.
    assert not _reports('env["PIP_FIND_LINKS"] = find_links_value(wheels)\n')
    assert not _reports('monkeypatch.setenv("PIP_FIND_LINKS", find_links_value(wheels))\n')
    assert not _reports('env["PIP_INDEX_URL"] = "https://pypi.org/simple"\n')
    assert not _reports('env["PIP_TRUSTED_HOST"] = "pypi.org"\n')
    assert not _reports('monkeypatch.delenv("PIP_FIND_LINKS", raising=False)\n')
    assert not _reports('links = os.environ.get("PIP_FIND_LINKS")\n')


def test_the_guard_names_the_line_it_reports():
    source = 'import os\n\nenv = {}\nenv["PIP_FIND_LINKS"] = str(p)\n'
    assert _bare_pip_list_assignments(source) == ['4: env["PIP_FIND_LINKS"] = str(p)']
