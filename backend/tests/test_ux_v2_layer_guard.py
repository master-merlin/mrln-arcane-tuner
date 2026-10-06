"""LANE-143 layer guard: ``ux-v2.css`` is one scoped, unlayered stylesheet.

Every rule hangs under ``html[data-ux="v2"]`` so with the switch OFF the file
contributes nothing, and it may not move, resize or hide an existing element.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CSS_PATH = REPO_ROOT / "frontend" / "src" / "styles" / "ux-v2.css"
ANGULAR_JSON = REPO_ROOT / "frontend" / "angular.json"
SCOPE = 'html[data-ux="v2"]'

# Box properties that would move/resize an element; only the overlay-measured
# box-neutral selectors may declare them.
BOX_PROPS = re.compile(r"(?<![\w-])(display|position|width|height|margin|padding)\s*:")
BOX_ALLOW = (".kpi-icon", ".tag")


def _strip_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def parse_rules(css: str) -> list[tuple[str, str]]:
    """Flat (selector-list, body) pairs; at-rules are descended into."""
    css = _strip_comments(css)
    out: list[tuple[str, str]] = []
    i = 0
    n = len(css)
    while i < n:
        j = css.find("{", i)
        if j == -1:
            break
        head = css[i:j].strip()
        depth = 1
        k = j + 1
        while k < n and depth:
            depth += {"{": 1, "}": -1}.get(css[k], 0)
            k += 1
        body = css[j + 1 : k - 1]
        if head.startswith("@"):
            out.append((head, ""))
            out.extend(parse_rules(body))
        else:
            out.append((head, body))
        i = k
    return out


def split_selectors(head: str) -> list[str]:
    """Top-level selector list: a comma inside ``:is(...)`` does not split."""
    parts, depth, cur = [], 0, ""
    for ch in head:
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    return parts


def scope_violations(css: str) -> list[str]:
    bad: list[str] = []
    for head, _body in parse_rules(css):
        if head.startswith("@"):
            continue
        for sel in split_selectors(head):
            if sel and not sel.startswith(SCOPE):
                bad.append(sel)
    return bad


def _read() -> str:
    assert CSS_PATH.is_file(), f"LANE-143: {CSS_PATH.relative_to(REPO_ROOT)} missing"
    return CSS_PATH.read_text(encoding="utf-8")


def test_ux_v2_css_is_the_third_angular_styles_entry() -> None:
    _read()
    cfg = json.loads(ANGULAR_JSON.read_text(encoding="utf-8"))
    styles = cfg["projects"]["frontend"]["architect"]["build"]["options"]["styles"]
    assert styles[2:3] == ["src/styles/ux-v2.css"], (
        f"LANE-143: ux-v2.css must be the third angular.json styles entry, got {styles}"
    )


def test_every_selector_is_scoped_under_data_ux_v2() -> None:
    bad = scope_violations(_read())
    assert not bad, f"LANE-143: unscoped selector(s) in ux-v2.css: {bad}"


def test_no_layer_no_important() -> None:
    css = _strip_comments(_read())
    assert "@layer" not in css, "LANE-143: ux-v2.css must stay unlayered (@layer found)"
    assert "!important" not in css, "LANE-143: ux-v2.css must not use !important"


def test_no_color_literals_outside_token_declarations() -> None:
    lit = re.compile(r"oklch\(|#[0-9a-fA-F]{3,8}\b")
    bad: list[str] = []
    for head, body in parse_rules(_read()):
        for decl in (d.strip() for d in body.split(";")):
            if decl and lit.search(decl) and not re.match(r"--(role|ux)-[\w-]+\s*:", decl):
                bad.append(f"{head} {{ {decl} }}")
    assert not bad, f"LANE-143: colour literal outside a --role-*/--ux-* declaration: {bad}"


def test_no_box_properties_on_existing_elements() -> None:
    bad: list[str] = []
    for head, body in parse_rules(_read()):
        if head.startswith("@") or not BOX_PROPS.search(body):
            continue
        if not all(any(a in s for a in BOX_ALLOW) for s in split_selectors(head)):
            bad.append(head)
    assert not bad, f"LANE-143: box property on an element the overlay never sized: {bad}"


def test_the_scope_check_rejects_an_unscoped_rule() -> None:
    """Positive control: a guard that cannot fail is not a guard."""
    assert scope_violations(f"{SCOPE} .a {{ color: red }}\n.chip {{ color: red }}") == [".chip"]
    assert scope_violations(f"{SCOPE} .a, .b {{ color: red }}") == [".b"]
    assert scope_violations("@media (x) { .chip { color: red } }") == [".chip"]
    # a comma inside :is() belongs to the scoped selector; a top-level one does not
    assert scope_violations(f"{SCOPE} :is(.a, .b), .c {{ color: red }}") == [".c"]
