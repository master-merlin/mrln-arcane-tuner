"""LANE-143 layer guard: ``ux-v2.css`` is one scoped, unlayered stylesheet.

Every rule hangs under ``html[data-ux="v2"]`` so with the switch OFF the file
contributes nothing, and it may not move, resize or hide an existing element.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CSS_PATH = REPO_ROOT / "frontend" / "src" / "styles" / "ux-v2.css"
ANGULAR_JSON = REPO_ROOT / "frontend" / "angular.json"
SCOPE = 'html[data-ux="v2"]'

# Properties that would hide, move or resize an element -- shorthands AND their
# longhands / logical forms (VERIFY 1.02: a shorthand-only matcher passed
# `.chip { margin-left: 100px }`). The allow-list is exactly what the spec names:
# any of them on the new `.kpi-icon` element, and `.tag { padding: 0 5px }`, the
# one box-neutral declaration the overlay measured (1px border + 1px less padding).
BOX_PROP = re.compile(
    r"(display|position|float|inset(-[\w-]+)?|top|right|bottom|left"
    r"|(min-|max-)?(width|height|inline-size|block-size)"
    r"|margin(-[\w-]+)?|padding(-[\w-]+)?|flex|flex-basis|flex-grow|flex-shrink"
    r"|transform|translate|scale|rotate|font|font-size|line-height|gap|row-gap|column-gap"
    r"|grid(-[\w-]+)?|visibility|zoom|aspect-ratio)"
)
BOX_ALLOW_ELEMENT = ".kpi-icon"
BOX_ALLOW_DECLS = {
    (".tag", "padding", "0 5px"),
    # the user's two recorded overrides of 2026-10-07 (exact declarations, nothing else):
    # the HPS help badge sits inline after the eyebrow text (static position, no height)...
    (".ds-hps-info", "top", "auto"),
    (".ds-hps-info", "right", "auto"),
    (".ds-hps-info", "margin-left", "6px"),
    (".ds-hps-info", "margin-top", "-2px"),
    # ...and the library card's "N suppressed" chip moves to the card's top-right corner.
    (".ds-card-excluded", "position", "absolute"),
    (".ds-card-excluded", "top", "8px"),
    (".ds-card-excluded", "right", "8px"),
    (".ds-card-excluded", "top", "38px"),
    # the user's UAT round 3 answer of 2026-10-08 (`UAT-LANE-143.3 no: ew-grid-padding=12px`):
    (".ew-grid", "padding", "12px"),
}


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


def box_violations(css: str) -> list[str]:
    """``selector { prop: value }`` for every box-affecting declaration outside
    the allow-list; a custom property (``--x``) never matches."""
    bad: list[str] = []
    for head, body in parse_rules(css):
        if head.startswith("@"):
            continue
        for decl in (d.strip() for d in body.split(";")):
            prop, _, value = decl.partition(":")
            prop, value = prop.strip().lower(), " ".join(value.split())
            if not BOX_PROP.fullmatch(prop):
                continue
            for sel in split_selectors(head):
                subject = sel.split()[-1] if sel.split() else sel
                if subject == BOX_ALLOW_ELEMENT:
                    continue
                if (subject, prop, value) in BOX_ALLOW_DECLS:
                    continue
                bad.append(f"{sel} {{ {prop}: {value} }}")
    return bad


def test_no_box_properties_on_existing_elements() -> None:
    bad = box_violations(_read())
    assert not bad, f"LANE-143: box property on an element the overlay never sized: {bad}"


@pytest.mark.parametrize(
    ("rule", "named"),
    [
        (f"{SCOPE} .tag {{ display: none }}", f"{SCOPE} .tag {{ display: none }}"),
        (f"{SCOPE} .tag {{ width: 500px }}", f"{SCOPE} .tag {{ width: 500px }}"),
        (f"{SCOPE} .chip {{ margin-left: 100px }}", f"{SCOPE} .chip {{ margin-left: 100px }}"),
        (f"{SCOPE} .tag {{ padding: 0 6px }}", f"{SCOPE} .tag {{ padding: 0 6px }}"),
        (f"{SCOPE} .tag {{ padding-left: 5px }}", f"{SCOPE} .tag {{ padding-left: 5px }}"),
        (f"{SCOPE} .kpi {{ min-height: 0 }}", f"{SCOPE} .kpi {{ min-height: 0 }}"),
        (f"{SCOPE} .card {{ transform: translateY(2px) }}", f"{SCOPE} .card {{ transform: translateY(2px) }}"),
        (f"{SCOPE} .eyebrow {{ font-size: 12px; line-height: 2 }}", f"{SCOPE} .eyebrow {{ font-size: 12px }}"),
        (f"{SCOPE} .ds-hps-info {{ display: none }}", f"{SCOPE} .ds-hps-info {{ display: none }}"),
        (f"{SCOPE} .ds-card-excluded {{ margin-left: 9px }}", f"{SCOPE} .ds-card-excluded {{ margin-left: 9px }}"),
        (f"{SCOPE} .ew-grid {{ padding: 14px }}", f"{SCOPE} .ew-grid {{ padding: 14px }}"),
        (f"{SCOPE} .ew-grid {{ margin-top: 12px }}", f"{SCOPE} .ew-grid {{ margin-top: 12px }}"),
        (f"{SCOPE} .row {{ gap: 4px }}", f"{SCOPE} .row {{ gap: 4px }}"),
        (f"{SCOPE} .x {{ inset-inline-start: 0 }}", f"{SCOPE} .x {{ inset-inline-start: 0 }}"),
    ],
)
def test_the_box_check_rejects_hiding_moving_and_resizing(rule: str, named: str) -> None:
    """Negative controls (VERIFY 1.02's probes and their longhand siblings): each
    must be reported, naming the selector and the property."""
    assert named in box_violations(rule), f"LANE-143: the box guard let through {rule!r}: {box_violations(rule)}"


def test_the_box_check_allows_exactly_the_spec_list() -> None:
    allowed = f"{SCOPE} .tag {{ padding: 0 5px }}\n{SCOPE} .kpi .kpi-icon {{ position: absolute; width: 24px }}"
    assert box_violations(allowed) == [], "LANE-143: the spec's allow-list must pass"
    assert box_violations(f"{SCOPE} .chip {{ --pad: 4px; color: red; font-weight: 600 }}") == []


WASH_MIN = 12.0  # the user's correction 2 (2026-10-06): a visible tint, never below 12 %


def kpi_wash(css: str) -> dict[str, float]:
    """``--ux-kpi-wash`` per theme: the dark value from the bare scope, the
    light value from the light scope (falling back to the dark one)."""
    out: dict[str, float] = {}
    for head, body in parse_rules(css):
        m = re.search(r"--ux-kpi-wash\s*:\s*([0-9.]+)%", body)
        if not m:
            continue
        for sel in split_selectors(head):
            if sel == SCOPE:
                out["dark"] = float(m.group(1))
            elif sel == f'{SCOPE}[data-theme="light"]':
                out["light"] = float(m.group(1))
    if "dark" in out:
        out.setdefault("light", out["dark"])
    return out


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_kpi_wash_is_a_visible_tint(theme: str) -> None:
    wash = kpi_wash(_read())
    assert theme in wash, f"LANE-143: --ux-kpi-wash is not declared for the {theme} theme"
    assert wash[theme] >= WASH_MIN, (
        f"LANE-143: --ux-kpi-wash ({theme}) is {wash[theme]:g}% < {WASH_MIN:g}% (the user's correction 2)"
    )


def test_the_wash_pin_reads_each_theme() -> None:
    css = f'{SCOPE} {{ --ux-kpi-wash: 5%; }}\n{SCOPE}[data-theme="light"] {{ --ux-kpi-wash: 12%; }}'
    assert kpi_wash(css) == {"dark": 5.0, "light": 12.0}


def test_the_scope_check_rejects_an_unscoped_rule() -> None:
    """Positive control: a guard that cannot fail is not a guard."""
    assert scope_violations(f"{SCOPE} .a {{ color: red }}\n.chip {{ color: red }}") == [".chip"]
    assert scope_violations(f"{SCOPE} .a, .b {{ color: red }}") == [".b"]
    assert scope_violations("@media (x) { .chip { color: red } }") == [".chip"]
    # a comma inside :is() belongs to the scoped selector; a top-level one does not
    assert scope_violations(f"{SCOPE} :is(.a, .b), .c {{ color: red }}") == [".c"]
