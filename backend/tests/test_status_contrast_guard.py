"""LANE-143: every status role clears its contrast duty in both themes, as RENDERED.

RULE-20 class **S** guard, the generalisation of ``test_danger_contrast_guard.py``
(DECISION-22) to every status role x theme x duty of the v2 design language in
``frontend/src/styles/ux-v2.css``:

* text: the recipe's ``color`` on the recipe's own background composited over the
  card surface >= 4.5:1, and on the bare surface >= 4.5:1;
* non-text: the recipe's border (an alpha / ``color-mix(..., transparent N%)``
  line composited over that background) and the dot fill >= 3:1 against the
  surface next to them.

It measures what the browser paints, not the raw tokens: each recipe's own
declarations are merged the way the cascade merges them (the base ``.chip`` rule,
then ``.chip.<role>``, custom properties included), every ``var()`` is resolved
against the theme's tokens (``styles.css`` + ``ux-v2.css``), ``color-mix()`` is
evaluated with the CSS Color 5 rules (premultiplied alpha, shorter hue arc, a
keyword colour's hue treated as missing), and alpha is composited before any
ratio. A guard that read the ``-line`` token instead of the composited border
would pass a pill whose border nobody can see -- the negative control below is
that recipe, and the guard must reject it.

The colour maths (OKLCH -> sRGB, luminance, ratio, source-over) has ONE
producer, ``test_danger_contrast_guard.py`` (RULE-21); this file imports it.

Positive controls: ``test_color_mix_matches_chromium`` pins the evaluator against
six values read from Chromium 153.0.8010.12 (Playwright, 2026-10-06,
``getComputedStyle`` of a probe ``div`` plus the canvas pixel of the computed
colour); ``test_the_overlay_light_danger_fg_is_rejected`` feeds back the
undarkened light danger text and requires a failure;
``test_the_border_guard_measures_the_composite`` is the synthetic recipe whose
raw line clears 3:1 but whose painted border does not.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from tests.test_danger_contrast_guard import (
    composite,
    contrast_ratio,
    oklch_to_srgb,
)
from tests.test_ux_v2_layer_guard import parse_rules, split_selectors

REPO_ROOT = Path(__file__).resolve().parents[2]
STYLES = REPO_ROOT / "frontend" / "src" / "styles.css"
UX_V2 = REPO_ROOT / "frontend" / "src" / "styles" / "ux-v2.css"

SCOPE = 'html[data-ux="v2"]'
LIGHT_SCOPE = 'html[data-ux="v2"][data-theme="light"]'

ROLES = ("success", "warning", "danger", "info", "neutral", "running", "queued", "paused")
THEMES = ("dark", "light")
# The lightest card surface a status primitive sits on, per theme (spec (b)).
SURFACE = {"dark": "--color-surface-low", "light": "--color-base"}
TEXT_MIN = 4.5
NON_TEXT_MIN = 3.0

RGB = tuple[int, int, int]


# ───────────────────────────── colour values ─────────────────────────────────
# A colour is (L, C, H | None, alpha) in OKLCH; H None = a missing hue.
Color = tuple[float, float, "float | None", float]

KEYWORDS: dict[str, Color] = {
    "white": (1.0, 0.0, None, 1.0),
    "black": (0.0, 0.0, None, 1.0),
    "transparent": (0.0, 0.0, None, 0.0),
}
_OKLCH = re.compile(
    r"^oklch\(\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*(?:/\s*([0-9.]+%?))?\s*\)$"
)


def _split_top(text: str, sep: str = ",") -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return parts


def _to_lab(c: Color) -> tuple[float, float, float, float]:
    lightness, chroma, hue, alpha = c
    h = math.radians(hue or 0.0)
    return lightness, chroma * math.cos(h), chroma * math.sin(h), alpha


def _from_lab(lightness: float, a: float, b: float, alpha: float) -> Color:
    chroma = math.hypot(a, b)
    hue = math.degrees(math.atan2(b, a)) % 360 if chroma > 1e-9 else None
    return lightness, chroma, hue, alpha


def mix(space: str, c1: Color, p1: float | None, c2: Color, p2: float | None) -> Color:
    """CSS Color 5 ``color-mix()`` in oklch / oklab: percentage normalisation,
    premultiplied alpha, shorter hue arc, missing hue taken from the other side."""
    if p1 is None and p2 is None:
        p1 = p2 = 0.5
    elif p1 is None:
        p1 = 1 - p2  # type: ignore[operator]
    elif p2 is None:
        p2 = 1 - p1
    total = p1 + p2  # type: ignore[operator]
    alpha_mult = min(total, 1.0)
    p1, p2 = p1 / total, p2 / total  # type: ignore[operator]
    a1, a2 = c1[3], c2[3]
    alpha = a1 * p1 + a2 * p2
    if space == "oklab":
        l1, x1, y1, _ = _to_lab(c1)
        l2, x2, y2, _ = _to_lab(c2)
        if alpha == 0:
            return (0.0, 0.0, None, 0.0)
        lab = [(u * a1 * p1 + v * a2 * p2) / alpha for u, v in ((l1, l2), (x1, x2), (y1, y2))]
        return _from_lab(lab[0], lab[1], lab[2], alpha * alpha_mult)
    assert space == "oklch", f"LANE-143: unsupported color-mix space {space!r}"
    if alpha == 0:
        return (0.0, 0.0, None, 0.0)
    lightness = (c1[0] * a1 * p1 + c2[0] * a2 * p2) / alpha
    chroma = (c1[1] * a1 * p1 + c2[1] * a2 * p2) / alpha
    h1, h2 = c1[2], c2[2]
    if h1 is None and h2 is None:
        hue = None
    elif h1 is None:
        hue = h2
    elif h2 is None:
        hue = h1
    else:
        delta = (h2 - h1) % 360
        if delta > 180:
            delta -= 360
        hue = (h1 + delta * p2) % 360
    return lightness, chroma, hue, alpha * alpha_mult


def _color_and_pct(text: str) -> tuple[str, float | None]:
    m = re.match(r"^(.*\S)\s+([0-9.]+)%$", text.strip())
    if m:
        return m.group(1), float(m.group(2)) / 100
    return text.strip(), None


def resolve_vars(value: str, env: dict[str, str], depth: int = 0) -> str:
    assert depth < 20, f"LANE-143: var() cycle while resolving {value!r}"
    m = re.search(r"var\(\s*(--[\w-]+)\s*", value)
    if not m:
        return value
    # the fallback may hold parentheses itself (`var(--x, oklch(0.6 0.18 30))`, a component rule)
    end = _matching_paren(value, m.start() + 3)
    name, rest = m.group(1), value[m.end():end].strip()
    fallback = rest[1:].strip() if rest.startswith(",") else None
    if name in env:
        sub = env[name]
    else:
        assert fallback is not None, f"LANE-143: token {name} is not defined"
        sub = fallback
    return resolve_vars(value[: m.start()] + sub + value[end + 1:], env, depth + 1)


def parse_color(text: str) -> Color:
    text = text.strip()
    if text in KEYWORDS:
        return KEYWORDS[text]
    m = _OKLCH.match(text)
    if m:
        alpha = 1.0
        if m.group(4):
            raw = m.group(4)
            alpha = float(raw[:-1]) / 100 if raw.endswith("%") else float(raw)
        return float(m.group(1)), float(m.group(2)), float(m.group(3)), alpha
    if text.startswith("color-mix(") and text.endswith(")"):
        args = _split_top(text[len("color-mix("):-1])
        assert len(args) == 3, f"LANE-143: malformed color-mix: {text!r}"
        space = args[0].replace("in", "", 1).strip()
        s1, p1 = _color_and_pct(args[1])
        s2, p2 = _color_and_pct(args[2])
        return mix(space, parse_color(s1), p1, parse_color(s2), p2)
    raise AssertionError(f"LANE-143: cannot evaluate colour {text!r}")


def evaluate(value: str, env: dict[str, str]) -> Color:
    return parse_color(resolve_vars(value, env))


def to_rgb(c: Color, backdrop: RGB) -> RGB:
    """What the browser paints: the colour composited over an opaque backdrop."""
    rgb = oklch_to_srgb(c[0], c[1], c[2] or 0.0)
    return rgb if c[3] >= 1.0 else composite(rgb, c[3], backdrop)


# ───────────────────────────── stylesheet reading ────────────────────────────

def _decls(body: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in _split_top(body, ";"):
        if ":" in part:
            prop, val = part.split(":", 1)
            out[prop.strip()] = val.strip()
    return out


def _block(css: str, head: str) -> str:
    m = re.search(re.escape(head) + r"\s*\{(.*?)\n\}", css, re.S)
    assert m, f"LANE-143: block {head!r} not found in styles.css"
    return re.sub(r"/\*.*?\*/", "", m.group(1), flags=re.S)


def _ux_rules() -> list[tuple[list[str], dict[str, str]]]:
    assert UX_V2.is_file(), "LANE-143: frontend/src/styles/ux-v2.css missing"
    rules = []
    for head, body in parse_rules(UX_V2.read_text(encoding="utf-8")):
        if head.startswith("@"):
            continue
        rules.append((split_selectors(head), _decls(body)))
    return rules


def theme_env(theme: str, rules=None) -> dict[str, str]:
    css = STYLES.read_text(encoding="utf-8")
    env = {k: v for k, v in _decls(_block(css, "@theme")).items() if k.startswith("--")}
    if theme == "light":
        env.update(_decls(_block(css, 'html[data-theme="light"]')))
    rules = _ux_rules() if rules is None else rules
    for heads, decls in rules:
        if SCOPE in heads or (theme == "light" and LIGHT_SCOPE in heads):
            env.update({k: v for k, v in decls.items() if k.startswith("--")})
    return env


def _compound_matches(comp: str, classes: frozenset[str], states: frozenset[str] = frozenset()) -> bool:
    """Does one compound selector match an element that carries ``classes``?

    The element model is class-only and enabled, at rest unless ``states``
    names the interaction state it is in (``hover``, ``active``): a type
    selector, any other pseudo-class, a pseudo-element or ``:has()`` never matches;
    ``:is()``/``:where()`` match on any argument, ``:not()`` on none;
    ``[class*="x"]`` on a class containing ``x``."""
    i, n = 0, len(comp)
    while i < n:
        ch = comp[i]
        if ch == ".":
            m = re.match(r"\.([\w-]+)", comp[i:])
            if m.group(1) not in classes:
                return False
            i += len(m.group(0))
        elif ch == ":":
            m = re.match(r"(::?)([\w-]+)(\()?", comp[i:])
            j = i + len(m.group(0))
            if not m.group(3):
                if m.group(1) == ":" and m.group(2) in states:
                    i = j
                    continue
                return False
            end = _matching_paren(comp, j - 1)
            args = _split_top(comp[j:end])
            name = m.group(2)
            if name in ("is", "matches", "where"):
                ok = any(_compound_matches(a, classes, states) for a in args)
            elif name == "not":
                ok = not any(_compound_matches(a, classes, states) for a in args)
            else:
                ok = False
            if not ok:
                return False
            i = end + 1
        elif ch == "[":
            end = comp.index("]", i)
            m = re.fullmatch(r'class\*="([^"]+)"', comp[i + 1 : end])
            if not m or not any(m.group(1) in c for c in classes):
                return False
            i = end + 1
        elif ch == "*":
            i += 1
        else:
            return False
    return True


def selector_matches(
    sel: str, theme: str, classes: frozenset[str], ancestors: tuple[frozenset[str], ...] = (),
    states: frozenset[str] = frozenset(),
) -> bool:
    """``sel`` (a full ux-v2.css selector) applies to an element with ``classes``
    whose ancestors (outermost first) carry ``ancestors``, in ``theme``."""
    if sel.startswith(LIGHT_SCOPE):
        if theme != "light":
            return False
        rest = sel[len(LIGHT_SCOPE):]
    elif sel.startswith(SCOPE):
        rest = sel[len(SCOPE):]
    else:
        return False
    if not rest[:1].isspace():
        return False  # a rule on <html> itself: tokens, not an element paint
    return plain_matches(rest, classes, ancestors, states)


def plain_matches(
    rest: str, classes: frozenset[str], ancestors: tuple[frozenset[str], ...] = (),
    states: frozenset[str] = frozenset(),
) -> bool:
    """An unscoped selector (a component rule, or a v2 rule past its scope)
    applies to the element model."""
    comps = compounds(rest)
    if any(c in "+~" for c in rest.replace(" > ", " ")) or not comps:
        return False
    if not _compound_matches(comps[-1], classes, states):
        return False
    k = 0
    for comp in comps[:-1]:
        while k < len(ancestors) and not _compound_matches(comp, ancestors[k]):
            k += 1
        if k == len(ancestors):
            return False
        k += 1
    return True


def cascade(theme: str, classes, ancestors=(), rules=None, states=frozenset()) -> tuple[dict[str, str], list[str]]:
    """The declarations the cascade gives an element, from ux-v2.css alone (the
    global sheets are layered and always lose to it): every matching rule in
    (specificity, source order), each property's last declaration winning,
    a shorthand resetting its longhand. Returns (declarations, winning selectors)."""
    rules = _ux_rules() if rules is None else rules
    classes, ancestors = frozenset(classes), tuple(frozenset(a) for a in ancestors)
    hits = []
    for idx, (heads, decls) in enumerate(rules):
        best = [specificity(s) for s in heads if selector_matches(s, theme, classes, ancestors, frozenset(states))]
        if best:
            hits.append((max(best), idx, heads, decls))
    merged: dict[str, str] = {}
    for _spec, _idx, _heads, decls in sorted(hits, key=lambda h: (h[0], h[1])):
        for prop, val in decls.items():
            if prop == "background":
                merged.pop("background-color", None)
            elif prop == "background-color":
                prop = "background"
            elif prop == "border":
                merged.pop("border-color", None)
            merged[prop] = val
    return merged, [s for h in sorted(hits, key=lambda h: (h[0], h[1])) for s in h[2]]


def recipe(theme: str, *selectors: str, rules=None) -> tuple[dict[str, str], list[str]]:
    """The painted declarations of the element the ``selectors`` describe, and
    which of those selectors exist as ux-v2.css rules (the wiring checks).

    The single-compound selectors name the element's classes (``.chip``,
    ``.chip.success`` -> an element ``chip success``); a descendant selector
    (``.chip .dot``) makes them the ancestor and its subject the element, which
    inherits the ancestor's custom properties. The declarations are the
    CASCADE's winners, not the named rules' (VERIFY round 2, MAJOR 1.01)."""
    rules = _ux_rules() if rules is None else rules
    own = frozenset(c for sel in selectors if len(compounds(sel)) == 1 for c in re.findall(r"\.([\w-]+)", sel))
    desc = [sel for sel in selectors if len(compounds(sel)) > 1]
    if desc:
        anc_decls, _ = cascade(theme, own, rules=rules)
        el = frozenset(re.findall(r"\.([\w-]+)", compounds(desc[-1])[-1]))
        decls, _ = cascade(theme, el, (own,), rules=rules)
        decls = {**{k: v for k, v in anc_decls.items() if k.startswith("--")}, **decls}
    else:
        decls, _ = cascade(theme, own, rules=rules)
    present = []
    for sel in selectors:
        wanted = [f"{SCOPE} {sel}"] + ([f"{LIGHT_SCOPE} {sel}"] if theme == "light" else [])
        if any(w in heads for heads, _d in rules for w in wanted):
            present.append(sel)
    return decls, present


def rule_body(sel: str, rules=None) -> str:
    rules = _ux_rules() if rules is None else rules
    for heads, decls in rules:
        if f"{SCOPE} {sel}" in heads:
            return "; ".join(f"{k}: {v}" for k, v in decls.items())
    raise AssertionError(f"LANE-143: no `{SCOPE} {sel}` rule in ux-v2.css")


def _border_value(decls: dict[str, str]) -> str:
    if "border-color" in decls:
        return decls["border-color"]
    m = re.match(r"^\S+\s+solid\s+(.+)$", decls.get("border", ""))
    assert m, f"LANE-143: recipe declares no border colour: {decls}"
    return m.group(1)


class Painted:
    """A recipe as the browser paints it on a theme's card surface."""

    def __init__(self, decls: dict[str, str], theme: str, rules=None) -> None:
        self.env = {**theme_env(theme, rules), **{k: v for k, v in decls.items() if k.startswith("--")}}
        self.decls = decls
        self.surface = to_rgb(evaluate(f"var({SURFACE[theme]})", self.env), (0, 0, 0))
        bg = decls.get("background", decls.get("background-color"))
        self.bg = to_rgb(evaluate(bg, self.env), self.surface) if bg else self.surface

    def fg(self) -> RGB:
        return to_rgb(evaluate(self.decls["color"], self.env), self.bg)

    def text_on_bg(self) -> float:
        return contrast_ratio(self.fg(), self.bg)

    def text_on_surface(self) -> float:
        fg = to_rgb(evaluate(self.decls["color"], self.env), self.surface)
        return contrast_ratio(fg, self.surface)

    def border(self) -> float:
        """The painted border: the line composited over the recipe's background,
        measured against the surface next to it."""
        line = to_rgb(evaluate(_border_value(self.decls), self.env), self.bg)
        return contrast_ratio(line, self.surface)

    def raw_line(self) -> float:
        return contrast_ratio(to_rgb(evaluate("var(--p-line)", self.env), self.surface), self.surface)


def _chip(theme: str, role: str, rules=None) -> Painted:
    decls, hit = recipe(theme, ".chip", f".chip.{role}", rules=rules)
    assert f".chip.{role}" in hit, f"LANE-143: no .chip.{role} recipe in ux-v2.css"
    return Painted(decls, theme, rules)


# ──────────────────────────────── the guard ─────────────────────────────────

@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("role", ROLES)
def test_chip_recipe_is_wired_to_its_role_token(role: str, theme: str) -> None:
    for sel in (f".chip.{role}", f".sdot.{role}"):
        assert f"--st-{role}-" in rule_body(sel), (
            f"LANE-143: {sel} no longer names --st-{role}-* -- the guard would measure the wrong colour"
        )


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("role", ROLES)
def test_chip_text_on_its_own_background(role: str, theme: str) -> None:
    ratio = _chip(theme, role).text_on_bg()
    assert ratio >= TEXT_MIN, (
        f"LANE-143: .chip.{role} text on its composited background ({theme}) "
        f"measures {ratio:.2f}:1 < {TEXT_MIN}:1"
    )


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("role", ROLES)
def test_role_fg_on_the_bare_surface(role: str, theme: str) -> None:
    ratio = _chip(theme, role).text_on_surface()
    note = " (DECISION-22)" if role == "danger" else ""
    assert ratio >= TEXT_MIN, (
        f"LANE-143: {role} -fg on {SURFACE[theme]} ({theme}, text) "
        f"measures {ratio:.2f}:1 < {TEXT_MIN}:1{note}"
    )


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("role", ROLES)
def test_chip_border_composited(role: str, theme: str) -> None:
    ratio = _chip(theme, role).border()
    assert ratio >= NON_TEXT_MIN, (
        f"LANE-143: .chip.{role} border composited measures {ratio:.1f}:1 < 3:1 ({theme})"
    )


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("role", ROLES)
def test_dot_fill(role: str, theme: str) -> None:
    decls, hit = recipe(theme, ".sdot", f".sdot.{role}")
    assert f".sdot.{role}" in hit, f"LANE-143: no .sdot.{role} recipe in ux-v2.css"
    sdot = Painted(decls, theme)  # background = the dot fill, surface behind it
    ratio = contrast_ratio(sdot.bg, sdot.surface)
    assert ratio >= NON_TEXT_MIN, (
        f"LANE-143: .sdot.{role} fill on {SURFACE[theme]} ({theme}) measures {ratio:.2f}:1 < 3:1"
    )
    chip = _chip(theme, role)
    dot_decls, dot_hit = recipe(theme, ".chip", f".chip.{role}", ".chip .dot")
    assert ".chip .dot" in dot_hit, "LANE-143: no `.chip .dot` recipe in ux-v2.css"
    dot = to_rgb(evaluate(dot_decls["background"], {**chip.env, **chip.decls}), chip.bg)
    ratio = contrast_ratio(dot, chip.bg)
    assert ratio >= NON_TEXT_MIN, (
        f"LANE-143: .chip.{role} .dot on its pill ({theme}) measures {ratio:.2f}:1 < 3:1"
    )


TAG_ROLES = ("success", "warning", "danger", "info")


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("role", (*TAG_ROLES, "neutral"))
def test_tag_recipe(role: str, theme: str) -> None:
    sels = (".tag",) if role == "neutral" else (".tag", f".tag.{role}")
    decls, hit = recipe(theme, *sels)
    assert hit[-1] == sels[-1], f"LANE-143: no {sels[-1]} recipe in ux-v2.css"
    assert f"--st-{role}-" in rule_body(sels[-1]), f"LANE-143: {sels[-1]} no longer names --st-{role}-*"
    tag = Painted(decls, theme)
    for duty, ratio, floor in (
        ("text on its background", tag.text_on_bg(), TEXT_MIN),
        ("border composited", tag.border(), NON_TEXT_MIN),
    ):
        assert ratio >= floor, (
            f"LANE-143: {sels[-1]} {duty} ({theme}) measures {ratio:.2f}:1 < {floor}:1"
        )


@pytest.mark.parametrize("util", ("success", "warning", "danger"))
def test_light_text_utilities_read_at_aa(util: str) -> None:
    decls, hit = recipe("light", f".text-{util}")
    assert hit, f"LANE-143: no light .text-{util} rule in ux-v2.css"
    assert f"--st-{util}-fg" in decls.get("color", ""), f"LANE-143: light .text-{util} must use --st-{util}-fg"
    ratio = Painted(decls, "light").text_on_surface()
    note = " (DECISION-22: the per-theme danger text token)" if util == "danger" else ""
    assert ratio >= TEXT_MIN, (
        f"LANE-143: .text-{util} on --color-base (light) measures {ratio:.2f}:1 < 4.5:1{note}"
    )


def test_light_brand_text_reads_at_aa() -> None:
    env = theme_env("light")
    base = to_rgb(evaluate("var(--color-base)", env), (0, 0, 0))
    for token in ("--color-brand-light",):
        ratio = contrast_ratio(to_rgb(evaluate(f"var({token})", env), base), base)
        assert ratio >= TEXT_MIN, (
            f"LANE-143: {token} (eyebrows, brand text) on --color-base (light) measures {ratio:.2f}:1 < 4.5:1"
        )


# ─────────────────────────────── positive controls ───────────────────────────

# (expression, Chromium's computed oklch(L C H [/ a]), canvas sRGB of it)
CHROMIUM_153 = [
    ("color-mix(in oklch, oklch(0.63 0.17 25), black 30%)", (0.441, 0.119, 25, 1.0), (136, 50, 47)),
    ("color-mix(in oklch, oklch(0.75 0.16 75), white 12%)", (0.779999, 0.140806, 75, 1.0), (235, 169, 64)),
    ("color-mix(in oklch, oklch(0.68 0.14 155), oklch(0.98 0.005 265) 88%)", (0.944, 0.0212, 251.8, 1.0), (227, 238, 251)),
    ("color-mix(in oklch, oklch(0.63 0.17 25), oklch(0.98 0.005 265) 88%)", (0.938, 0.0248, 279.4, 1.0), (231, 233, 251)),
    ("color-mix(in oklch, oklch(0.70 0.13 240), oklch(0.14 0.01 265) 86%)", (0.2184, 0.0268, 261.5, 1.0), (19, 26, 39)),
    ("color-mix(in oklch, oklch(0.63 0.17 25), transparent 55%)", (0.63, 0.17, 25, 0.45), None),
]


@pytest.mark.parametrize(("expr", "browser", "rgb"), CHROMIUM_153, ids=[str(i) for i in range(len(CHROMIUM_153))])
def test_color_mix_matches_chromium(expr: str, browser: tuple, rgb: RGB | None) -> None:
    got = parse_color(expr)
    assert got[2] is not None
    for name, ours, theirs, tol in zip(("L", "C", "H", "alpha"), got, browser, (6e-4, 6e-4, 0.06, 1e-6)):
        assert abs(ours - theirs) <= tol, (
            f"LANE-143: color-mix evaluator {name}={ours:.5f}, Chromium {theirs} for {expr}"
        )
    if rgb is not None:
        ours_rgb = oklch_to_srgb(got[0], got[1], got[2])
        assert all(abs(a - b) <= 1 for a, b in zip(ours_rgb, rgb)), (
            f"LANE-143: sRGB {ours_rgb} vs Chromium canvas {rgb} for {expr}"
        )


def test_the_overlay_light_danger_fg_is_rejected() -> None:
    """The pre-lane overlay never measured its light -fg mixes (README:69); the
    undarkened mix (``black 0%``) is today's 3.58:1 and must fail the bar."""
    env = theme_env("light")
    base = to_rgb(evaluate("var(--color-base)", env), (0, 0, 0))
    fg = to_rgb(evaluate("color-mix(in oklch, var(--role-danger), black 0%)", env), base)
    ratio = contrast_ratio(fg, base)
    assert ratio < TEXT_MIN, f"LANE-143: the undarkened light danger text must be rejected, got {ratio:.2f}:1"
    assert 3.5 < ratio < 3.65, f"LANE-143: expected ~3.58:1 (styles.css DECISION-22 note), got {ratio:.2f}:1"


def test_the_border_guard_measures_the_composite() -> None:
    """Negative control: the raw line clears 3:1, the painted border does not.
    A guard that measured the token would pass this recipe and is dead."""
    synthetic = {
        "--p-line": "oklch(0.62 0.15 25)",
        "background": "color-mix(in oklab, var(--p-line), var(--color-base) 88%)",
        "color": "color-mix(in oklch, var(--p-line), black 35%)",
        "border-color": "color-mix(in oklch, var(--p-line), transparent 55%)",
    }
    painted = Painted(synthetic, "light")
    raw, drawn = painted.raw_line(), painted.border()
    assert raw >= NON_TEXT_MIN, f"LANE-143: control setup: raw line must clear 3:1, got {raw:.2f}:1"
    assert drawn < NON_TEXT_MIN, (
        f"LANE-143: the border guard must measure the composite (raw {raw:.2f}:1, painted {drawn:.2f}:1)"
    )


# ───────────────────── the cascade: a recipe that loses is not painted ─────────
#
# VERIFY 1.01: the rows above measured the ``.sdot`` recipe, but the Jobs
# screen's encapsulated ``.sdot.success`` rule (``jobs-screen.css``, scoped by
# Angular to ``.sdot.success[_ngcontent-x]`` = 0,3,0) out-ranked the v2 rule that
# declared the dot background (``html[data-ux="v2"] .sdot`` = 0,2,1), so the light
# dot painted 2.56:1. Global ``styles.css``/``components.css`` live in ``@layer``
# and always lose to the unlayered ``ux-v2.css``; component styles are unlayered
# AND injected after it, so a v2 rule wins only on STRICTLY higher specificity.
# This guard enumerates every component stylesheet (``*.css`` and inline
# ``styles``) and fails on a rule that redeclares a COLOUR property of an element
# a v2 recipe claims, with a specificity >= the most specific v2 rule that paints
# that property there.
#
# "Claims": ux-v2.css has a rule whose subject names exactly the component rule's
# subject classes (``.sdot.success`` claims ``.sdot.success``), and whose ancestor
# classes, if any, the component selector names too (``.chip .dot`` does not
# claim ``.kpi-status .dot``). A component variant v2 has no recipe for
# (``.chip.violet``) is the component's own and is not judged. Interaction
# states (``:hover``/``:focus``/``:active``) are the component's until L3 adds
# them to the recipes.

APP_DIR = REPO_ROOT / "frontend" / "src" / "app"
# property -> the colour family the v2 recipes own (a shorthand overrides its longhand)
OWNED = {
    "color": "color",
    "background": "background",
    "background-color": "background",
    "border": "border-color",
    "border-color": "border-color",
    "box-shadow": "box-shadow",
    "outline": "outline",
    "outline-color": "outline",
    # the label-caps recipe (VERIFY round 1 residual, round 2 scope D): weight and
    # tracking are paint too -- five workspace `.eyebrow` rules and
    # `.ca-model .card-title` out-ranked the v2 label weight
    "font": "font-weight",
    "font-weight": "font-weight",
    "letter-spacing": "letter-spacing",
}
_STATE = re.compile(r":(hover|focus|focus-visible|focus-within|active)\b")
Spec = tuple[int, int, int]
_FUNC = re.compile(r"::?([\w-]+)\(")
_NOT_OWN_BOX = re.compile(
    r"::?(before|after|placeholder|selection|first-line|first-letter|-webkit-[\w-]+|-moz-[\w-]+)\b"
)


def _matching_paren(s: str, i: int) -> int:
    depth = 0
    for k in range(i, len(s)):
        depth += {"(": 1, ")": -1}.get(s[k], 0)
        if depth == 0:
            return k
    return len(s) - 1


def specificity(sel: str) -> Spec:
    """Selectors Level 4 specificity: ``:is/:not/:has`` take their most specific
    argument, ``:where`` counts 0, any other pseudo-class 0,1,0."""
    a = b = c = 0
    i, n = 0, len(sel)
    start = True  # at the start of a compound: a type selector may follow
    while i < n:
        ch = sel[i]
        if ch in " >+~":
            start = True
            i += 1
            continue
        if ch in "#.":
            m = re.match(r"[#.][\w-]+", sel[i:])
            a, b = (a + 1, b) if ch == "#" else (a, b + 1)
            i += len(m.group(0))
        elif ch == "[":
            b += 1
            i = sel.index("]", i) + 1
        elif ch == ":":
            m = re.match(r"(::?)([\w-]+)(\()?", sel[i:])
            colons, name, paren = m.groups()
            j = i + len(m.group(0))
            if paren:
                end = _matching_paren(sel, j - 1)
                if name in ("is", "not", "has", "matches"):
                    sa = max(specificity(x) for x in _split_top(sel[j:end]))
                    a, b, c = a + sa[0], b + sa[1], c + sa[2]
                elif name != "where":
                    b += 1
                j = end + 1
            elif colons == "::" or name in ("before", "after", "first-line", "first-letter"):
                c += 1
            else:
                b += 1
            i = j
        elif ch == "*":
            i += 1
        else:
            m = re.match(r"[\w-]+", sel[i:])
            c += 1 if (m and start) else 0
            i += len(m.group(0)) if m else 1
        start = False
    return (a, b, c)


def compounds(sel: str) -> list[str]:
    """A complex selector split at its top-level combinators."""
    parts, depth, cur = [], 0, ""
    for ch in sel.strip():
        depth += {"(": 1, "[": 1, ")": -1, "]": -1}.get(ch, 0)
        if depth == 0 and ch in " >+~":
            if cur:
                parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur:
        parts.append(cur)
    return parts


def class_sets(sel: str) -> list[frozenset[str]]:
    """The class sets the subject (last compound) of ``sel`` requires; a
    ``:is(a, b)`` yields one alternative per argument, ``:not()`` adds none."""
    comps = compounds(sel)
    if not comps:
        return []
    comp, own, alts, i = comps[-1], "", [], 0
    while i < len(comp):
        m = _FUNC.match(comp, i)
        if m:
            end = _matching_paren(comp, m.end() - 1)
            if m.group(1) in ("is", "matches"):
                alts.extend(_split_top(comp[m.end() : end]))
            i = end + 1
        else:
            own += comp[i]
            i += 1
    base = frozenset(re.findall(r"\.([\w-]+)", own))
    if not alts:
        return [base]
    return [base | cs for alt in alts for cs in class_sets(alt)]


def component_sheets() -> list[tuple[str, str]]:
    """Every encapsulated stylesheet: ``*.css`` under ``src/app`` and the inline
    ``styles`` of every component."""
    sheets = [(p.relative_to(REPO_ROOT).as_posix(), p.read_text(encoding="utf-8")) for p in sorted(APP_DIR.rglob("*.css"))]
    inline = re.compile(r"styles\s*:\s*\[?\s*`(.*?)`", re.S)
    for p in sorted(APP_DIR.rglob("*.ts")):
        if not p.name.endswith(".spec.ts"):
            sheets += [(p.relative_to(REPO_ROOT).as_posix(), m.group(1)) for m in inline.finditer(p.read_text(encoding="utf-8"))]
    return sheets


def encapsulated(sel: str) -> tuple[str, Spec] | None:
    """What Angular's emulated encapsulation makes of ``sel``: every compound
    before ``::ng-deep`` gains one ``[_ngcontent-x]``. ``None`` for a rule that
    paints a pseudo-element or the host, not the element a v2 recipe styles."""
    if _NOT_OWN_BOX.search(sel.replace("::ng-deep", "")):
        return None
    head, deep, tail = sel.partition("::ng-deep")
    plain = f"{head} {tail}" if deep else sel
    comps = compounds(plain)
    if not comps or ":host" in comps[-1]:
        return None
    a, b, c = specificity(re.sub(r":host(-context)?", ":h", plain))
    return " ".join(comps), (a, b + len(compounds(head)), c)


def _ancestor_classes(sel: str) -> frozenset[str]:
    return frozenset(c for comp in compounds(sel)[:-1] for c in re.findall(r"\.([\w-]+)", comp))


def v2_rules(theme: str, rules=None) -> list[tuple[str, frozenset[str], frozenset[str], Spec, set[str]]]:
    """(selector, subject classes, ancestor classes, specificity, colour families)
    of every v2 rule that applies in ``theme``, custom-property-only rules included
    (they still claim their subject)."""
    out = []
    for heads, decls in _ux_rules() if rules is None else rules:
        fams = {OWNED[p] for p in decls if p in OWNED}
        for sel in heads:
            if LIGHT_SCOPE in sel and theme != "light":
                continue
            anc, spec = _ancestor_classes(sel), specificity(sel)
            out += [(sel, cs, anc, spec, fams, decls) for cs in class_sets(sel) if cs]
    return out


TYPE_FAMS = ("font-weight", "letter-spacing")


def cascade_losses(theme: str, sheets: list[tuple[str, str]], rules=None) -> list[str]:
    v2 = v2_rules(theme, rules)
    bad = []
    for path, css in sheets:
        for head, body in parse_rules(css):
            comp_decls = _decls(body)
            fams = {OWNED[p] for p in comp_decls if p in OWNED}
            if comp_decls.get("text-transform") == "none":
                # a label that opts OUT of the caps look (`.ws-hps .eyebrow`, `.ca-model
                # .card-title`: sentence case, tracking 0) is the component's own
                # variant, not a label-caps recipe the v2 weight/tracking must win on
                fams -= set(TYPE_FAMS)
            for sel in split_selectors(head) if fams and not head.startswith("@") else ():
                enc = encapsulated(sel)
                if not enc or _STATE.search(compounds(enc[0])[-1]):
                    continue
                plain, comp_spec = enc
                anc = _ancestor_classes(plain) | frozenset()
                for subject in class_sets(plain):
                    here = [r for r in v2 if r[1] <= subject and r[2] <= anc | subject]
                    if not any(r[1] == subject for r in here):
                        continue  # a variant v2 has no recipe for: the component's own
                    for fam in sorted(fams):
                        if "btn" in subject and not any(r[1] == subject and fam in r[4] for r in here):
                            # a component button v2 restyles only in part (`.btn.cta.violet`:
                            # its fill, not its white ink or 800 weight): the families no exact
                            # v2 rule declares stay the component's, and the paint they make
                            # is MEASURED per state by component_button_failures below
                            continue
                        # the v2 side paints with its MOST specific matching rule
                        mine = [(spec, s, d) for s, _cs, _a, spec, f, d in here if fam in f]
                        if mine and comp_spec >= max(mine)[0]:
                            v2_spec, v2_sel, v2_decls = max(mine, key=lambda m: m[0])
                            if fam in TYPE_FAMS and comp_decls.get(fam) == v2_decls.get(fam):
                                continue  # the same weight/tracking: nothing visible is lost
                            bad.append(f"{path} `{sel}` {comp_spec} >= `{v2_sel}` {v2_spec} on {fam} ({theme})")
    return bad


@pytest.mark.parametrize("theme", THEMES)
def test_no_component_rule_out_ranks_a_v2_recipe(theme: str) -> None:
    bad = cascade_losses(theme, component_sheets())
    assert not bad, "LANE-143: a component rule out-ranks the v2 recipe (VERIFY 1.01):\n" + "\n".join(bad)


def test_the_cascade_guard_catches_the_jobs_dot() -> None:
    """Positive control: the pre-fix dot recipe (background declared only on the
    bare ``.sdot``) against the Jobs rule must be reported; the specificities are
    the browser's (0,3,0 encapsulated vs 0,2,1)."""
    assert specificity(f"{SCOPE} .sdot") == (0, 2, 1)
    assert specificity(f"{SCOPE} :is(.a, .b .c)") == (0, 3, 1)
    assert specificity(f"{SCOPE} :where(.a) .b:not(.c)") == (0, 3, 1)
    assert encapsulated(".sdot.success") == (".sdot.success", (0, 3, 0))
    assert encapsulated(".x ::ng-deep .sdot.success") == (".x .sdot.success", (0, 4, 0))
    assert encapsulated(".sdot.success::before") is None
    jobs = [("jobs-screen.css", ".sdot.success { background: var(--color-success); }")]
    pre_fix = [([f"{SCOPE} .sdot"], {"background": "var(--d)"}), ([f"{SCOPE} .sdot.success"], {"--d": "x"})]
    lost = cascade_losses("light", jobs, rules=pre_fix)
    assert lost and "on background" in lost[0], f"LANE-143: the cascade guard must report the Jobs dot, got {lost}"
    fixed = pre_fix + [([f"{SCOPE} .sdot:is(.success)"], {"background": "var(--d)"})]
    assert cascade_losses("light", jobs, rules=fixed) == [], (
        "LANE-143: a v2 role rule that out-ranks the component rule must clear it"
    )


# ─────────────── the winning rule: what the browser paints is the cascade's ───────────────
#
# VERIFY round 2, MAJOR 1.01: the rows above once read the rules by EXACT selector
# string (`.chip` then `.chip.<role>`), while the paint the browser applies comes
# from the role-specificity override `.chip:is(.neutral, .success, ...)`. A
# transparent text colour in that rule passed every row. The guard now resolves
# the cascade per element: every v2 rule whose selector MATCHES the element
# (``:is()`` expanded, ``:not()`` honoured), ordered by specificity then source
# order, each property's last declaration winning.


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("role", ("success", "danger"))
def test_a_transparent_winning_colour_fails_the_guard(role: str, theme: str) -> None:
    """Negative control: the override rule that WINS paints the text transparent;
    the guard must measure that rule and fail it."""
    rules = _ux_rules() + [([f"{SCOPE} .chip:is(.{role}, .neutral)"], {"color": "transparent"})]
    ratio = _chip(theme, role, rules).text_on_bg()
    assert ratio < TEXT_MIN, (
        f"LANE-143: the guard must measure the WINNING rule: a transparent .chip:is(.{role}) colour "
        f"still measured {ratio:.2f}:1 ({theme})"
    )


def test_the_cascade_guard_covers_label_weight_and_tracking() -> None:
    """Positive control (scope D): an encapsulated caps eyebrow that out-ranks the
    v2 label recipe on weight or tracking is reported; a sentence-case opt-out and
    an identical value are not; the doubled-class lift clears it."""
    v2 = [([f"{SCOPE} .eyebrow"], {"font-weight": "600", "letter-spacing": "0.08em"})]
    sheet = [("ws.css", ".ws-x .eyebrow { font-weight: 700; letter-spacing: 0.12em; }")]
    lost = cascade_losses("dark", sheet, rules=v2)
    assert any("on font-weight" in x for x in lost) and any("on letter-spacing" in x for x in lost), (
        f"LANE-143: the cascade guard must report weight and tracking losses, got {lost}"
    )
    assert cascade_losses("dark", [("ws.css", ".ws-x .eyebrow { text-transform: none; font-weight: 400; }")], rules=v2) == []
    assert cascade_losses("dark", [("ws.css", ".ws-x .eyebrow { font-weight: 600; }")], rules=v2) == []
    lifted = v2 + [([f"{SCOPE} .ws-x .eyebrow.eyebrow"], {"font-weight": "600", "letter-spacing": "0.08em"})]
    assert cascade_losses("dark", sheet, rules=lifted) == []


# ───────────────────────── L3 · buttons, ink and the focus ring ─────────────────────────

def _btn(theme: str, classes: tuple[str, ...], state: str | None = None, rules=None) -> Painted:
    """A button recipe as painted: the cascade's winners for the classes, then the
    declared state rule (hover / pressed) on top, composited over the card surface."""
    sels = (".btn", *(f".btn.{c}" for c in classes))
    decls, _hit = recipe(theme, *sels, rules=rules)
    if state:
        head = ".btn" + "".join(f".{c}" for c in classes) + f":not(:disabled):{state}"
        decls = {**decls, **_decls(rule_body(head, rules))}
    return Painted(decls, theme, rules)


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("state", (None, "hover", "active"))
def test_primary_ink_on_the_brand_fill(theme: str, state: str | None) -> None:
    ratio = _btn(theme, ("primary",), state).text_on_bg()
    assert ratio >= TEXT_MIN, (
        f"LANE-143: .btn.primary ink on --color-brand ({theme}, {state or 'rest'}) measures {ratio:.2f}:1 < {TEXT_MIN}:1"
    )


@pytest.mark.parametrize("theme", THEMES)
def test_primary_ink_white_is_rejected(theme: str) -> None:
    """Negative control (the row's mutant): white ink on the brand fill must fail."""
    rules = _ux_rules() + [([f"{SCOPE} .btn.primary"], {"color": "white"})]
    ratio = _btn(theme, ("primary",), rules=rules).text_on_bg()
    assert ratio < TEXT_MIN, f"LANE-143: white ink on --color-brand ({theme}) must fail the guard, measured {ratio:.2f}:1"


@pytest.mark.parametrize("theme", THEMES)
def test_primary_carries_no_brand_glow(theme: str) -> None:
    shadow = _btn(theme, ("primary",)).decls.get("box-shadow", "")
    assert "--color-brand" not in shadow, f"LANE-143: .btn.primary box-shadow must carry no brand glow ({theme}): {shadow}"


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize(("classes", "state"), [
    ((), None), ((), "hover"), ((), "active"),
    (("ghost",), None), (("ghost",), "hover"),
    (("danger-out",), None), (("danger-out",), "hover"), (("danger-out",), "active"),
    (("success",), None),
])
def test_button_text_on_its_own_fill(theme: str, classes: tuple[str, ...], state: str | None) -> None:
    ratio = _btn(theme, classes, state).text_on_bg()
    name = ".btn" + "".join(f".{c}" for c in classes)
    assert ratio >= TEXT_MIN, f"LANE-143: {name} text on its fill ({theme}, {state or 'rest'}) measures {ratio:.2f}:1 < {TEXT_MIN}:1"


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize(("cls", "state"), [("danger-out", None), ("danger-out", "hover"), ("success", None)])
def test_button_border_composited(theme: str, cls: str, state: str | None) -> None:
    """The painted border (the line, faded by its own mix, composited over the
    button's fill and measured against the surface next to it) clears 3:1."""
    ratio = _btn(theme, (cls,), state).border()
    assert ratio >= NON_TEXT_MIN, (
        f"LANE-143: .btn.{cls} composited border ({theme}, {state or 'rest'}) measures {ratio:.2f}:1 < {NON_TEXT_MIN}:1"
    )


@pytest.mark.parametrize(("cls", "theme"), [("danger-out", "dark"), ("danger-out", "light"), ("success", "light")])
def test_button_border_fade_is_rejected(theme: str, cls: str) -> None:
    """Negative control: the 40 % fade re-applied in memory must fail, naming the button and the ratio
    (measured 2.50 / 2.40 / 2.39:1 on the three; dark success clears 3:1 even faded, so it is no control)."""
    role = "danger" if cls == "danger-out" else "success"
    faded = f"color-mix(in oklch, var(--st-{role}-line), transparent 40%)"
    rules = _ux_rules() + [([f"{SCOPE} .btn.{cls}"], {"border-color": faded})]
    ratio = _btn(theme, (cls,), rules=rules).border()
    assert ratio < NON_TEXT_MIN, f"LANE-143: .btn.{cls} with the 40% fade must fail the border guard ({theme}), measured {ratio:.2f}:1"


FOCUS_RING_RULE = ":is(.btn, .icon-btn, .q-icon, .ctx-pill, .tc-pill, .side-item, button, a):focus-visible"


def focus_ring_ratio(theme: str, surface: str, rules=None) -> float:
    """The ring as the browser paints it: the WINNING outline declaration of the
    :focus-visible rule resolved (mix, alpha), composited over the surface."""
    rules = _ux_rules() if rules is None else rules
    decls: dict[str, str] = {}
    for heads, d in rules:
        if f"{SCOPE} {FOCUS_RING_RULE}" in heads:
            decls.update(d)
    assert decls, "LANE-143: no :focus-visible rule in ux-v2.css"
    if "outline-color" in decls:
        colour = decls["outline-color"]
    else:
        m = re.match(r"^\S+\s+\w+\s+(.+)$", decls.get("outline", ""))
        assert m, f"LANE-143: the focus-visible rule declares no outline colour: {decls}"
        colour = m.group(1)
    env = theme_env(theme, rules)
    back = to_rgb(evaluate(f"var({surface})", env), (0, 0, 0))
    return contrast_ratio(to_rgb(evaluate(colour, env), back), back)


@pytest.mark.parametrize("theme", THEMES)
def test_focus_ring_on_the_page(theme: str) -> None:
    for surface in ("--color-base", SURFACE[theme]):
        ratio = focus_ring_ratio(theme, surface)
        assert ratio >= NON_TEXT_MIN, f"LANE-143: the focus ring on {surface} ({theme}) measures {ratio:.2f}:1 < 3:1"


@pytest.mark.parametrize("theme", THEMES)
def test_a_transparent_focus_ring_is_rejected(theme: str) -> None:
    """Negative control: an outline of ``color-mix(in oklch, var(--ux-ring), transparent 99%)`` must fail."""
    ghost = "2px solid color-mix(in oklch, var(--ux-ring), transparent 99%)"
    rules = _ux_rules() + [([f"{SCOPE} {FOCUS_RING_RULE}"], {"outline": ghost})]
    ratio = focus_ring_ratio(theme, "--color-base", rules)
    assert ratio < NON_TEXT_MIN, f"LANE-143: a 99%-transparent focus ring must fail ({theme}), measured {ratio:.2f}:1"


# ──────────── the dataset workspace "Mass ..." buttons (user remark, round 3) ────────────
#
# `.ws-mass-mask` (success green) and `.ws-mass-edit` (violet) are coloured ON PURPOSE
# by the workspace component's own (0,2,0) rules; the v2 secondary `.btn` recipe
# out-ranks them, so under v2 they must carry a role treatment of their own, measured
# as rendered like every other button.

MASS_BUTTONS = {"ws-mass-mask": "success", "ws-mass-edit": "ux-violet"}
WORKSPACE_HTML = REPO_ROOT / "frontend" / "src" / "app" / "workspace" / "dataset-workspace.component.html"


def mass_button_problems(classes: tuple[str, ...], rules=None) -> list[str]:
    """The muted-button check: each mass button's painted colour, fill and border
    must name its own role tokens (a missing v2 rule paints the plain secondary)."""
    problems = []
    for cls in classes:
        token = MASS_BUTTONS[cls]
        for theme in THEMES:
            decls = _btn(theme, (cls,), rules=rules).decls
            for prop, suffix in (("color", "fg"), ("background", "bg")):
                if f"--st-{token}-{suffix}" not in decls.get(prop, "") and f"--{token}-{suffix}" not in decls.get(prop, ""):
                    problems.append(f"LANE-143: .btn.{cls} paints the muted secondary button ({theme}): {prop}={decls.get(prop)}")
    return problems


def workspace_mass_classes(html: str) -> list[str]:
    return sorted({c for attr in re.findall(r'class="([^"]*)"', html) for c in attr.split() if c.startswith("ws-mass-") and c != "ws-mass-caption"})


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("cls", sorted(MASS_BUTTONS))
@pytest.mark.parametrize("state", (None, "hover", "active"))
def test_mass_button_text_on_its_own_fill(theme: str, cls: str, state: str | None) -> None:
    painted = _btn(theme, (cls,), state)
    ratio = painted.text_on_bg()
    assert ratio >= TEXT_MIN, f"LANE-143: .btn.{cls} text on its fill ({theme}, {state or 'rest'}) measures {ratio:.2f}:1 < {TEXT_MIN}:1"


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("cls", sorted(MASS_BUTTONS))
def test_mass_button_text_on_the_bare_surface_and_border(theme: str, cls: str) -> None:
    painted = _btn(theme, (cls,))
    assert painted.text_on_surface() >= TEXT_MIN, f"LANE-143: .btn.{cls} text on the bare surface ({theme}) measures {painted.text_on_surface():.2f}:1 < {TEXT_MIN}:1"
    assert painted.border() >= NON_TEXT_MIN, f"LANE-143: .btn.{cls} composited border ({theme}) measures {painted.border():.2f}:1 < {NON_TEXT_MIN}:1"


def test_mass_buttons_are_not_muted_under_v2() -> None:
    assert mass_button_problems(tuple(MASS_BUTTONS)) == []


def test_removing_a_mass_button_rule_is_caught() -> None:
    """Negative control: with the v2 rules of the two buttons removed the guard names the muted buttons."""
    rules = [(h, d) for h, d in _ux_rules() if not any("ws-mass-" in s for s in h)]
    problems = mass_button_problems(tuple(MASS_BUTTONS), rules=rules)
    assert any("ws-mass-mask" in p for p in problems) and any("ws-mass-edit" in p for p in problems), problems


def test_every_workspace_mass_button_carries_a_v2_rule() -> None:
    classes = workspace_mass_classes(WORKSPACE_HTML.read_text(encoding="utf-8"))
    assert classes, "LANE-143: no .ws-mass-* button found in dataset-workspace.component.html"
    heads = [h for hs, _d in _ux_rules() for h in hs]
    for cls in classes:
        assert f"{SCOPE} .btn.{cls}" in heads, f"LANE-143: .btn.{cls} is coloured by the workspace but has no v2 rule (it paints muted under v2)"


def test_the_mass_scan_sees_a_planted_button() -> None:
    assert workspace_mass_classes('<button class="btn sm ws-mass-new"></button>') == ["ws-mass-new"]


# ─────── component-painted buttons: the WHOLE cascade per state (VERIFY r2, T4) ───────
#
# verify-lane-LANE-143-20261008T101628Z, MAJOR 1.01: the generic v2 `.btn:hover`
# (0,4,1) out-ranks a component CTA's own (0,3,0)/(0,4,0) fill but not its ink, so
# the mass-mask CTA hovered at 1.26:1 (dark) and the mass-caption CTA at 1.43:1
# (light). The cascade check above skips a component variant v2 has no recipe for,
# and the button rows read the v2 rules alone. Here every component rule that paints
# a `.btn` is the subject of an element model, and that element's colour and fill
# are resolved from BOTH sheets -- ux-v2.css and the component's own (encapsulated
# specificity; on a tie the component wins, its styles load after the global ones)
# -- at rest, hovered and pressed. Wherever v2 paints the colour or the fill, the
# text must read at 4.5:1 on the fill.

BUTTON_STATES: tuple[frozenset[str], ...] = (frozenset(), frozenset({"hover"}), frozenset({"hover", "active"}))
PAINT = ("color", "background")


def _state_name(states: frozenset[str]) -> str:
    return "pressed" if "active" in states else "hover" if "hover" in states else "rest"


def _norm(prop: str) -> str:
    return "background" if prop == "background-color" else prop


_TEMPLATE_INLINE = re.compile(r"template\s*:\s*`(.*?)`", re.S)
_TEMPLATE_URL = re.compile(r"templateUrl\s*:\s*['\"]([^'\"]+)['\"]")
_CLASS_ATTR = re.compile(r'(?<![\w.\[-])class="([^"]*)"')


def component_template(path: str) -> str:
    """The template(s) of the component that owns the sheet at ``path`` (repo-relative):
    ``x.css`` -> ``x.html`` and the inline/linked template of ``x.ts``; ``x.ts`` -> the same.
    Only the LAST suffix is replaced: ``x.component.css`` -> ``x.component.html`` (VERIFY
    20261008T225306Z 1.01: stripping ``.component`` too resolved nothing for every
    ``.component`` sheet). A sheet no template owns raises, never yields an empty scan."""
    sheet = REPO_ROOT / path
    out = []
    html = sheet.with_suffix(".html")
    if html.is_file():
        out.append(html.read_text(encoding="utf-8"))
    ts = sheet.with_suffix(".ts")
    if ts.is_file():
        src = ts.read_text(encoding="utf-8")
        out += _TEMPLATE_INLINE.findall(src)
        for url in _TEMPLATE_URL.findall(src):
            linked = (ts.parent / url).resolve()
            if linked.is_file() and linked != html.resolve():
                out.append(linked.read_text(encoding="utf-8"))
    if not out:
        raise LookupError(f"LANE-143: no component template owns {path} (looked for {html.name} and {ts.name})")
    return "\n".join(out)


def template_buttons(template: str) -> list[frozenset[str]]:
    """The static class sets of every element of ``template`` that carries ``btn``."""
    sets = {frozenset(attr.split()) for attr in _CLASS_ATTR.findall(template)}
    return sorted((s for s in sets if "btn" in s), key=sorted)


def component_button_models(sheets: list[tuple[str, str]]) -> list[tuple[str, str, str, frozenset[str], tuple[frozenset[str], ...]]]:
    """(path, sheet css, selector, element classes, ancestors) of every component rule
    whose subject is a `.btn` ELEMENT and that declares a colour or a fill, in any state.

    VERIFY 20261008T195032Z, MAJOR 1.01: a rule is judged by the element it styles, not
    by its selector text -- `.ds-bulk-danger` names no `.btn`, but the element it paints
    is `<button class="btn ds-bulk-danger">`; a subject without `btn` is matched against
    the static class sets of the component's own template, and every `.btn` element it
    styles becomes a model with that element's full class set."""
    out = []
    for path, css in sheets:
        buttons: list[frozenset[str]] | None = None
        for head, body in parse_rules(css):
            if head.startswith("@") or not any(_norm(p) in PAINT + ("border-color", "border") for p in _decls(body)):
                continue
            for sel in split_selectors(head):
                enc = encapsulated(sel)
                if not enc:
                    continue
                plain = _STATE.sub("", enc[0])
                anc = _ancestor_classes(plain)
                for subject in class_sets(plain):
                    if "btn" in subject:
                        out.append((path, css, sel, subject, (anc,) if anc else ()))
                    elif subject:
                        if buttons is None:
                            buttons = template_buttons(component_template(path))
                        out += [(path, css, sel, el, (anc,) if anc else ()) for el in buttons if subject <= el]
    return out


def painted_button(theme: str, css: str, classes: frozenset[str], ancestors, states: frozenset[str], rules=None, winners=None):
    """The (declarations, origin per paint property) the browser gives the element:
    every matching rule of ux-v2.css and of the component sheet, in (specificity,
    origin, source order); origin 0 = v2, 1 = the component. ``winners`` (a dict),
    when given, receives (origin, selector) of the rule that wins each property and
    ``("declared", ...)`` the properties any component rule declares."""
    rules = _ux_rules() if rules is None else rules
    hits = []
    for idx, (heads, decls) in enumerate(rules):
        best = [(specificity(s), s) for s in heads if selector_matches(s, theme, classes, ancestors, states)]
        if best:
            spec, sel = max(best)
            hits.append((spec, 0, idx, decls, sel))
    for idx, (head, body) in enumerate(parse_rules(css)):
        if head.startswith("@"):
            continue
        best = []
        for sel in split_selectors(head):
            enc = encapsulated(sel)
            if enc and plain_matches(enc[0], classes, ancestors, states):
                best.append((enc[1], sel))
        if best:
            spec, sel = max(best)
            hits.append((spec, 1, idx, _decls(body), sel))
    merged: dict[str, str] = {}
    origin: dict[str, int] = {}
    declared: dict[str, str] = {}
    for _spec, org, _idx, decls, sel in sorted(hits, key=lambda h: (h[0], h[1], h[2])):
        for prop, val in decls.items():
            if prop == "border":
                merged.pop("border-color", None)
            prop = _norm(prop)
            merged[prop] = val
            origin[prop] = org
            if winners is not None:
                key = "border-color" if prop == "border" else prop
                winners[key] = (org, sel)
                if org == 1:
                    declared[key] = val
                    if prop in ("border", "border-style"):
                        declared["border-style"] = val
    if winners is not None:
        winners["declared"] = declared
    return merged, origin


MUTE_PROPS = ("color", "background", "border-color")
# a component paint that carries MEANING: a status / brand / accent colour, not a surface,
# border or text token (a neutral `.btn.sm` taking the generic hover loses nothing)
_PURPOSE = re.compile(r"--color-(danger|success|warning|info|violet|brand)\b|--st-|--role-|oklch\(|#[0-9a-f]{3}|\bwhite\b", re.I)
_NO_BORDER = re.compile(r"^\s*(none|0(px)?)\b", re.I)


def component_button_mutes(theme: str, sheets=None, rules=None) -> list[str]:
    """A component button whose OWN purpose colour, fill or border (declared by its
    component rules, in any state) is won by a v2 rule that does not name any of the
    component's classes: a generic `.btn` recipe painting over it (the bulk Delete
    button turned neutral). A v2 rule written FOR the variant (it names one of the
    element's non-`btn` classes) is the deliberate recipe, judged by the rows above.
    A border the component switched off (`border: none`) paints no colour."""
    sheets = component_sheets() if sheets is None else sheets
    bad, seen = [], set()
    for path, css, sel, classes, ancestors in component_button_models(sheets):
        key = (path, classes, ancestors)
        if key in seen:
            continue
        seen.add(key)
        purpose = classes - {"btn"}
        for states in BUTTON_STATES:
            winners: dict = {}
            painted_button(theme, css, classes, ancestors, states, rules, winners)
            own = winners["declared"]
            for prop in MUTE_PROPS:
                if prop not in own or winners.get(prop, (1,))[0] != 0 or not _PURPOSE.search(own[prop]):
                    continue
                if prop == "border-color" and _NO_BORDER.match(own.get("border-style", "")):
                    continue
                v2_sel = winners[prop][1]
                named = {c for cs in class_sets(v2_sel[len(SCOPE):] if v2_sel.startswith(SCOPE) else v2_sel) for c in cs}
                if not named & purpose:
                    bad.append(
                        f"{path} .{'.'.join(sorted(classes))} ({theme}, {_state_name(states)}): its own {prop} "
                        f"is painted over by the generic `{v2_sel}`"
                    )
    return bad


def component_button_rows(theme: str, sheets: list[tuple[str, str]], rules=None) -> list[tuple[str, float, bool]]:
    """(label, text-on-fill ratio, v2 paints part of it) per component button x state."""
    rows, seen = [], set()
    for path, css, sel, classes, ancestors in component_button_models(sheets):
        key = (path, classes, ancestors)
        if key in seen:
            continue
        seen.add(key)
        for states in BUTTON_STATES:
            decls, origin = painted_button(theme, css, classes, ancestors, states, rules)
            if not all(p in decls for p in PAINT):
                continue
            ratio = Painted(decls, theme, rules).text_on_bg()
            v2_paints = any(origin.get(p) == 0 for p in PAINT)
            rows.append((f"{path} `{sel}` .{'.'.join(sorted(classes))} ({theme}, {_state_name(states)})", ratio, v2_paints))
    return rows


def component_button_failures(theme: str, sheets=None, rules=None) -> list[str]:
    sheets = component_sheets() if sheets is None else sheets
    return [
        f"{label} measures {ratio:.2f}:1 < {TEXT_MIN}:1"
        for label, ratio, v2_paints in component_button_rows(theme, sheets, rules)
        if v2_paints and ratio < TEXT_MIN
    ]


@pytest.mark.parametrize("theme", THEMES)
def test_component_painted_buttons_read_in_every_state(theme: str) -> None:
    bad = component_button_failures(theme)
    assert not bad, "LANE-143: a v2 rule paints part of a component button and its text no longer reads:\n" + "\n".join(bad)


def test_the_component_button_scan_measures_the_modal_ctas() -> None:
    """Wiring: the two CTAs of the finding are element models of the scan, in every state."""
    labels = [label for label, _r, _v in component_button_rows("dark", component_sheets())]
    for path, sel in (("mass-mask.component.ts", ".btn.cta.success"), ("mass-caption.component.ts", ".btn.cta")):
        for state in ("rest", "hover", "pressed"):
            assert any(path in lb and f"`{sel}`" in lb and f"(dark, {state})" in lb for lb in labels), (
                f"LANE-143: the component button scan does not measure {path} {sel} ({state})"
            )


@pytest.mark.parametrize(("theme", "path", "sel"), [
    ("dark", "mass-mask.component.ts", ".btn.cta.success"),
    ("light", "mass-caption.component.ts", ".btn.cta"),
])
def test_the_generic_hover_on_a_component_cta_is_caught(theme: str, path: str, sel: str) -> None:
    """Negative control (the finding): ux-v2.css without its `.cta` rules re-applies
    the generic hover to the component CTAs; the guard must name the CTA and its ratio."""
    rules = [(h, d) for h, d in _ux_rules() if not any(".cta" in s for s in h)]
    bad = component_button_failures(theme, rules=rules)
    hit = [b for b in bad if path in b and f"`{sel}`" in b and "hover" in b]
    assert hit, f"LANE-143: the generic v2 hover on {path} {sel} ({theme}) must fail the guard, got {bad}"


@pytest.mark.parametrize("theme", THEMES)
def test_no_component_button_is_muted_by_a_generic_v2_rule(theme: str) -> None:
    bad = component_button_mutes(theme)
    assert not bad, "LANE-143: a generic v2 button rule paints over a component button's own colour (VERIFY 1.01):\n" + "\n".join(bad)


BULK_DANGER_SHEET = "frontend/src/app/screens/datasets-screen/datasets-screen.css"
BULK_DANGER = frozenset({"btn", "ds-bulk-danger"})

# VERIFY 20261008T225306Z, MAJOR 1.01: the resolver stripped `.component` with the
# extension, so `x.component.css` looked for `x.html` and every one of the `.component`
# sheets resolved to NO template -- the element models it should feed were silently
# absent. These rows use REAL files of the tree, never a stubbed component_template.
WORKSPACE_SHEET = "frontend/src/app/workspace/dataset-workspace.component.css"
INLINE_SHEET = "frontend/src/app/modals/mass-mask/mass-mask.component.ts"


@pytest.mark.parametrize(("sheet", "needle"), [
    (WORKSPACE_SHEET, "ws-mass-mask"),
    (INLINE_SHEET, "btn"),
    (BULK_DANGER_SHEET, "ds-bulk-danger"),
])
def test_a_component_sheet_resolves_its_own_template(sheet: str, needle: str) -> None:
    assert (REPO_ROOT / sheet).is_file(), f"LANE-143: fixture sheet {sheet} moved"
    assert needle in component_template(sheet), f"LANE-143: component_template resolved no template for {sheet}"


def test_the_button_scan_models_the_workspace_mass_buttons() -> None:
    """The workspace `.ws-mass-*` rules name no `.btn`; their elements come from the
    `.component.html` template, so the models exist only if the resolver keeps `.component`."""
    models = {(p, c) for p, _css, _s, c, _a in component_button_models(component_sheets())}
    for cls in workspace_mass_classes(WORKSPACE_HTML.read_text(encoding="utf-8")):
        assert any(p == WORKSPACE_SHEET and cls in c for p, c in models), (
            f"LANE-143: the button scan does not model .btn.{cls} from {WORKSPACE_SHEET}"
        )


def test_an_unowned_sheet_is_reported_not_emptied(tmp_path) -> None:
    """Unresolved ownership is an error naming the sheet, never an empty template."""
    with pytest.raises(LookupError, match="LANE-143: no component template owns"):
        component_template("frontend/src/app/__no_such_dir__/orphan.component.css")


def test_the_button_scan_models_the_bulk_danger_element() -> None:
    """Wiring: `.ds-bulk-danger` names no `.btn`, yet the scan models its element
    `<button class="btn ds-bulk-danger">` from the datasets-screen template."""
    models = [(p, c) for p, _css, _s, c, _a in component_button_models(component_sheets())]
    assert (BULK_DANGER_SHEET, BULK_DANGER) in models, "LANE-143: the button scan does not model .btn.ds-bulk-danger"


def _bulk_danger(theme: str, states: frozenset[str], rules=None) -> Painted:
    css = dict(component_sheets())[BULK_DANGER_SHEET]
    decls, _origin = painted_button(theme, css, BULK_DANGER, (), states, rules)
    return Painted(decls, theme, rules)


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("states", BUTTON_STATES, ids=_state_name)
def test_bulk_danger_reads_as_danger(theme: str, states: frozenset[str]) -> None:
    """The bulk Delete / Remove button keeps its danger paint under v2: text on its fill
    and on the bare surface >= 4.5:1, the composited border >= 3:1, every colour from
    the danger role tokens."""
    p = _bulk_danger(theme, states)
    st = _state_name(states)
    for prop in ("color", "border-color"):
        assert "--st-danger-" in p.decls.get(prop, ""), (
            f"LANE-143: .btn.ds-bulk-danger {prop} ({theme}, {st}) is not the danger role: {p.decls.get(prop)}"
        )
    assert p.text_on_bg() >= TEXT_MIN, f"LANE-143: .btn.ds-bulk-danger text on its fill ({theme}, {st}) measures {p.text_on_bg():.2f}:1 < {TEXT_MIN}:1"
    assert p.text_on_surface() >= TEXT_MIN, f"LANE-143: .btn.ds-bulk-danger text on the surface ({theme}, {st}) measures {p.text_on_surface():.2f}:1 < {TEXT_MIN}:1"
    assert p.border() >= NON_TEXT_MIN, f"LANE-143: .btn.ds-bulk-danger composited border ({theme}, {st}) measures {p.border():.2f}:1 < {NON_TEXT_MIN}:1"


@pytest.mark.parametrize("theme", THEMES)
def test_removing_the_bulk_danger_rules_is_caught(theme: str) -> None:
    """Negative control (the finding): without its v2 rules the bulk danger button is
    painted by the generic secondary recipe, and the mute check names it."""
    rules = [(h, d) for h, d in _ux_rules() if not any("ds-bulk-danger" in s for s in h)]
    bad = component_button_mutes(theme, rules=rules)
    for st in ("rest", "hover", "pressed"):
        assert any("ds-bulk-danger" in b and f"({theme}, {st})" in b for b in bad), (
            f"LANE-143: a generic v2 rule over .ds-bulk-danger ({theme}, {st}) must fail the mute check, got {bad}"
        )


def test_the_mute_check_sees_a_planted_component_button(monkeypatch) -> None:
    """Positive control on synthetic sources: a component colour rule whose selector
    names no `.btn` is modelled through its template and caught when a generic v2
    `.btn` rule out-ranks it; a v2 rule naming the variant clears it."""
    monkeypatch.setattr(
        __import__(__name__, fromlist=["_"]), "component_template",
        lambda _path: '<div class="bar"><button class="btn x-warn" type="button">go</button></div>',
    )
    sheets = [("x.css", ".x-warn { color: var(--color-danger); }")]
    generic = [([f"{SCOPE} .btn"], {"color": "var(--color-text-primary)"})]
    bad = component_button_mutes("dark", sheets, rules=generic)
    assert len(bad) == 3 and all("x-warn" in b and "color" in b for b in bad), (
        f"LANE-143: a generic .btn colour over a planted .x-warn button must be caught in 3 states, got {bad}"
    )
    own = generic + [([f"{SCOPE} .btn.x-warn"], {"color": "var(--st-danger-fg)"})]
    assert component_button_mutes("dark", sheets, rules=own) == []
    # a neutral component paint, and a border the component switched off, lose nothing
    neutral = [("x.css", ".x-warn { color: var(--color-text-secondary); }")]
    assert component_button_mutes("dark", neutral, rules=generic) == []
    unbordered = [("x.css", ".x-warn { border: none; border-color: var(--color-brand); }")]
    assert component_button_mutes("dark", unbordered, rules=[([f"{SCOPE} .btn"], {"border-color": "var(--color-border-bright)"})]) == []


def test_a_declared_button_family_is_still_judged() -> None:
    """Positive control for the per-family button claim: a component rule that
    out-ranks a v2 button rule on a family that rule DECLARES is still a loss."""
    v2 = [([f"{SCOPE} .btn.cta"], {"color": "var(--ux-ink-on-brand)", "background": "var(--color-brand)"})]
    assert cascade_losses("dark", [("m.ts", ".btn.cta { color: white; }")], rules=v2) == []
    lost = cascade_losses("dark", [("m.ts", ".foot .btn.cta { color: white; }")], rules=v2)
    assert lost and "on color" in lost[0], f"LANE-143: a (0,4,0) ink over the v2 .btn.cta ink must be reported, got {lost}"
