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
    m = re.search(r"var\(\s*(--[\w-]+)\s*(?:,\s*([^()]*))?\)", value)
    if not m:
        return value
    name, fallback = m.group(1), m.group(2)
    if name in env:
        sub = env[name]
    else:
        assert fallback is not None, f"LANE-143: token {name} is not defined"
        sub = fallback
    return resolve_vars(value[: m.start()] + sub + value[m.end():], env, depth + 1)


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


def recipe(theme: str, *selectors: str, rules=None) -> tuple[dict[str, str], list[str]]:
    """Merged declarations of the rules for ``selectors`` (in cascade order), and
    the names of the rules that contributed (for the wiring assertions)."""
    rules = _ux_rules() if rules is None else rules
    merged: dict[str, str] = {}
    hit: list[str] = []
    for sel in selectors:
        for heads, decls in rules:
            wanted = [f"{SCOPE} {sel}"] + ([f"{LIGHT_SCOPE} {sel}"] if theme == "light" else [])
            if any(w in heads for w in wanted):
                merged.update(decls)
                hit.append(sel)
    return merged, hit


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


def _chip(theme: str, role: str) -> Painted:
    decls, hit = recipe(theme, ".chip", f".chip.{role}")
    assert f".chip.{role}" in hit, f"LANE-143: no .chip.{role} recipe in ux-v2.css"
    return Painted(decls, theme)


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
