"""Every white-on-danger surface clears the 3:1 WCAG 1.4.11 asks of a control.

RULE-20 class **S** (structural) guard for DECISION-22 / LANE-36: ``--color-danger``
shipped at ``oklch(0.70 0.17 25)`` = rgb(246,109,103), which measures **2.88:1
against white when fully opaque**. That is not a property of one button — it is a
property of the *token*, so every white-glyph-on-danger control in the product was
under the bar simultaneously, and no alpha could rescue it (raising the alpha of a
2.88:1 fill only moves it toward 2.88). It went unnoticed for the life of the
product because nothing computed the number.

So this guard computes it. It reads the token values out of
``frontend/src/styles.css``, converts OKLCH to sRGB the way a browser does,
composites each surface at the alpha its own rule uses over the worst-case
(lightest) tokened backdrop **in each theme**, and fails below 3:1 with the
number. A test that asserted ``--color-danger == "oklch(0.63 0.17 25)"`` would
pin today's value without knowing why, and would have to be edited — not
consulted — by the next person who wants a different red. This one lets them pick
any red that passes.

**Positive control** (CONVENTIONS "Tests" #11): this guard's shape is not the
vacuous "collect offenders, assert empty" — every row is a positive assertion
about a named surface — but the *math* underneath it could rot silently, so
``test_conversion_matches_browser_measurements`` pins the OKLCH conversion
against five values LANE-33 measured in a real browser, and
``test_guard_rejects_the_value_that_shipped`` feeds the old token back in and
requires the same function to reject it. If either goes quiet, the guard is dead.

**Wiring** (a served value is not a wired value): each row also asserts its source
line still exists and still names the token, so renaming a surface out from under
the guard fails loudly instead of silently reducing the covered set to zero.

Scope, deliberately: this covers the white-on-danger surfaces that exist today.
It cannot know about one added later — the token comment in ``styles.css`` is what
tells the next author that the lightness is a contract.

**Reachability, stated because it changes what one row means** (CONVENTIONS
"Tests" #10). ``.footer-action.danger:hover`` is the surface DECISION-22 named,
and it is a *primitive* in the shared sheet — but the product contains exactly one
``class="footer-action danger"`` element (``workspace/modes/details-mode.ts``),
and that component overrides the hover ``background`` with a 25% tint while the
primitive's partner declaration ``color: white`` lives in the global sheet. So the
solid fill this guard measures is the contract every future consumer inherits, not
necessarily pixels painted today. The split pair is a live hazard — an override
that moves the background without the colour — and resolving which declaration
wins needs the workspace rendered against a running backend. LANE-36 could not
measure it (backend not answering on :8000 during that pass) and deliberately did
not guess: see ``_harness/LESSONS.md``.

The *other* duty of
``--color-danger`` — danger-as-TEXT on a surface, which wants 4.5:1 — is NOT
pinned here: no single lightness satisfies it in both themes (the dark theme wants
a light red, the light theme a dark one), so it needs a per-theme token and that
is a design decision parked on DECISION-22, not a threshold to assert.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STYLES = REPO_ROOT / "frontend" / "src" / "styles.css"
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"

pytestmark = pytest.mark.skipif(
    not STYLES.is_file(), reason="frontend sources not present"
)

MIN_RATIO = 3.0
WHITE = (255, 255, 255)


# ─────────────────────────── colour maths (browser-equivalent) ──────────────

def oklch_to_srgb(lightness: float, chroma: float, hue_deg: float) -> tuple[int, int, int]:
    """OKLCH -> 8-bit sRGB, the conversion a browser performs for `oklch()`.

    Pinned against browser-measured values in
    ``test_conversion_matches_browser_measurements``.
    """
    hue = math.radians(hue_deg)
    a = chroma * math.cos(hue)
    b = chroma * math.sin(hue)
    l_ = lightness + 0.3963377774 * a + 0.2158037573 * b
    m_ = lightness - 0.1055613458 * a - 0.0638541728 * b
    s_ = lightness - 0.0894841775 * a - 1.2914855480 * b
    l3, m3, s3 = l_**3, m_**3, s_**3
    linear = (
        +4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3,
        -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3,
        -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3,
    )
    out = []
    for value in linear:
        v = max(value, 0.0)
        encoded = 12.92 * v if v <= 0.0031308 else 1.055 * (v ** (1 / 2.4)) - 0.055
        out.append(round(min(1.0, max(0.0, encoded)) * 255))
    return (out[0], out[1], out[2])


def relative_luminance(rgb: tuple[int, int, int]) -> float:
    """WCAG 2.x relative luminance."""

    def channel(c: int) -> float:
        s = c / 255
        return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    """WCAG 2.x contrast ratio, 1.0 .. 21.0."""
    hi, lo = sorted((relative_luminance(a), relative_luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def composite(
    fg: tuple[int, int, int], alpha: float, bg: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Source-over of an alpha fill on an opaque backdrop."""
    return tuple(round(f * alpha + b * (1 - alpha)) for f, b in zip(fg, bg))  # type: ignore[return-value]


# ───────────────────────────── token extraction ─────────────────────────────

_OKLCH = re.compile(
    r"oklch\(\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*\)"
)
_LIGHT_BLOCK = re.compile(
    r'html\[data-theme="light"\]\s*\{(.*?)\n\}', re.DOTALL
)


def _read_styles() -> str:
    return STYLES.read_text(encoding="utf-8")


def _declarations(css: str) -> str:
    """Strip /* ... */ comments so a measurement quoted in prose is never parsed
    as a declaration — the token comment names the OLD value on purpose."""
    return re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)


def token(name: str, theme: str = "dark") -> tuple[int, int, int]:
    """Resolve a token to sRGB for a theme, honouring the light-theme override.

    Raises if the token is missing entirely, so deleting or renaming one fails
    here rather than making the guard cover nothing.
    """
    css = _declarations(_read_styles())
    scope = css
    if theme == "light":
        block = _LIGHT_BLOCK.search(css)
        assert block, "the html[data-theme=\"light\"] override block has moved"
        override = re.search(rf"{re.escape(name)}\s*:\s*([^;]+);", block.group(1))
        if override:
            scope = override.group(0)
        else:
            scope = css  # deliberately shared with the dark theme
    match = re.search(rf"{re.escape(name)}\s*:\s*([^;]+);", scope)
    assert match, f"token {name} not found in styles.css (theme={theme})"
    values = _OKLCH.search(match.group(1))
    assert values, f"token {name} is not a plain oklch() value: {match.group(1)!r}"
    return oklch_to_srgb(*(float(v) for v in values.groups()))


# ─────────────────────────────── the surfaces ───────────────────────────────
# (label, source file, a substring of the rule that must still exist,
#  fill token, alpha the rule uses, backdrop token per theme)
#
# The backdrop is the WORST case: the lightest tokened surface the control can
# sit on, because a lighter backdrop raises the composite's luminance and so
# lowers its contrast against the white glyph. An opaque fill ignores it.
WHITE_ON_DANGER = [
    (
        ".footer-action.danger:hover — the destructive action in a card footer",
        "styles/components.css",
        ".footer-action.danger:hover { background: var(--color-danger); color: white; }",
        "--color-danger",
        1.0,
    ),
    (
        ".si-tag.danger — the DUPLICATE badge on a similar-images card",
        "app/modals/similar-images/similar-images.component.ts",
        ".si-tag.danger  { background: var(--color-danger); color: white;",
        "--color-danger",
        1.0,
    ),
    (
        ".tile-action delete — grid tile, composited over the thumbnail",
        "app/components/dataset/dataset-viewer/components/viewer-grid-view.ts",
        "bg-danger-overlay/88 hover:bg-danger-overlay text-white",
        "--color-danger-overlay",
        0.88,
    ),
    (
        "masking sidebar delete — composited over the thumbnail",
        "app/components/dataset/dataset-viewer/components/detail-masking-sidebar.ts",
        "bg-danger-overlay/88 hover:bg-danger-overlay text-white",
        "--color-danger-overlay",
        0.88,
    ),
]

# Lightest tokened backdrop per theme. A photo-composited surface is not bounded
# by a token at all — pure white is the worst a photo pixel can be.
WORST_BACKDROP = {
    "dark": "--color-surface-higher",
    "light": "--color-base",
}


def _worst_backdrop(theme: str, fill_token: str) -> tuple[int, int, int]:
    if fill_token == "--color-danger-overlay":
        return WHITE  # arbitrary photo pixels, the reason that token exists
    return token(WORST_BACKDROP[theme], theme)


def measure(fill_token: str, alpha: float, theme: str) -> float:
    fill = token(fill_token, theme)
    if alpha >= 1.0:
        return contrast_ratio(WHITE, fill)
    return contrast_ratio(WHITE, composite(fill, alpha, _worst_backdrop(theme, fill_token)))


# ──────────────────────────────── the guard ─────────────────────────────────

@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize(
    ("label", "source", "rule", "fill_token", "alpha"),
    WHITE_ON_DANGER,
    ids=[row[0].split(" —")[0] for row in WHITE_ON_DANGER],
)
def test_white_on_danger_clears_three_to_one(
    theme: str, label: str, source: str, rule: str, fill_token: str, alpha: float
) -> None:
    path = FRONTEND_SRC / source
    assert path.is_file(), f"{label}: {source} has moved — this guard now covers nothing"
    assert rule in path.read_text(encoding="utf-8"), (
        f"{label}: the rule this guard measures is no longer in {source}. "
        f"Expected to find {rule!r}. Update the row, do not delete it."
    )

    ratio = measure(fill_token, alpha, theme)
    assert ratio >= MIN_RATIO, (
        f"{label} in the {theme} theme measures {ratio:.2f}:1 white-on-fill, "
        f"under the {MIN_RATIO}:1 WCAG 1.4.11 asks of a non-text control. "
        f"{fill_token} is {token(fill_token, theme)} at alpha {alpha}. "
        f"Darken the token — raising the alpha cannot help once the opaque "
        f"colour is itself under the bar (DECISION-22 / LANE-36)."
    )


def test_conversion_matches_browser_measurements() -> None:
    """Pin the OKLCH maths against values measured in a real browser (LANE-33).

    Without this the guard could go green on a drifted conversion.
    """
    old_danger = oklch_to_srgb(0.70, 0.17, 25)
    overlay = oklch_to_srgb(0.55, 0.19, 25)
    assert old_danger == (246, 109, 103)
    assert overlay == (201, 47, 51)
    # LANE-33 measured these in the browser and recorded them in DECISION-22.
    assert round(contrast_ratio(WHITE, old_danger), 2) == 2.88
    assert round(contrast_ratio(WHITE, composite(old_danger, 0.80, (220, 221, 219))), 2) == 2.53
    assert round(contrast_ratio(WHITE, composite(overlay, 0.88, (220, 221, 219))), 2) == 4.71


def test_guard_rejects_the_value_that_shipped() -> None:
    """The positive control: feed the old token back in and require a failure.

    A threshold nobody has watched reject something is a threshold nobody knows
    is connected.
    """
    shipped = oklch_to_srgb(0.70, 0.17, 25)
    ratio = contrast_ratio(WHITE, shipped)
    assert ratio < MIN_RATIO, "the old value must still be rejected by this guard"
    assert 2.87 < ratio < 2.89, f"expected 2.88:1 for the shipped value, got {ratio:.2f}"


def test_danger_overlay_is_still_justified() -> None:
    """`--color-danger-overlay` survives only because a photo is not a token.

    If `--color-danger` ever gains enough headroom to clear 3:1 at 0.88 over
    pure white WITH margin, the second token is dead weight and should collapse
    into the first. This states the condition under which that becomes true, so
    the question is answered by measurement instead of being carried forever.
    """
    danger_over_white = contrast_ratio(
        WHITE, composite(token("--color-danger"), 0.88, WHITE)
    )
    overlay_over_white = contrast_ratio(
        WHITE, composite(token("--color-danger-overlay"), 0.88, WHITE)
    )
    assert overlay_over_white > danger_over_white, (
        "--color-danger-overlay no longer buys any headroom over --color-danger "
        f"({overlay_over_white:.2f}:1 vs {danger_over_white:.2f}:1) — collapse it "
        "into --color-danger and retire the token (DECISION-22 item 2)."
    )
    assert danger_over_white < 4.0, (
        f"--color-danger now measures {danger_over_white:.2f}:1 at 0.88 over pure "
        "white, comfortable margin over the 3:1 bar even against the brightest "
        "photo pixel. The reason --color-danger-overlay exists has expired: "
        "retire it and point its two consumers at --color-danger."
    )


def test_no_danger_hue_literal_shadows_the_token() -> None:
    """A hard-coded danger red is invisible to everything above.

    The token was darkened; 20 literals of the old hue were spread across 6 files
    as tints and borders, and a literal cannot be re-measured by reading
    styles.css. This is the collect-offenders shape, so it carries its own
    positive control below.
    """
    offenders = _find_danger_literals(FRONTEND_SRC)
    assert not offenders, (
        "hard-coded danger-hue literals shadow --color-danger and escape the "
        "contrast guard above — use color-mix(in oklab, var(--color-danger) N%, "
        "transparent):\n  " + "\n  ".join(offenders)
    )


def _find_danger_literals(root: Path) -> list[str]:
    """Old danger hues written as literals, outside comments."""
    patterns = (re.compile(r"oklch\(\s*0\.70\s+0\.17\s+25\s*[/)]"),
                re.compile(r"oklch\(\s*0\.65\s+0\.20\s+25\s*[/)]"))
    found: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.suffix not in {".css", ".ts", ".html"} or not path.is_file():
            continue
        text = _declarations(path.read_text(encoding="utf-8"))
        for lineno, line in enumerate(text.splitlines(), start=1):
            if any(p.search(line) for p in patterns):
                found.append(f"{path.relative_to(root)}:{lineno}: {line.strip()[:100]}")
    return found


def test_literal_scan_catches_a_known_offender(tmp_path: Path) -> None:
    """Positive control for the scan above (CONVENTIONS "Tests" #11).

    Without this, a drifted regex or a changed file layout leaves the scan
    returning an empty list forever and it passes for the wrong reason.
    """
    (tmp_path / "bad.css").write_text(
        ".x { background: oklch(0.70 0.17 25 / 0.12); }", encoding="utf-8"
    )
    (tmp_path / "ok.css").write_text(
        "/* was oklch(0.70 0.17 25) */\n"
        ".y { background: color-mix(in oklab, var(--color-danger) 12%, transparent); }",
        encoding="utf-8",
    )
    offenders = _find_danger_literals(tmp_path)
    assert len(offenders) == 1, offenders
    assert offenders[0].startswith("bad.css:1:")
