"""Repo-hygiene guard: nothing per-card may cost a compositor pass or a full decode.

RULE-20 class S guard for the dataset-library scroll defect (UAT round 3): fast
scrolling through a long library was impossible — cards visibly lost position
sync and only settled once scrolling stopped.

Two independent causes, measured on the running app with a rAF-delta sweep:

  1. `backdrop-filter` on pills that repeat per card. 377 blurred layers, each a
     compositor pass per frame: median frame 50ms, 22 of 24 frames dropped.
     Removing it (the alpha carries the legibility instead) took the median to
     16.7ms. A 0.75 fill behind a 6px blur and a 0.86 flat fill are
     near-indistinguishable over a photo; only one of them costs a pass per card.
  2. Full-size covers. 94 images, median 2.36 MP and one at 58 MP (9339x6223),
     598 MP of decoded bitmap for 3.9 MP of screen. With the images hidden the
     identical sweep ran at a flat 60fps with zero dropped frames, which is what
     identified this as the entire residual. Covers now request a bounded
     thumbnail rendition.

Why a source scan and not only the per-site specs: both causes are the kind of
thing re-introduced by a single innocuous line — one more pill with a blur, one
more card `<img>` pointed at `/media`. A spec pins the sites that were fixed; a
scan is what stops the next one being written. A hit here means "go read the
site and measure", never "mechanically delete the property".
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"

# Selectors that render once per dataset card or per grid tile. A blur here is
# multiplied by the item count; a blur on a modal or a single toolbar is not.
PER_ITEM_SELECTOR = re.compile(
    r"^\s*\.(ds-card[\w-]*|hps-pill|state-pills-pad|cell\b[\w.-]*|tile[\w-]*)",
)
BACKDROP_FILTER = re.compile(r"^\s*(-webkit-)?backdrop-filter\s*:")


def _css_files() -> list[Path]:
    return sorted(FRONTEND_SRC.rglob("*.css"))


def test_the_scan_sees_the_stylesheets_it_claims_to_scan() -> None:
    """Prove the negative: an empty sweep would pass every assertion below."""
    files = _css_files()
    assert len(files) > 5, f"only found {len(files)} stylesheets under {FRONTEND_SRC}"
    names = {f.name for f in files}
    assert "datasets-screen.css" in names
    assert "components.css" in names


def test_no_backdrop_filter_on_a_per_card_selector() -> None:
    """Cause 1. Each instance is a compositor pass per frame, per card."""
    offenders: list[str] = []

    for path in _css_files():
        current: str | None = None
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = PER_ITEM_SELECTOR.match(line)
            if match:
                current = match.group(1)
            elif line.strip().startswith("}"):
                current = None
            elif current and BACKDROP_FILTER.match(line):
                rel = path.relative_to(REPO_ROOT).as_posix()
                offenders.append(f"{rel}:{lineno} — .{current} {line.strip()}")

    assert not offenders, (
        "backdrop-filter on a selector that repeats per card/tile. Measured at "
        "377 instances: median frame 50ms, 22 of 24 frames dropped. Raise the "
        "background alpha instead.\n  " + "\n  ".join(offenders)
    )


# Templates whose `<img>` shows a dataset cover inside a card-sized box.
COVER_TEMPLATES = (
    "app/screens/datasets-screen/datasets-screen.html",
    "app/screens/projects-screen/project-detail.ts",
    "app/components/training/dynamic-form-group/dynamic-form-group.ts",
)


@pytest.mark.parametrize("rel_path", COVER_TEMPLATES)
def test_cover_images_are_lazy_and_decoded_off_the_main_thread(rel_path: str) -> None:
    """Cause 2, the cheap half — a cover must never block the scroll to decode."""
    path = FRONTEND_SRC / rel_path
    assert path.exists(), f"{rel_path} moved; re-point this guard"
    source = path.read_text(encoding="utf-8")

    # Only the library grid renders enough covers at once for this to matter;
    # the other two show one or a handful, so they are exempt from `loading`.
    if rel_path.endswith("datasets-screen.html"):
        assert 'loading="lazy"' in source
        assert 'decoding="async"' in source


# A URL built on the media base — the shape that serves original bytes.
RAW_MEDIA_URL = re.compile(r"\$\{[\w.]*mediaBase[\w]*\}/")
# ...naming the cover field. Full-size IS correct for the viewer, the edit
# canvas and the mask preview; those show one image, deliberately unresized.
# `preview_image` is only ever painted into a card.
COVER_FIELD = re.compile(r"preview_image")


def test_no_cover_is_built_from_the_raw_media_base() -> None:
    """Cause 2, the half that mattered — 598 MP of bitmap for 3.9 MP of screen.

    `mediaBaseUrl` interpolated straight into a cover URL is the shape that
    served originals into 260px cards. It is legitimate exactly once, inside
    the shared helper's `directDatasetMediaUrl` (animated GIFs, and the
    fallback for a format Pillow cannot decode).

    Found on its first run: a third copy of this builder in
    `project-export.service.ts`, which had also drifted into leaving
    `preview_image` un-encoded.
    """
    shared_helper = FRONTEND_SRC / "app/shared/media-preview.ts"

    offenders: list[str] = []
    for path in sorted(FRONTEND_SRC.rglob("*.ts")):
        if path == shared_helper or path.name.endswith(".spec.ts"):
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("*", "//", "/*")):
                continue  # prose about the shape, not the shape
            if RAW_MEDIA_URL.search(line) and COVER_FIELD.search(line):
                rel = path.relative_to(REPO_ROOT).as_posix()
                offenders.append(f"{rel}:{lineno} — {stripped}")

    assert not offenders, (
        "a dataset cover URL built from mediaBaseUrl directly serves the "
        "full-size original into a card-sized box. Use datasetPreviewUrl().\n  "
        + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# The workspace browse grid — the same defect, one surface over.
#
# The scan above keys on `preview_image`, the LIBRARY's cover field, so it is
# structurally blind to a grid whose tiles are named `media_file`. Measured on
# the browse grid of a 263-item dataset before the fix: 5887.7 MP of decoded
# bitmap for 28.5 MP of `<img>` boxes (median source 8.19 MP, max 42.33 MP),
# a rAF-delta sweep at 3.9-6.9 fps with 433 of 433 frames over 20ms, and a
# full sweep that repeatedly killed the renderer.
#
# `media_file` cannot be scanned repo-wide the way `preview_image` is: the
# detail viewer, the edit canvas and the mask preview all show ONE image and
# are deliberately unresized. So this is pinned per template that repeats a
# tile, by name.
# ---------------------------------------------------------------------------

TILE_TEMPLATE = (
    "app/components/dataset/dataset-viewer/components/viewer-grid-view.ts"
)


def _method_body(source: str, signature_prefix: str) -> str:
    """Return the text of the method whose line starts with *signature_prefix*.

    Brace-counted from the signature line, so an inner `{` (an object literal,
    a template string) does not truncate the body the way a naive
    "up to the next blank line" split would.
    """
    start = source.index(signature_prefix)
    depth = 0
    for i in range(start, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
    raise AssertionError(f"unbalanced braces after {signature_prefix!r}")


def test_the_grid_tile_img_binds_the_resolver_this_guard_pins() -> None:
    """Prove the negative: pinning a method nothing renders would pass anyway."""
    source = (FRONTEND_SRC / TILE_TEMPLATE).read_text(encoding="utf-8")
    assert '<img [src]="getDisplayUrl(pair)"' in source, (
        "the grid tile no longer binds getDisplayUrl; re-point this guard at "
        "whatever resolves the tile image now"
    )
    # Decoding a tile image on the main thread is a scroll stall per tile.
    assert 'decoding="async"' in source
    assert 'loading="lazy"' in source
    # Two independent fallbacks share this one handler and are NOT
    # interchangeable: a failed overlay drops to the source image, a failed
    # rendition drops to the source's original bytes. Binding `onOverlayError`
    # here — which is what it used to be — leaves an undecodable format as a
    # permanently blank tile, because nothing about its URL changes.
    assert '(error)="onTileImageError(pair)"' in source


def test_the_grid_tile_image_requests_a_bounded_rendition() -> None:
    """A 320px-tall tile must not decode a 42 MP training source.

    Two of the four sources that feed a tile stay on `/media` on purpose and
    the assertion is written to allow exactly that: `masked/<stem>.jpg` and
    `overlays/<stem>.png` are rewritten in place by paths that never call
    `thumbnails.invalidate_thumbnail` on those paths, so a rendition of either
    would paint pre-edit pixels. What is pinned here is the DEFAULT branch —
    the one every tile of an ordinary browse grid takes.
    """
    source = (FRONTEND_SRC / TILE_TEMPLATE).read_text(encoding="utf-8")
    body = _method_body(source, "    getDisplayUrl(pair: GridPair): string {")

    assert "this.renditionUrl(" in body, (
        "viewer-grid-view.getDisplayUrl resolves a tile without a bounded "
        "rendition. Measured at 5887.7 MP of bitmap for 28.5 MP of boxes and "
        "3.9 fps; a full sweep killed the renderer. Route it through "
        "renditionUrl() (GET /datasets/{name}/thumbnail)."
    )
    # The plain-media and effective-target branches must BOTH be renditioned:
    # an edit dataset's tiles paint effective_target and are just as large.
    for raw in ("return this.getMediaUrl(pair.media_file)",
                "return this.getMediaUrl(pair.effective_target)"):
        assert raw not in body, f"{raw} — full-size original into a 320px tile"

    rendition = _method_body(
        source, "    protected renditionUrl(relativePath: string): string {",
    )
    assert "/thumbnail" in rendition
    assert "max_edge=${PREVIEW_MAX_EDGE}" in rendition, (
        "the rendition size must come from the shared PREVIEW_MAX_EDGE, not a "
        "literal: 512 chosen against one machine's DPR is what shipped visibly "
        "soft covers on the library"
    )
    # Both deliberate exceptions, or the grid gets holes and frozen GIFs.
    assert "staysAnimated(relativePath)" in rendition
    assert "this.failedRenditions().has(relativePath)" in rendition


def test_the_video_poster_is_sized_from_the_same_constant() -> None:
    """A video tile's poster shares the still tiles' box, so it shares the size.

    Found by the LANE-29 agent AFTER the still tiles were fixed: `thumbnailUrl`
    omitted `max_edge` entirely and took the endpoint's 256 default into the
    same 320px box — soft at DPR 1 and worse above it. That is the identical
    mistake the library shipped once already by sizing a rendition against one
    machine, reappearing in the surface next door, which is why the constant is
    pinned rather than the number.
    """
    source = (FRONTEND_SRC / TILE_TEMPLATE).read_text(encoding="utf-8")
    poster = _method_body(source, "    thumbnailUrl(pair: GridPair): string {")

    assert "max_edge=${PREVIEW_MAX_EDGE}" in poster, (
        "the video poster falls back to the endpoint's 256 default in a 320px "
        "tile. Size it from PREVIEW_MAX_EDGE like every other tile image."
    )


# Utility-class blurs the CSS scan above is structurally blind to: Tailwind
# emits `backdrop-blur-*` from a template `class` attribute, so no stylesheet
# ever contains the property.
#
# The budget is ZERO, decided by the user as DECISION-21 after a one-variable
# ablation on the running browse grid (263 tiles, production build). What the
# ablation established, so nobody re-derives a wrong lesson from a stale
# number:
#
#   * Cost scales with the `backdrop-filter` INSTANCE COUNT, not with the
#     visible instances. Gating the blur so only the hovered tile's cluster
#     was blurred measured IDENTICALLY to leaving all 1479 on — Chromium pays
#     for the layer, not for what you can see. "Only show it on hover" is not
#     a fix.
#   * Cost does not scale with RADIUS either. 8px -> 2px measured 35.3ms vs
#     35.4ms: inside noise. The "compromise, keep a small blur" option is
#     worth exactly nothing and must not be reintroduced as a middle ground.
#   * SHADOWS ARE NOT THE LEVER. 189 shadow instances on the library measured
#     0ms — they are painted into the tile's own layer, not composited per
#     frame. Do not "optimise" `shadow-lg`, `.hps-pill` or `.state-pills-pad`
#     chasing this defect; they were measured free and are deliberately
#     untouched.
#   * Removing all six took the median frame from 35.7ms to 17.7ms
#     (231 dropped, 51.2 fps) on a full top-to-bottom sweep.
#
# THE OTHER HALF, which is not optional: the blur was doing legibility work.
# At the old resting alpha 0.60 the muted grey glyphs lose separation over a
# bright photo once the blur is gone. Dropping the blur WITHOUT raising the
# resting fill to 0.88 ships a contrast regression that looks like a rendering
# bug. The two halves travel together — the alpha guard below pins the second
# half so a future "simplify the alphas" pass cannot silently undo it.
#
# Counted from the `@for` that repeats the item, so the ONE blur on the grid's
# own toolbar (rendered once, whatever the tile count) is correctly not in the
# budget — the cost this pins is the multiplication, not the property.
TEMPLATE_BLUR_BUDGET = {
    TILE_TEMPLATE: ("@for (pair of pairs()", 0),
}
TAILWIND_BLUR = re.compile(r"backdrop-blur")


@pytest.mark.parametrize(
    ("rel_path", "repeat_marker", "budget"),
    [(k, *v) for k, v in sorted(TEMPLATE_BLUR_BUDGET.items())],
)
def test_per_item_templates_do_not_grow_more_utility_blurs(
    rel_path: str, repeat_marker: str, budget: int,
) -> None:
    path = FRONTEND_SRC / rel_path
    assert path.exists(), f"{rel_path} moved; re-point this guard"
    lines = path.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if repeat_marker in line]
    assert starts, f"{rel_path} no longer contains {repeat_marker!r}; re-point this guard"
    hits = [
        f"{lineno}: {line.strip()[:90]}"
        for lineno, line in enumerate(lines[starts[0]:], starts[0] + 1)
        if TAILWIND_BLUR.search(line)
    ]
    assert len(hits) <= budget, (
        f"{rel_path} now has {len(hits)} utility blurs on a per-tile element, "
        f"budget {budget}. Each is a compositor pass per tile per frame, "
        "whether or not it is visible: 6 per tile x 263 tiles = 1479 layers, "
        "35.7ms median. Removing them all measured 17.7ms / 51.2 fps "
        "(DECISION-21). Gating the blur to the hovered tile measured no better "
        "and shrinking the radius to 2px measured no better — the only lever "
        "is the instance count. Raise the background alpha instead (0.88 flat, "
        "which is what carries legibility now).\n  " + "\n  ".join(hits)
    )


# The second half of DECISION-21. The blur was carrying contrast as well as
# style; the alpha that replaced it is load-bearing, and a "tidy the opacity
# scale" pass would revert it without touching a single blur.
#
# Both forms are pinned because the five overlay actions are written two ways:
# adjust/crop use the Tailwind utility, pin/exclude use a `color-mix` fill in
# the component stylesheet (they need a theme-aware tint layered on top).
TILE_ACTION_ALPHA = {
    # substring that must appear, minimum count, the form it replaced
    "bg-surface-low/88": (2, "bg-surface-low/60"),
    "color-mix(in oklab, var(--color-surface-low) 88%, transparent)": (
        4, "color-mix(in oklab, var(--color-surface-low) 60%, transparent)",
    ),
}


@pytest.mark.parametrize(
    ("required", "minimum", "superseded"),
    [(k, *v) for k, v in sorted(TILE_ACTION_ALPHA.items())],
)
def test_the_tile_actions_keep_the_alpha_that_replaced_the_blur(
    required: str, minimum: int, superseded: str,
) -> None:
    source = (FRONTEND_SRC / TILE_TEMPLATE).read_text(encoding="utf-8")
    # Prose about the old value is allowed to survive in a comment; a live
    # declaration is not. Strip nothing — the superseded form is distinctive
    # enough that any occurrence is worth a human look.
    declarations = [
        line.strip()
        for line in source.splitlines()
        if superseded in line and not line.strip().startswith(("*", "//", "/*"))
    ]
    assert not declarations, (
        f"{TILE_TEMPLATE} still declares {superseded!r}. The per-tile "
        "backdrop-blur was removed under DECISION-21 and the resting alpha is "
        "what replaces its legibility — at 0.60 the muted glyphs lose "
        "separation over a bright photo. The two halves travel together.\n  "
        + "\n  ".join(declarations)
    )
    assert source.count(required) >= minimum, (
        f"{TILE_TEMPLATE} has {source.count(required)} occurrences of "
        f"{required!r}, expected at least {minimum}. Prove the negative: this "
        "guard is worthless if the elements it pins were renamed away."
    )


# ---------------------------------------------------------------------------
# Why the raised alpha is safe in BOTH themes, and what keeps it safe.
#
# The alpha above is only half of a contrast; the other half is the glyph. It
# holds because the fill and the glyph flip TOGETHER: the fill resolves
# `--color-surface-low` (dark 0.14 L / light 0.96 L) and the glyph resolves
# `--color-text-muted` (dark 0.66 L / light 0.48 L), so the pad inverts under
# the icon instead of sliding toward it. Measured in the browser on the two
# ends of the backdrop range, composited over the actual photo:
#
#            backdrop RGB 220/221/219      backdrop RGB 15/13/15
#   dark               5.07:1                      6.37:1
#   light              5.68:1                      4.48:1
#
# All four muted actions clear the 3:1 that WCAG 1.4.11 asks of a non-text UI
# component in both themes, and clear it by more than the 2.09:1 the old
# alpha 0.60 gave over the bright backdrop.
#
# That pairing is NOT automatic, which is the whole reason this is a test and
# not a comment. `.hps-pill` and `.ds-card-size` sit on the same tiles with a
# FIXED dark `oklch` fill precisely because a theme-flipping glyph on a
# theme-independent pad is unreadable in one of the two themes — that is the
# documented precedent for getting this wrong. `.tile-action` takes the other
# valid option, both-sides-flipping. What is forbidden is MIXING them, and a
# mix is a one-word edit: `text-text-muted` -> `text-white` on a pad that
# still flips, and the light theme ships white-on-white.
#
# The delete action is deliberately exempt: `bg-danger/80` + `text-white` is
# fixed on BOTH sides, the `.hps-pill` pattern, self-consistent across themes.
# It measures 2.53:1 over the bright backdrop in both themes — a pre-existing
# property of the danger hue, unchanged by DECISION-21 (a blur does not move
# the mean luminance it composites over, so it measured the same before).
# ---------------------------------------------------------------------------

# Element carrying the flipping fill -> the glyph token it must carry with it.
UTILITY_PAIR = ("bg-surface-low/88", "text-text-muted")
# CSS rules whose flipping fill must be declared alongside the flipping color.
CSS_RULE_PAIR = (".tile-pin", ".tile-exclude")


def test_a_theme_flipping_tile_action_fill_keeps_a_theme_flipping_glyph() -> None:
    """The utility half: adjust and crop are styled entirely in the template."""
    fill, glyph = UTILITY_PAIR
    source = (FRONTEND_SRC / TILE_TEMPLATE).read_text(encoding="utf-8")
    offenders = [
        f"{lineno}: {line.strip()[:110]}"
        for lineno, line in enumerate(source.splitlines(), 1)
        if fill in line and glyph not in line
    ]
    assert not offenders, (
        f"a per-tile element carries {fill!r} without {glyph!r}. The 0.88 fill "
        "flips with the theme (0.14 L dark / 0.96 L light); a glyph that does "
        "not flip with it is unreadable in one of the two themes. Either keep "
        "both theme-driven (measured 5.07-6.37:1 dark, 4.48-5.68:1 light) or "
        "make BOTH fixed like .hps-pill — never one of each.\n  "
        + "\n  ".join(offenders)
    )


def _rule_block(source: str, selector: str) -> str:
    """The declarations of `selector {` — the base rule, not its :hover/state."""
    start = source.index(f"{selector} {{")
    return source[start:source.index("}", start)]


@pytest.mark.parametrize("selector", CSS_RULE_PAIR)
def test_the_tile_action_css_rules_declare_fill_and_glyph_together(
    selector: str,
) -> None:
    """The stylesheet half: pin and exclude need a theme-aware tint on top."""
    source = (FRONTEND_SRC / TILE_TEMPLATE).read_text(encoding="utf-8")
    block = _rule_block(source, selector)

    assert "var(--color-surface-low)" in block, (
        f"{selector} no longer fills from --color-surface-low; re-point this "
        "guard, and re-measure both themes before you do"
    )
    assert "var(--color-text-muted)" in block, (
        f"{selector} sets a theme-flipping fill but not a theme-flipping "
        "color. Its glyph must move with its pad: --color-text-muted is 0.66 L "
        "in dark and 0.48 L in light, which is what keeps the icon legible "
        "when --color-surface-low inverts underneath it. Fixing one side only "
        "is the mistake .hps-pill's fixed-oklch comment exists to warn about."
    )
#
# This is the cost the element-count theory was hiding. A 263-item workspace
# ran 1503 CSS animations SIMULTANEOUSLY - 789 filmstrip spinner dots plus 714
# grid loader dots, three spans each - and the app rendered at 35.5ms / 28fps
# *while completely idle*: the idle rAF median equalled the scrolling median,
# so it was never a scroll problem and virtualization was never the fix.
# Bounding the animations took the same production-build sweep from 35.8ms /
# 27.9fps / 117-of-120 dropped to 18.1ms / 55.2fps.
#
# Three cheaper explanations were measured and all three were WRONG:
#   * `content-visibility: auto` on the item - no effect at 789 animations,
#     and after the fix it measured WORSE (35.1ms vs 18.1ms). Do not ship it.
#   * `display: none` on the spinner - 789 animations still running. Chromium
#     keeps them; only removal from the DOM (`@if`) stops one.
#   * hiding the images - 18.0ms vs 18.1ms, i.e. nothing.
#
# And the OBVIOUS gate is insufficient, which is the part worth pinning: the
# media is lazily loaded, so an off-screen image never loads and never fires
# `load`. 34 of 263 had loaded; a load-only gate would have left 687
# animations running. The bound must come from VISIBILITY.
# ---------------------------------------------------------------------------

FILMSTRIP_TEMPLATE = "app/workspace/filmstrip-scrubber/filmstrip-scrubber.component.ts"

# rel_path -> (class of the element carrying the infinite animation,
#              the repeat marker of the item it sits inside)
PERPETUAL_ANIMATION_OWNERS = {
    TILE_TEMPLATE: ("grid-thumb-loader", "@for (pair of pairs()"),
    FILMSTRIP_TEMPLATE: ("thumb-spinner", "@for (c of cells()"),
}


@pytest.mark.parametrize(
    ("rel_path", "css_class", "repeat_marker"),
    [(k, *v) for k, v in sorted(PERPETUAL_ANIMATION_OWNERS.items())],
)
def test_the_animated_element_this_guard_pins_still_exists(
    rel_path: str, css_class: str, repeat_marker: str,
) -> None:
    """Prove the negative: a guard on an element nobody renders passes anyway."""
    source = (FRONTEND_SRC / rel_path).read_text(encoding="utf-8")
    assert repeat_marker in source, (
        f"{rel_path} no longer repeats with {repeat_marker!r}; re-point this guard"
    )
    assert f'class="{css_class}"' in source, (
        f"{rel_path} no longer renders .{css_class}; re-point this guard"
    )
    assert "infinite" in source, (
        f"{rel_path} declares no infinite animation any more - if the dots are "
        "gone, delete this guard deliberately rather than let it pass empty"
    )


@pytest.mark.parametrize(
    ("rel_path", "css_class", "repeat_marker"),
    [(k, *v) for k, v in sorted(PERPETUAL_ANIMATION_OWNERS.items())],
)
def test_a_perpetual_animation_in_a_repeated_item_is_gated_on_visibility(
    rel_path: str, css_class: str, repeat_marker: str,
) -> None:
    lines = (FRONTEND_SRC / rel_path).read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if repeat_marker in line)
    uses = [
        i for i, line in enumerate(lines)
        if i >= start and f'class="{css_class}"' in line
    ]
    assert uses, f"{rel_path}: .{css_class} is not rendered inside the repeat"

    for use in uses:
        # Walk back to the nearest control-flow block opening this element.
        gate = next(
            (lines[i] for i in range(use, start - 1, -1) if "@if (" in lines[i]),
            None,
        )
        assert gate is not None and "isPending(" in gate, (
            f"{rel_path}:{use + 1} renders .{css_class} - an "
            "`animation: ... infinite` - without an `@if (isPending(...))` "
            "around it. Measured: 263 items x 3 dots kept 789 animations alive "
            "and cost 18ms of EVERY frame, idle included. Covering the dots "
            "with the loaded image does not stop them, and neither does "
            "`display: none` (both measured). Only removing the element does."
            f"\n  nearest gate: {(gate or chr(60) + chr(110) + chr(111) + chr(110) + chr(101) + chr(62)).strip()[:100]}"
        )


@pytest.mark.parametrize("rel_path", sorted(PERPETUAL_ANIMATION_OWNERS))
def test_the_visibility_gate_is_bounded_by_the_viewport_not_by_load(
    rel_path: str,
) -> None:
    source = (FRONTEND_SRC / rel_path).read_text(encoding="utf-8")
    assert "createInViewTracker" in source, (
        f"{rel_path} gates its perpetual animation without the in-view tracker. "
        "A load-only gate does NOT bound it: the media is lazily loaded, so an "
        "off-screen item never loads and never fires `load`. Measured 34 of "
        "263 loaded - 687 animations would have survived that gate."
    )
    body = _method_body(source, "    protected isPending(")
    assert "inView.has(" in body, (
        f"{rel_path}: isPending no longer consults the in-view tracker, so the "
        "gate has silently become load-only again - the exact insufficient "
        "bound this guard exists to stop."
    )


def test_the_in_view_tracker_degrades_to_showing_not_to_hiding() -> None:
    """A missing IntersectionObserver must not blank a list."""
    source = (FRONTEND_SRC / "app/shared/in-view-tracker.ts").read_text(encoding="utf-8")
    assert "typeof IntersectionObserver !== 'undefined'" in source, (
        "the tracker no longer feature-detects IntersectionObserver"
    )
    body = _method_body(source, "        has(index: number): boolean {")
    assert "if (!supported" in body and "return true" in body, (
        "in-view-tracker.has() must answer true where no observer exists. "
        "Degrading to 'hidden' turns every item of a list into an empty box "
        "on any engine or test environment without IntersectionObserver."
    )


# ---------------------------------------------------------------------------
# LANE-33 - the fifth tile action, the one DECISION-21 left exempt.
#
# The four muted actions were fixed by raising an alpha because their fill and
# their glyph flip together. The delete action cannot be fixed that way: its
# fill and its glyph are BOTH fixed, so the only lever is the fill's colour.
# `--color-danger` is rgb(246,109,103), and at the 0.80 alpha these overlays
# carried it measured 2.53:1 white-glyph-to-fill over a near-white photo
# (RGB 220/221/219) in BOTH themes. It is not an alpha problem: that hue is
# 2.88:1 even FULLY OPAQUE, so there is no alpha at which it clears the 3:1
# WCAG 1.4.11 asks of a non-text UI component. The fix is
# `--color-danger-overlay`, rgb(201,47,51), used wherever a danger control is
# composited over media.
#
# WHAT THIS SECTION CAN AND CANNOT PROVE, stated plainly rather than dressed up:
#
#   * It CAN compute the property. The tests below implement OKLCH -> sRGB and
#     the WCAG relative-luminance ratio, read the shipped token literal out of
#     `styles.css`, and assert the composited ratio at both ends of the
#     backdrop range. That is not a literal pin - lighten the token to ANY
#     value that fails and the arithmetic fails with it, whatever number was
#     chosen.
#
#   * The model is CALIBRATED, not assumed: the first test reproduces the
#     browser's 2.53:1 baseline to two decimals from the OLD token. A model
#     that drifted from the renderer would lose that assertion first.
#
#   * The numbers it computes were CHECKED against the running app, not merely
#     asserted. Chromium resolves the token to `oklab(0.55 0.172198 0.0802975
#     / 0.88)` -> rgb(201,47,51) and composites it to 4.46 / 4.68 / 6.43:1 over
#     white / RGB 220,221,219 / RGB 15,13,15; this model says 4.47 / 4.71 /
#     6.38:1, and the PAINTED pixels of an element screenshot measure 4.39:1
#     (the SVG stroke antialiases to rgb(248,230,229), not pure white, so the
#     real glyph is slightly dimmer than the token that declares it). The
#     spread is under 0.1 in either direction - the renderer blends in oklab
#     where this blends in 8-bit sRGB - and it is why the threshold is asserted
#     at 3.0 with a measured margin near 1.4, not shaved to the edge.
#
#   * It CANNOT prove what the browser paints. Alpha compositing happens in the
#     renderer, the glyph is an antialiased SVG stroke, and `shadow-lg` and the
#     tile's own layers sit in between. The screenshots in the LANE-33 report
#     are that half, and no assertion here replaces them. What the arithmetic
#     buys is that the SOURCE can no longer drift back under 3:1 unnoticed
#     between browser passes - the failure mode that let 2.53:1 ship.
#
#   * The backdrop is an ASSUMPTION, and a stated one: RGB 220/221/219 is the
#     brightest tile measured on the fixture dataset, not a proof about every
#     photo a user owns. Pure white is asserted alongside it precisely because
#     a photo brighter than the fixture certainly exists.
# ---------------------------------------------------------------------------

# The two surfaces where a danger control is composited over arbitrary photo
# pixels. Both take the overlay token; the failing form is barred below.
DANGER_OVERLAY_TEMPLATES = (
    TILE_TEMPLATE,
    "app/components/dataset/dataset-viewer/components/detail-masking-sidebar.ts",
)
# A translucent danger fill under a FIXED white glyph - the signature of the
# defect, matched as a pattern so a NEW overlay written the old way is caught
# too, not only the two that were fixed.
TRANSLUCENT_DANGER_FILL = re.compile(r"bg-danger/\d")
DANGER_OVERLAY_FILL = "bg-danger-overlay/88"

# Backdrop extremes, sRGB. BRIGHT is tile 124 of the fixture dataset, the
# brightest measured; WHITE is the bound no photo can exceed; DARK is the
# other end, where a translucent fill is at its safest.
BACKDROP_WHITE = (255, 255, 255)
BACKDROP_BRIGHT = (220, 221, 219)
BACKDROP_DARK = (15, 13, 15)
GLYPH_WHITE = (255, 255, 255)
WCAG_NON_TEXT_MINIMUM = 3.0

OKLCH_LITERAL = re.compile(
    r"--color-danger-overlay:\s*oklch\(\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\)",
)


def _oklch_to_srgb(lightness: float, chroma: float, hue_deg: float) -> tuple[int, int, int]:
    """OKLCH -> 8-bit sRGB, the conversion the renderer performs.

    Bit-for-bit against Chromium on the two colours this section uses:
    oklch(0.70 0.17 25) -> (246, 109, 103) and oklch(0.55 0.19 25) ->
    (201, 47, 51), both read back out of a canvas.
    """
    hue = math.radians(hue_deg)
    a = chroma * math.cos(hue)
    b = chroma * math.sin(hue)
    long_ = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    med_ = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    short_ = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    linear = (
        +4.0767416621 * long_ - 3.3077115913 * med_ + 0.2309699292 * short_,
        -1.2684380046 * long_ + 2.6097574011 * med_ - 0.3413193965 * short_,
        -0.0041960863 * long_ - 0.7034186147 * med_ + 1.7076147010 * short_,
    )
    out = []
    for channel in linear:
        c = max(0.0, min(1.0, channel))
        encoded = 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055
        out.append(round(max(0.0, min(1.0, encoded)) * 255))
    return out[0], out[1], out[2]


def _relative_luminance(rgb: tuple[float, ...]) -> float:
    channels = []
    for value in rgb:
        c = value / 255
        channels.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast(one: tuple[float, ...], two: tuple[float, ...]) -> float:
    a, b = _relative_luminance(one), _relative_luminance(two)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def _composite(
    fill: tuple[int, int, int], alpha: float, backdrop: tuple[int, int, int],
) -> tuple[float, ...]:
    return tuple(alpha * f + (1 - alpha) * b for f, b in zip(fill, backdrop))


def _overlay_token() -> tuple[float, float, float]:
    source = (FRONTEND_SRC / "styles.css").read_text(encoding="utf-8")
    match = OKLCH_LITERAL.search(source)
    assert match, (
        "styles.css no longer declares `--color-danger-overlay` as a literal "
        "oklch() triple. It is the input this section measures; expressed any "
        "other way the arithmetic below silently stops covering the shipped "
        "colour. Keep the literal, or re-point this reader deliberately."
    )
    return float(match.group(1)), float(match.group(2)), float(match.group(3))


def test_the_contrast_model_reproduces_the_browser_measurement() -> None:
    """Calibration. Without this the arithmetic below is unfalsifiable.

    The browser measured the OLD fill at 2.53:1 over the bright backdrop. This
    model must reach the same number from the same inputs, or every other
    assertion in this section is measuring something the renderer does not do.
    """
    old_fill = _oklch_to_srgb(0.70, 0.17, 25)
    assert old_fill == (246, 109, 103), (
        f"OKLCH->sRGB gives {old_fill} for --color-danger; Chromium gives "
        "(246, 109, 103) read back out of a canvas. The model has drifted "
        "from the renderer and the ratios below no longer describe it."
    )
    measured = _contrast(GLYPH_WHITE, _composite(old_fill, 0.80, BACKDROP_BRIGHT))
    assert round(measured, 2) == 2.53, (
        f"the LANE-33 defect reproduces at {measured:.2f}:1, not the 2.53:1 "
        "measured in the browser."
    )
    assert round(_contrast(GLYPH_WHITE, old_fill), 2) == 2.88, (
        "the old hue must still fail FULLY OPAQUE (2.88:1) - that is why this "
        "was never an alpha problem and why raising the alpha was not the fix."
    )


@pytest.mark.parametrize(
    ("where", "backdrop", "alpha"),
    [
        ("resting, over pure white", BACKDROP_WHITE, 0.88),
        ("resting, over the brightest fixture tile", BACKDROP_BRIGHT, 0.88),
        ("resting, over a near-black tile", BACKDROP_DARK, 0.88),
        ("hover (fully opaque), over the brightest fixture tile", BACKDROP_BRIGHT, 1.0),
        ("hover (fully opaque), over pure white", BACKDROP_WHITE, 1.0),
    ],
)
def test_a_danger_control_on_media_clears_3_to_1_at_both_backdrop_extremes(
    where: str, backdrop: tuple[int, int, int], alpha: float,
) -> None:
    """The property, computed from the shipped token - not a literal pin."""
    fill = _oklch_to_srgb(*_overlay_token())
    ratio = _contrast(GLYPH_WHITE, _composite(fill, alpha, backdrop))
    assert ratio >= WCAG_NON_TEXT_MINIMUM, (
        f"--color-danger-overlay resolves to rgb{fill} and measures "
        f"{ratio:.2f}:1 against its white glyph {where} - under the "
        f"{WCAG_NON_TEXT_MINIMUM}:1 WCAG 1.4.11 asks of a non-text UI "
        "component. This control is composited over photo pixels nobody "
        "chose, so the bright end is the one that governs. Darken the token "
        "or raise the alpha - but a delete control that stops reading as "
        "dangerous is a worse outcome than the contrast it fixes, so move the "
        "lightness and keep the hue."
    )


def test_the_overlay_token_is_not_overridden_by_the_light_theme() -> None:
    """Theme-independent on purpose: the glyph over it is a fixed white."""
    source = (FRONTEND_SRC / "styles.css").read_text(encoding="utf-8")
    marker = 'html[data-theme="light"]'
    start = source.find(marker)
    assert start > 0, "styles.css no longer carries the light-theme block"
    assert "--color-danger-overlay" not in source[start:], (
        "the light theme now overrides --color-danger-overlay. The glyph on "
        "these controls is a fixed `white`, so a fill that flips with the "
        "theme ships white-on-light - the `.hps-pill` precedent, and exactly "
        "the mix the tile-action pairing test above forbids. If the light "
        "theme really needs its own fill, the glyph has to flip with it and "
        "BOTH ends need measuring again."
    )


@pytest.mark.parametrize("rel_path", DANGER_OVERLAY_TEMPLATES)
def test_no_danger_control_on_media_uses_the_translucent_base_danger(
    rel_path: str,
) -> None:
    path = FRONTEND_SRC / rel_path
    assert path.exists(), f"{rel_path} moved; re-point this guard"
    source = path.read_text(encoding="utf-8")
    offenders = [
        f"{lineno}: {line.strip()[:110]}"
        for lineno, line in enumerate(source.splitlines(), 1)
        if TRANSLUCENT_DANGER_FILL.search(line) and "text-white" in line
    ]
    assert not offenders, (
        f"{rel_path} puts a `bg-danger/<alpha>` fill under a fixed white glyph "
        "on a control that sits over media. That is the LANE-33 defect: "
        "rgb(246,109,103) is 2.53:1 over a near-white photo and 2.88:1 even "
        "opaque. Use `bg-danger-overlay/88` + `hover:bg-danger-overlay`, "
        "which measures 4.47-6.38:1 across the backdrop range.\n  "
        + "\n  ".join(offenders)
    )
    # Prove the negative: the check above is fully satisfied by a file that no
    # longer has the control at all.
    assert DANGER_OVERLAY_FILL in source, (
        f"{rel_path} no longer declares {DANGER_OVERLAY_FILL!r}. Either the "
        "delete control was removed - in which case drop the file from "
        "DANGER_OVERLAY_TEMPLATES deliberately - or it was restyled and the "
        "pattern check above is now guarding nothing."
    )
