"""LANE-143 (the user's correction 1): every KPI rail tile carries an icon.

Static scan of the five rail templates the spec names: every ``<app-kpi-tile``
binds ``icon=`` / ``[icon]=``, and every raw ``<div class="kpi compact">`` copy on
the Training estimate wall and the LIVE ESTIMATE rail holds a gated
``kpi-icon`` element. Modal copies (analyze, tag analytics, LoRA tools,
training stats) are not rails and are not scanned.

The icon itself renders only under ``data-ux="v2"`` (``KpiTileComponent`` and the
raw copies gate it on ``UxStore.v2()``); the shot script counts the painted
icons per screen (26 ON, 0 OFF).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APP = REPO_ROOT / "frontend" / "src" / "app"

RAILS = {
    "screens/datasets-screen/datasets-screen.html": 6,
    "screens/server-screen/server-screen.html": 4,
    "screens/jobs-screen/jobs-screen.html": 6,
    "screens/projects-screen/projects-screen.html": 4,
}
RAW = {
    "components/training/estimate-wall/estimate-wall.html": 5,
    "components/training/training-estimate-rail/training-estimate-rail.ts": 1,
}
TOTAL = 26

_TILE = re.compile(r"<app-kpi-tile\b(.*?)/?>", re.S)
_LABEL = re.compile(r'\blabel="([^"]*)"')
_ICON = re.compile(r'(?:\s|^)(?:icon|\[icon\])="[^"]+"')


def tiles_without_icon(name: str, text: str) -> list[str]:
    bad = []
    for m in _TILE.finditer(text):
        attrs = m.group(1)
        if not _ICON.search(attrs):
            label = _LABEL.search(attrs)
            bad.append(f'{name} tile "{label.group(1) if label else "?"}" carries no icon')
    return bad


def raw_tiles(text: str) -> list[str]:
    """The bodies of every ``<div class="kpi compact">`` (balanced ``div`` count)."""
    out = []
    for m in re.finditer(r'<div class="kpi compact">', text):
        depth, i = 1, m.end()
        while depth and i < len(text):
            nxt = re.search(r"<div\b|</div>", text[i:])
            if not nxt:
                break
            depth += 1 if nxt.group(0) == "<div" else -1
            i += nxt.end()
        out.append(text[m.end() : i])
    return out


def raw_without_icon(name: str, text: str) -> list[str]:
    bad = []
    for body in raw_tiles(text):
        label = re.search(r'class="kpi-label">([^<]*)<', body)
        if 'class="kpi-icon"' not in body or "<app-ico" not in body:
            bad.append(f'{name} tile "{label.group(1) if label else "?"}" carries no kpi-icon')
    return bad


def _read(rel: str) -> str:
    return (APP / rel).read_text(encoding="utf-8")


@pytest.mark.parametrize(("rel", "count"), list(RAILS.items()))
def test_every_rail_tile_carries_an_icon(rel: str, count: int) -> None:
    text = _read(rel)
    n = len(_TILE.findall(text))
    assert n == count, f"LANE-143: expected {count} KPI tiles in {rel}, found {n}"
    bad = tiles_without_icon(Path(rel).name, text)
    assert not bad, "LANE-143: " + "; ".join(bad)


@pytest.mark.parametrize(("rel", "count"), list(RAW.items()))
def test_every_raw_compact_tile_carries_a_gated_icon(rel: str, count: int) -> None:
    text = _read(rel)
    bodies = raw_tiles(text)
    assert len(bodies) == count, f"LANE-143: expected {count} raw .kpi.compact tiles in {rel}, found {len(bodies)}"
    bad = raw_without_icon(Path(rel).name, text)
    assert not bad, "LANE-143: " + "; ".join(bad)
    # the hand-written slot renders only under v2, like the component's
    assert len(re.findall(r"@if \(ux\.v2\(\)\)\s*\{\s*<span class=\"kpi-icon\"", text)) == count, (
        f"LANE-143: every kpi-icon in {rel} must be gated by @if (ux.v2())"
    )


def test_the_rails_total_26_tiles() -> None:
    total = sum(RAILS.values()) + sum(RAW.values())
    assert total == TOTAL, f"LANE-143: the rails hold {TOTAL} tiles, the table says {total}"


def test_the_scan_catches_a_tile_without_an_icon() -> None:
    """Positive control: a planted tile without an icon, and a raw tile without
    the square, are both named."""
    planted = '<app-kpi-tile label="CAPTIONED" [value]="1" accent="success"/>\n<app-kpi-tile label="X" icon="Box" [value]="1"/>'
    assert tiles_without_icon("t.html", planted) == ['t.html tile "CAPTIONED" carries no icon']
    raw = '<div class="kpi compact"><div class="kpi-accent"></div><div class="kpi-label">WALL TIME</div></div>'
    assert raw_without_icon("w.html", raw) == ['w.html tile "WALL TIME" carries no kpi-icon']
