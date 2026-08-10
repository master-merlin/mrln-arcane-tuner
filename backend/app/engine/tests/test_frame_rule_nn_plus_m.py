"""Nn+M frame-rule generalization.

Existing families (4n+1, 8n+1) must behave EXACTLY as before — this is a
widening, not a change. MiniMax H3 needs 17n+5, which the Nn+1-only parser
cannot express.
"""

from __future__ import annotations

import pytest

from app.engine.components.bucketing import BucketManager
from app.engine.core.video_contract import frame_predicate, snap_frames


@pytest.mark.parametrize("rule,frames,valid", [
    # --- existing families: UNCHANGED behaviour (regression guard) ---
    ("4n+1", 1, True), ("4n+1", 5, True), ("4n+1", 81, True), ("4n+1", 4, False),
    ("8n+1", 1, True), ("8n+1", 9, True), ("8n+1", 121, True), ("8n+1", 8, False),
    # --- MiniMax H3 ---
    ("17n+5", 5, True), ("17n+5", 22, True), ("17n+5", 107, True),
    ("17n+5", 124, True), ("17n+5", 345, True),
    ("17n+5", 97, False),   # our ORIGINAL default — invalid, 97 % 17 == 12
    ("17n+5", 1, False),    # H3's floor is 5, not 1
    ("17n+5", 106, False),
])
def test_frame_predicate_handles_nn_plus_m(rule, frames, valid):
    assert frame_predicate(rule)(frames) is valid


@pytest.mark.parametrize("rule,expected", [
    ("4n+1", (4, 1)),
    ("8n+1", (8, 1)),
    ("17n+5", (17, 5)),
])
def test_parse_frame_step_returns_step_and_offset(rule, expected):
    assert BucketManager._parse_frame_step(rule) == expected


def test_existing_ladders_are_byte_identical_to_today():
    """Hard regression guard: the two shipped rules must not move at all."""
    assert BucketManager.frame_ladder(81, "4n+1") == list(range(1, 82, 4))
    assert BucketManager.frame_ladder(121, "8n+1") == list(range(1, 122, 8))


def test_h3_ladder_anchors_at_5_not_1():
    ladder = BucketManager.frame_ladder(124, "17n+5")
    assert ladder[0] == 5, f"H3 ladder must start at 5, got {ladder[0]}"
    assert 107 in ladder and 124 in ladder
    assert all((f - 5) % 17 == 0 for f in ladder)


@pytest.mark.parametrize("rule,raw,snapped", [
    ("4n+1", 100, 97),
    ("17n+5", 120, 107),
    ("17n+5", 350, 345),
])
def test_snap_frames_rounds_down_to_a_valid_count(rule, raw, snapped):
    assert snap_frames(raw, rule) == snapped
