"""Clip base-fps resolution is robust to a stringified target_fps.

The form layer can submit ``target_fps`` as the literal string ``"0"`` (numbers
arrive as strings). The non-empty string ``"0"`` is truthy, so the old
``config.get("target_fps") or None`` kept it, and ``float("0")`` then collapsed
the base fps to 0.0 → the clip loader raised ``target_fps must be > 0`` and
pre-cache failed (G2/G4/G1-stills). Integer ``0`` was falsy and fell through to
the clip's real fps, so those jobs passed — the bug was purely the string form.
"""

from __future__ import annotations

from app.engine.core.pipeline.pipeline_data import (
    _coerce_fps,
    _resolve_clip_base_fps,
)


class TestCoerceFps:
    def test_string_zero_is_zero_not_truthy(self):
        # The crux: "0" must coerce to 0.0 (unset), not survive as truthy.
        assert _coerce_fps("0") == 0.0

    def test_int_and_float_zero(self):
        assert _coerce_fps(0) == 0.0
        assert _coerce_fps(0.0) == 0.0

    def test_positive_string_and_number(self):
        assert _coerce_fps("24") == 24.0
        assert _coerce_fps(23.976) == 23.976

    def test_none_and_junk_and_negative_are_zero(self):
        assert _coerce_fps(None) == 0.0
        assert _coerce_fps("garbage") == 0.0
        assert _coerce_fps("-5") == 0.0


class TestResolveClipBaseFps:
    def test_string_zero_target_falls_through_to_clip_fps(self):
        # The actual regression: stringified "0" must NOT zero out the rate;
        # the clip's own fps wins.
        assert _resolve_clip_base_fps("0", 23.976, 24.0) == 23.976

    def test_int_zero_target_falls_through_to_clip_fps(self):
        assert _resolve_clip_base_fps(0, 23.976, 24.0) == 23.976

    def test_none_target_uses_clip_fps(self):
        assert _resolve_clip_base_fps(None, 23.976, 24.0) == 23.976

    def test_explicit_positive_target_wins(self):
        assert _resolve_clip_base_fps("24", 0, 24.0) == 24.0
        assert _resolve_clip_base_fps(12.0, 23.976, 24.0) == 12.0

    def test_missing_clip_fps_falls_back_to_model_native(self):
        # Freshly split segment with no probed fps → model native (LTX-2 / WAN).
        assert _resolve_clip_base_fps(0, 0, 24.0) == 24.0
        assert _resolve_clip_base_fps("0", 0, 16.0) == 16.0

    def test_clip_fps_as_string_is_coerced(self):
        assert _resolve_clip_base_fps(0, "23.976", 24.0) == 23.976
