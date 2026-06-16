"""LTX-2 _batch_frame_rate reads the effective target_fps (no GPU, no weights)."""

from __future__ import annotations

from app.engine.models.families.ltx2.driver import Ltx2Driver


def _driver():
    d = object.__new__(Ltx2Driver)
    d.frame_rate = 24.0  # native default
    return d


def test_reads_batch_target_fps_when_present():
    d = _driver()
    assert d._batch_frame_rate({"target_fps": 12.0}) == 12.0


def test_falls_back_to_native_when_absent():
    d = _driver()
    assert d._batch_frame_rate({}) == 24.0


def test_handles_list_form():
    d = _driver()
    assert d._batch_frame_rate({"target_fps": [12.0, 12.0]}) == 12.0
