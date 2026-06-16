"""Axis B frame stride — effective fps math + batch wiring (no GPU)."""

from __future__ import annotations

from app.engine.core.pipeline.pipeline_data import PipelineDataMixin


class _Host(PipelineDataMixin):
    pass


def _host(config):
    h = _Host()
    h.config = config
    return h


def test_effective_fps_divides_native_by_stride():
    h = _host({"frame_stride": 2})
    # native 24 fps, stride 2 → effective 12 fps
    assert h._effective_fps(native_or_target_fps=24.0) == 12.0


def test_stride_one_is_identity():
    h = _host({"frame_stride": 1})
    assert h._effective_fps(native_or_target_fps=24.0) == 24.0


def test_zero_fps_stays_zero():
    h = _host({"frame_stride": 4})
    assert h._effective_fps(native_or_target_fps=0.0) == 0.0


def test_falsy_stride_coalesces_to_identity():
    # The `or 1` coalescing: a missing key, None, or 0 stride must behave as
    # stride 1 (identity), never divide-by-zero.
    assert _host({})._effective_fps(native_or_target_fps=24.0) == 24.0
    assert _host({"frame_stride": None})._effective_fps(24.0) == 24.0
    assert _host({"frame_stride": 0})._effective_fps(24.0) == 24.0
