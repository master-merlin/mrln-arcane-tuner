"""Axis A tiled — window enumeration math + inventory emission (no GPU)."""

from __future__ import annotations

from app.engine.core.pipeline.pipeline_data import PipelineDataMixin


class _Host(PipelineDataMixin):
    pass


def _win(trim_start, end_s, *, window_span_s, overlap, max_windows):
    # window_span_s = the seconds a single window must cover (target_frames / eff_fps)
    return PipelineDataMixin._compute_tiled_windows(
        _Host(),
        trim_start_s=trim_start,
        end_s=end_s,
        window_span_s=window_span_s,
        overlap=overlap,
        max_windows=max_windows,
    )


def test_abutting_windows_partition_the_clip():
    # 4s clip, 1s windows, no overlap → 4 abutting windows.
    wins = _win(0.0, 4.0, window_span_s=1.0, overlap=0.0, max_windows=10)
    assert wins == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]


def test_overlap_steps_by_one_minus_overlap():
    # 1s window, 0.5 overlap → step 0.5s. 2s clip → starts 0.0,0.5,1.0 (end<=2.0).
    wins = _win(0.0, 2.0, window_span_s=1.0, overlap=0.5, max_windows=10)
    assert wins == [(0.0, 1.0), (0.5, 1.5), (1.0, 2.0)]


def test_max_windows_caps_count():
    wins = _win(0.0, 100.0, window_span_s=1.0, overlap=0.0, max_windows=3)
    assert wins == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]


def test_trim_start_offset_respected():
    wins = _win(2.0, 5.0, window_span_s=1.0, overlap=0.0, max_windows=10)
    assert wins == [(2.0, 3.0), (3.0, 4.0), (4.0, 5.0)]


def test_clip_shorter_than_one_window_yields_no_windows():
    # Usable < window_span → CANNOT supply target_frames at eff_fps, so emit
    # nothing here; the caller (_emit_temporal_items) falls back to the single
    # legacy window. (A sub-span window would crash load_clip — punch-list #2.)
    wins = _win(0.0, 0.6, window_span_s=1.0, overlap=0.0, max_windows=10)
    assert wins == []


def test_no_window_ever_ends_past_clip():
    wins = _win(0.0, 2.3, window_span_s=1.0, overlap=0.0, max_windows=10)
    assert all(end <= 2.3 + 1e-6 for _, end in wins)
    assert len(wins) == 2  # [0,1],[1,2]; the trailing 0.3s partial is dropped
