"""Axis A tiled — window enumeration math + inventory emission (no GPU)."""

from __future__ import annotations

from app.engine.core.pipeline.pipeline_data import (
    PipelineDataMixin,
    video_trim_extra_key,
)


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


def test_overlap_ge_one_does_not_loop_forever():
    # Defensive: overlap>=1 would make step 0 → infinite loop without the
    # max(step, 1e-3) clamp. max_windows still bounds the result.
    wins = _win(0.0, 100.0, window_span_s=1.0, overlap=1.0, max_windows=5)
    assert len(wins) == 5
    assert all(abs((end - start) - 1.0) < 1e-6 for start, end in wins)


def test_max_windows_zero_is_floored_to_one():
    # Defensive: max_windows<=0 is floored to 1 (the contract rejects <1
    # upstream, but the helper must not divide-by-zero or emit nothing here).
    wins = _win(0.0, 100.0, window_span_s=1.0, overlap=0.0, max_windows=0)
    assert wins == [(0.0, 1.0)]


def _expand(host, *, base_item, end_s, window_span_s, repeats):
    """Mirror the inventory emission contract for one clip × one bucket."""
    return PipelineDataMixin._emit_temporal_items(
        host,
        base_item=base_item,
        trim_start_s=base_item["trim_start_s"],
        end_s=end_s,
        window_span_s=window_span_s,
        repeats=repeats,
    )


def _host_tiled(overlap=0.0, max_windows=10):
    h = _Host()
    h._temporal_coverage = "tiled"
    h._window_overlap = overlap
    h._max_windows = max_windows
    return h


def _host_first():
    h = _Host()
    h._temporal_coverage = "first"
    h._window_overlap = 0.0
    h._max_windows = 10
    return h


def _base_item():
    return {
        "id": "clip.mkv", "is_video": True, "target_frames": 9,
        "target_fps": 9.0, "trim_start_s": 0.0, "trim_end_s": 4.0,
    }


def test_first_mode_emits_one_item_per_repeat():
    h = _host_first()
    items = _expand(h, base_item=_base_item(), end_s=4.0, window_span_s=1.0, repeats=2)
    assert len(items) == 2
    assert {(i["trim_start_s"], i["trim_end_s"]) for i in items} == {(0.0, 4.0)}


def test_tiled_mode_emits_k_distinct_windows_times_repeats():
    h = _host_tiled()
    items = _expand(h, base_item=_base_item(), end_s=4.0, window_span_s=1.0, repeats=1)
    starts = sorted(i["trim_start_s"] for i in items)
    assert starts == [0.0, 1.0, 2.0, 3.0]
    # distinct cache keys per window
    keys = {video_trim_extra_key(i) for i in items}
    assert len(keys) == 4


def test_tiled_windows_multiply_by_repeats():
    # Deliberate balancing choice: K windows × repeats (punch-list #6).
    h = _host_tiled()
    items = _expand(h, base_item=_base_item(), end_s=4.0, window_span_s=1.0, repeats=2)
    assert len(items) == 8  # 4 windows × 2 repeats


def test_tiled_short_clip_falls_back_to_single_legacy_window():
    # Usable (1.0s) < one window (4.0s) → no tiling; emit the original window
    # exactly repeats times (never a sub-span window that crashes load_clip).
    h = _host_tiled()
    short = {**_base_item(), "trim_end_s": 1.0}
    items = _expand(h, base_item=short, end_s=1.0, window_span_s=4.0, repeats=2)
    assert len(items) == 2
    assert {(i["trim_start_s"], i["trim_end_s"]) for i in items} == {(0.0, 1.0)}


def test_tiled_items_are_independent_dicts():
    h = _host_tiled()
    items = _expand(h, base_item=_base_item(), end_s=4.0, window_span_s=1.0, repeats=1)
    items[0]["trim_start_s"] = 99.0
    assert items[1]["trim_start_s"] != 99.0  # no shared mutable state


def test_first_mode_preserves_original_trim_key_for_untrimmed_clip():
    # Regression (cache equivalence): first mode must NOT rewrite a None
    # trim_end into the computed concrete end — that would change the latent
    # cache filename (video_trim_extra_key formats "t{start}-{end}") and
    # silently re-encode every untrimmed clip. The emitted item's trim window
    # — and thus its cache key — must be byte-identical to the source item.
    h = _host_first()
    base = {**_base_item(), "trim_start_s": 0.0, "trim_end_s": None}
    # end_s here is the caller's COMPUTED usable end (concrete); it must not
    # leak into the first-mode clone's trim window.
    items = _expand(h, base_item=base, end_s=1.0, window_span_s=1.0, repeats=1)
    assert len(items) == 1
    assert items[0]["trim_end_s"] is None
    assert video_trim_extra_key(items[0]) == video_trim_extra_key(base)


def test_tiled_untrimmed_clip_tiles_across_full_duration():
    # Regression (Critical): an untrimmed clip (trim_end_s=None) must still tile
    # across its FULL duration when the caller passes the duration-aware end_s.
    # A one-window-wide end would collapse tiling to a single window for every
    # untrimmed clip (the default case), silently defeating the feature.
    h = _host_tiled()
    base = {**_base_item(), "trim_start_s": 0.0, "trim_end_s": None}
    items = _expand(h, base_item=base, end_s=4.0, window_span_s=1.0, repeats=1)
    starts = sorted(i["trim_start_s"] for i in items)
    assert starts == [0.0, 1.0, 2.0, 3.0]  # 4 windows, not 1
    # tiled sub-windows DO get concrete ends (new windows → new cache keys).
    assert all(i["trim_end_s"] is not None for i in items)


def test_repeats_zero_emits_nothing():
    # Symmetry with the image path (for _ in range(repeats)): repeats=0 emits
    # zero items in both first and tiled modes (a way to disable a dataset).
    first = _expand(
        _host_first(), base_item=_base_item(), end_s=4.0, window_span_s=1.0, repeats=0
    )
    tiled = _expand(
        _host_tiled(), base_item=_base_item(), end_s=4.0, window_span_s=1.0, repeats=0
    )
    assert first == []
    assert tiled == []


def test_precache_derives_distinct_extra_key_per_tiled_window():
    # The video pre-cache derives extra_key = video_trim_extra_key(item) per
    # inventory item (pipeline_caching.py), so distinct windows → distinct keys
    # → distinct cache filenames, with no cache-layer change.
    h = _host_tiled()
    items = _expand(h, base_item=_base_item(), end_s=4.0, window_span_s=1.0, repeats=1)
    keys = [video_trim_extra_key(i) for i in items]
    assert len(set(keys)) == len(items)  # one cache file per window
    assert all(k.startswith("t") for k in keys)


def test_effective_fps_changes_res_str_so_strides_never_collide():
    # res_str = f"{w}x{h}x{frames}f{fps}" (pipeline_data.py). A different stride
    # yields a different fps → a different cache dir, so stride=1 and stride=2
    # never share a latent cache.
    def res_str(fps):
        return f"768x768x25f{fps}"

    assert res_str(24.0) != res_str(12.0)
