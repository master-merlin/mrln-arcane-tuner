"""A clip whose length the frame rule cuts says so (LANE-92 real-job run,
check 17/18: a 97-frame clip trained as 90 frames and no line said it).

Production caller: ``pipeline_data.py:prepare_data`` — the one place a clip's
usable frame count meets the frame ladder (``get_bucket_for_video``). When the
ladder hands back FEWER frames than the clip could supply under the run's frame
cap, the difference is the frame RULE's doing (``17n+5``: 97 -> 90; ``4n+1``:
50 -> 49), and ``prepare_data`` logs one ``clip_frames_snapped`` line per clip
and counts them as ``snapped_clips`` on the ``data_prepared`` summary — the
mirror of ``short_clip_skipped`` / ``skipped_short_clips``.

A clip cut only by the CAP (the run's ``num_frames`` or the family ceiling) is
not a snap: the cap is a setting the user chose, and that path stays exactly as
quiet as it was.

These tests drive the REAL ``prepare_data`` through the short-clip suite's
harness (the dataset API faked at the HTTP client).
"""

from __future__ import annotations

from app.engine.tests.test_pipeline_data_short_clips import H3, WAN, _pair, _pipeline, _prepare

EVENT = "clip_frames_snapped"


def _snap_lines(t) -> list:
    return [c for c in t.logger.info.call_args_list if c.args and c.args[0] == EVENT]


def _snapped_count(t) -> int | None:
    lines = [c for c in t.logger.info.call_args_list if c.args and c.args[0] == "data_prepared"]
    assert lines, "no data_prepared summary line"
    return lines[-1].kwargs.get("snapped_clips")


def test_97_frame_clip_under_17n_plus_5_says_it_uses_90(monkeypatch, tmp_path):
    t = _pipeline(H3, monkeypatch, tmp_path, [_pair("clip97.mp4", 97)])
    items = _prepare(t)
    assert items["clip97.mp4"]["target_frames"] == 90
    lines = _snap_lines(t)
    assert len(lines) == 1, f"expected one {EVENT} line for the 97 -> 90 cut, got {len(lines)}"
    kw = lines[0].kwargs
    assert (kw.get("media"), kw.get("source_frames"), kw.get("used_frames"), kw.get("frame_rule")) == (
        "clip97.mp4",
        97,
        90,
        "17n+5",
    ), kw
    assert kw.get("dataset") == "ds"
    assert _snapped_count(t) == 1


def test_a_clip_already_on_the_rule_logs_nothing(monkeypatch, tmp_path):
    t = _pipeline(H3, monkeypatch, tmp_path, [_pair("clip90.mp4", 90)])
    items = _prepare(t)
    assert items["clip90.mp4"]["target_frames"] == 90
    assert _snap_lines(t) == []
    assert _snapped_count(t) == 0


def test_the_count_is_per_clip(monkeypatch, tmp_path):
    pairs = [_pair("a.mp4", 97), _pair("b.mp4", 90), _pair("c.mp4", 30)]
    t = _pipeline(H3, monkeypatch, tmp_path, pairs)
    _prepare(t)
    assert sorted(c.kwargs["media"] for c in _snap_lines(t)) == ["a.mp4", "c.mp4"]
    assert _snapped_count(t) == 2


def test_4n_plus_1_family_reports_its_own_rule_cut(monkeypatch, tmp_path):
    """The event is the shared pipeline's, not one family's: 50 -> 49 under
    `4n+1`, and the target frame count is what it always was."""
    t = _pipeline(WAN, monkeypatch, tmp_path, [_pair("clip50.mp4", 50)])
    items = _prepare(t)
    assert items["clip50.mp4"]["target_frames"] == 49
    lines = _snap_lines(t)
    assert len(lines) == 1
    kw = lines[0].kwargs
    assert (kw.get("source_frames"), kw.get("used_frames"), kw.get("frame_rule")) == (50, 49, "4n+1"), kw


def test_a_clip_cut_only_by_the_frame_cap_is_not_a_snap(monkeypatch, tmp_path):
    """200 frames under the `4n+1` ceiling of 81: the cap is a setting, not the
    rule rounding — no line, as before."""
    t = _pipeline(WAN, monkeypatch, tmp_path, [_pair("long.mp4", 200)])
    items = _prepare(t)
    assert items["long.mp4"]["target_frames"] == 81
    assert _snap_lines(t) == []
    assert _snapped_count(t) == 0


def test_the_per_dataset_cap_is_the_cap_that_counts(monkeypatch, tmp_path):
    """A dataset capped at 90 frames: the 97-frame clip is cut by the CAP to a
    legal 90, the rule rounds nothing."""
    t = _pipeline(H3, monkeypatch, tmp_path, [_pair("clip97.mp4", 97)])
    t.config["datasets"] = [{"dataset_name": "ds", "num_frames": 90}]
    items = _prepare(t)
    assert items["clip97.mp4"]["target_frames"] == 90
    assert _snap_lines(t) == []
