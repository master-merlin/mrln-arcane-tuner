"""Cutlist parsing — LosslessCut .llc (JSON5), CSV, TSV; clamping + warnings."""

import json

from app.core.video.cutlist import Segment, parse_cutlist


def _starts(result):
    return [round(s.start_s, 3) for s in result.segments]


def _ends(result):
    return [round(s.end_s, 3) for s in result.segments]


# ── .llc (LosslessCut JSON5) ────────────────────────────────────────────────


def test_llc_cutsegments_basic():
    body = b"""{
        cutSegments: [
            {start: 1.0, end: 3.0, name: "intro"},
            {start: 5.5, end: 9.0, name: "action"},
        ],
    }"""  # trailing commas + unquoted keys → needs JSON5
    res = parse_cutlist(body, "project.llc", source_duration_s=60.0)
    assert res.format == "llc"
    assert _starts(res) == [1.0, 5.5]
    assert _ends(res) == [3.0, 9.0]
    assert [s.label for s in res.segments] == ["intro", "action"]
    assert res.warnings == []


def test_llc_valid_json():
    obj = {"cutSegments": [{"start": 0.0, "end": 2.5}, {"start": 2.5, "end": 5.0}]}
    res = parse_cutlist(json.dumps(obj).encode(), "x.llc", 10.0)
    assert res.format == "llc"
    assert _starts(res) == [0.0, 2.5]
    assert [s.label for s in res.segments] == [None, None]


def test_llc_missing_end_falls_back_to_duration():
    obj = {"cutSegments": [{"start": 4.0}]}  # no end
    res = parse_cutlist(json.dumps(obj).encode(), "x.llc", source_duration_s=12.0)
    assert len(res.segments) == 1
    assert res.segments[0].start_s == 4.0
    assert res.segments[0].end_s == 12.0


def test_llc_missing_name_is_none():
    obj = {"cutSegments": [{"start": 1.0, "end": 2.0}]}
    res = parse_cutlist(json.dumps(obj).encode(), "x.llc", 10.0)
    assert res.segments[0].label is None


def test_llc_clamps_to_duration():
    obj = {"cutSegments": [{"start": 1.0, "end": 999.0}]}
    res = parse_cutlist(json.dumps(obj).encode(), "x.llc", source_duration_s=10.0)
    assert res.segments[0].end_s == 10.0


def test_llc_drops_zero_length_with_warning():
    obj = {"cutSegments": [{"start": 5.0, "end": 5.0}, {"start": 1.0, "end": 2.0}]}
    res = parse_cutlist(json.dumps(obj).encode(), "x.llc", 10.0)
    assert _starts(res) == [1.0]
    assert len(res.warnings) == 1
    assert "zero/negative" in res.warnings[0]


def test_llc_key_drift_from_to_label():
    # A variant exporter using from/to/tag instead of start/end/name.
    obj = [{"from": 2.0, "to": 4.0, "tag": "v"}]
    res = parse_cutlist(json.dumps(obj).encode(), "x.llc", 10.0)
    assert _starts(res) == [2.0]
    assert res.segments[0].label == "v"


def test_llc_malformed_returns_warning_not_exception():
    res = parse_cutlist(b"{not valid json at all ::::", "x.llc", 10.0)
    assert res.segments == []
    assert res.warnings  # collected, not raised


# ── CSV / TSV ────────────────────────────────────────────────────────────────


def test_csv_with_end_and_label():
    body = b"start,end,label\n1.0,2.0,a\n3.0,4.5,b\n"
    res = parse_cutlist(body, "cuts.csv", 10.0)
    assert res.format == "csv"
    assert _starts(res) == [1.0, 3.0]
    assert _ends(res) == [2.0, 4.5]
    assert [s.label for s in res.segments] == ["a", "b"]


def test_csv_without_end_uses_duration():
    body = b"1.5\n"
    res = parse_cutlist(body, "cuts.csv", source_duration_s=8.0)
    assert len(res.segments) == 1
    assert res.segments[0].start_s == 1.5
    assert res.segments[0].end_s == 8.0


def test_csv_no_header():
    body = b"0.0,1.0\n1.0,2.0\n"
    res = parse_cutlist(body, "cuts.csv", 10.0)
    assert _starts(res) == [0.0, 1.0]


def test_tsv_basic():
    body = b"start\tend\tlabel\n0.0\t1.0\tfoo\n"
    res = parse_cutlist(body, "cuts.tsv", 10.0)
    assert res.format == "tsv"
    assert _starts(res) == [0.0]
    assert res.segments[0].label == "foo"


def test_csv_malformed_row_warns_not_raises():
    body = b"start,end\nNOTANUMBER,2.0\n1.0,3.0\n"
    res = parse_cutlist(body, "cuts.csv", 10.0)
    # First data row has a bad start → warning; second row parses.
    assert _starts(res) == [1.0]
    assert any("missing/invalid start" in w for w in res.warnings)


def test_csv_clamps_negative_start():
    body = b"-2.0,3.0\n"
    res = parse_cutlist(body, "cuts.csv", 10.0)
    assert res.segments[0].start_s == 0.0


def test_segment_model_shape():
    s = Segment(start_s=1.0, end_s=2.0, label="x")
    assert s.model_dump() == {"start_s": 1.0, "end_s": 2.0, "label": "x"}
