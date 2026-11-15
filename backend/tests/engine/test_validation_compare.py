"""Comparing runs' held-out validation: refusals, the noise floor, intervals and verdicts."""
import json
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from app.engine.core import validation_compare as vc

CENTRES = [0.25, 0.75]
KEYS = [f"ds/img{i}.png" for i in range(6)]


def _split(**changes):
    content = {
        "holdout": KEYS,
        "representatives": [
            {"key": key, "path": f"/data/{key}", "caption": "", "target_w": 64, "target_h": 64, "sha256": key}
            for key in KEYS
        ],
        "split_seed": 0,
        "global_triggerword": "",
        "noise_draws": 1,
        **changes,
    }
    return json.dumps(content, indent=2)


def _v2(step, losses, seed=None, keys=None, centres=CENTRES):
    """A record as the pass's ``_record`` builds it, from ``[image][draw][band]`` losses."""
    keys = keys or KEYS[: len(losses)]
    draws = len(losses[0])
    band_loss = [sum(sum(image[d][b] for d in range(draws)) / draws for image in losses) / len(losses)
                 for b in range(len(centres))]
    return {
        "step": step, "images": len(keys), "band_centres": centres, "band_loss": band_loss,
        "mean": sum(band_loss) / len(band_loss), "worst_band": band_loss.index(max(band_loss)), "seconds": 1.0,
        "version": 2, "image_keys": keys, "noise_draws": draws, "train_seed": seed, "image_band_loss": losses,
    }


def _v1(step, band_loss, centres=CENTRES):
    return {"step": step, "images": 3, "band_centres": centres, "band_loss": band_loss,
            "mean": sum(band_loss) / len(band_loss), "worst_band": 0, "seconds": 1.0}


def _write(folder: Path, records, split=None):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / vc.METRICS_FILE).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    (folder / vc.SPLIT_FILE).write_text(split or _split(), encoding="utf-8")
    return folder


def _losses(values, images=6, draws=1):
    """Every image and draw gets ``values`` per band, plus the image's own offset."""
    return [[[v + 0.01 * i for v in values] for _ in range(draws)] for i in range(images)]


def _shifted(per_image_band, draws=1):
    return [[list(bands) for _ in range(draws)] for bands in per_image_band]


def _runs(tmp_path, spec):
    """``spec``: {name: (arm, records)} → [(arm, folder)]."""
    return [(arm, _write(tmp_path / name, records)) for name, (arm, records) in spec.items()]


def _overall(report, a="A", b="B"):
    pair = next(p for p in report["pairs"] if (p["a"], p["b"]) == (a, b))
    return pair["overall"]


# ── Refusals ─────────────────────────────────────────────────────────────────


def test_runs_with_different_splits_are_refused_naming_the_files(tmp_path):
    a = _write(tmp_path / "a", [_v2(0, _losses([1, 1]), seed=1)])
    b = _write(tmp_path / "b", [_v2(0, _losses([1, 1]), seed=1)], split=_split(split_seed=9))
    with pytest.raises(vc.Refusal, match="validation_split.json"):
        vc.compare([("A", a), ("B", b)])


def test_runs_with_different_band_centres_are_refused(tmp_path):
    a = _write(tmp_path / "a", [_v2(0, _losses([1, 1]))])
    b = _write(tmp_path / "b", [_v2(0, _losses([1, 1]), centres=[0.2, 0.8])])
    with pytest.raises(vc.Refusal, match=r"band centres.*validation_metrics.jsonl|validation_metrics.jsonl.*band centres"):
        vc.compare([("A", a), ("B", b)])


def test_runs_that_measured_other_images_are_refused(tmp_path):
    a = _write(tmp_path / "a", [_v2(0, _losses([1, 1]))])
    keys = [*KEYS[:5], "ds/other.png"]
    b = _write(tmp_path / "b", [_v2(0, _losses([1, 1]), keys=keys)])
    with pytest.raises(vc.Refusal, match="other images"):
        vc.compare([("A", a), ("B", b)])


def test_a_file_mixing_band_grids_is_refused_naming_the_file_and_step(tmp_path):
    a = _write(tmp_path / "a", [_v2(0, _losses([1, 1])), _v2(500, _losses([1, 1, 1]), centres=[0.1, 0.5, 0.9])])
    b = _write(tmp_path / "b", [_v2(0, _losses([1, 1]))])
    with pytest.raises(vc.Refusal, match=r"a[\\/]validation_metrics.jsonl: step 500 .*mixed band grid"):
        vc.compare([("A", a), ("B", b)])


def test_a_step_missing_from_a_run_is_refused(tmp_path):
    a = _write(tmp_path / "a", [_v2(0, _losses([1, 1])), _v2(500, _losses([1, 1]))])
    b = _write(tmp_path / "b", [_v2(0, _losses([1, 1]))])
    with pytest.raises(vc.Refusal, match=r"b[\\/]validation_metrics.jsonl: no record at step 500"):
        vc.compare([("A", a), ("B", b)], step=500)


def test_a_repeated_step_is_refused_with_the_file_line_and_step(tmp_path):
    records = [_v2(s, _losses([1, 1])) for s in (0, 500, 1000, 500)]
    a = _write(tmp_path / "a", records)
    b = _write(tmp_path / "b", [_v2(0, _losses([1, 1]))])
    with pytest.raises(vc.Refusal) as refused:
        vc.compare([("A", a), ("B", b)])
    message = str(refused.value)
    assert "validation_metrics.jsonl: line 4 repeats step 500" in message
    assert vc.FRESH_FOLDER in message


def test_a_pass_written_twice_in_a_row_is_refused(tmp_path):
    # A run stopped right after its pass at 500 and resumed from checkpoint-000500 writes it again.
    a = _write(tmp_path / "a", [_v2(s, _losses([1, 1])) for s in (0, 500, 500)])
    b = _write(tmp_path / "b", [_v2(0, _losses([1, 1]))])
    with pytest.raises(vc.Refusal, match="line 3 repeats step 500"):
        vc.compare([("A", a), ("B", b)])


def test_a_rolled_back_resume_is_refused_with_the_file_line_and_step(tmp_path):
    records = [_v2(s, _losses([1, 1])) for s in (0, 1000, 1999, 1500, 1600)]
    a = _write(tmp_path / "a", records)
    b = _write(tmp_path / "b", [_v2(0, _losses([1, 1]))])
    with pytest.raises(vc.Refusal) as refused:
        vc.compare([("A", a), ("B", b)])
    message = str(refused.value)
    assert "validation_metrics.jsonl: line 4 goes back to step 1500 after step 1999" in message
    assert vc.FRESH_FOLDER in message


def test_a_single_arm_is_refused(tmp_path):
    a = _write(tmp_path / "a", [_v2(0, _losses([1, 1]), seed=1)])
    b = _write(tmp_path / "b", [_v2(0, _losses([1, 1]), seed=2)])
    with pytest.raises(vc.Refusal, match="at least two arms"):
        vc.compare([("A", a), ("A", b)])


def test_one_folder_given_twice_is_refused_also_under_another_spelling(tmp_path):
    a = _write(tmp_path / "a", [_v2(0, _losses([1, 1]))])
    (tmp_path / "x").mkdir()
    with pytest.raises(vc.Refusal, match="given twice"):
        vc.compare([("A", a), ("B", tmp_path / "x" / ".." / "a")])


def test_two_runs_of_one_arm_with_one_seed_are_refused(tmp_path):
    runs = _runs(tmp_path, {
        "a1": ("A", [_v2(0, _losses([1, 1]), seed=42)]),
        "a2": ("A", [_v2(0, _losses([1, 1]), seed=42)]),
        "b": ("B", [_v2(0, _losses([1, 1]), seed=42)]),
    })
    with pytest.raises(vc.Refusal, match="seed 42"):
        vc.compare(runs)


def test_two_unseeded_runs_are_not_refused_and_give_no_floor(tmp_path):
    runs = _runs(tmp_path, {
        "a1": ("A", [_v2(0, _losses([1, 1]))]),
        "a2": ("A", [_v2(0, _losses([1, 1]))]),
        "b": ("B", [_v2(0, _losses([2, 2]))]),
    })
    report = vc.compare(runs)
    assert report["floor"] is None
    assert any("has no recorded training seed" in reason for reason in report["reasons"])
    assert _overall(report)["verdict"].startswith("no verdict: ")


def test_an_unreadable_file_names_its_path(tmp_path):
    a = _write(tmp_path / "a", [_v2(0, _losses([1, 1]))])
    b = tmp_path / "b"
    b.mkdir()
    with pytest.raises(vc.Unreadable, match=r"b[\\/]validation_split.json"):
        vc.compare([("A", a), ("B", b)])


def test_a_line_that_is_not_a_record_is_invalid_naming_the_line(tmp_path):
    a = _write(tmp_path / "a", [_v2(0, _losses([1, 1]))])
    b = _write(tmp_path / "b", [_v2(0, _losses([1, 1]))])
    with open(b / vc.METRICS_FILE, "a", encoding="utf-8") as metrics:
        metrics.write('{"step": 1}\n')
    with pytest.raises(vc.Unreadable, match="line 2"):
        vc.compare([("A", a), ("B", b)])


def _diverged(tmp_path):
    """B's pass at step 500 has a loss the pass wrote as null (non-finite)."""
    bad = _v2(500, _losses([1, 1]))
    bad["image_band_loss"][0][0][0] = None
    a = _write(tmp_path / "a", [_v2(0, _losses([1, 1])), _v2(500, _losses([1, 1]))])
    b = _write(tmp_path / "b", [_v2(0, _losses([2, 2])), bad])
    return a, b


def test_a_non_finite_loss_at_another_step_still_compares(tmp_path):
    a, b = _diverged(tmp_path)
    report = vc.compare([("A", a), ("B", b)], step=0)
    assert _overall(report)["gap"] == pytest.approx(-1.0)


def test_a_non_finite_loss_at_the_compared_step_is_unreadable(tmp_path):
    a, b = _diverged(tmp_path)
    with pytest.raises(vc.Unreadable, match="step 500 has a non-finite loss"):
        vc.compare([("A", a), ("B", b)], step=500)


def _null_in_draw_1():
    """Two draws per image; the pass wrote image 0's draw-1 loss in band 0 as null."""
    record = _v2(0, _losses([1, 1], draws=2))
    record["image_band_loss"][0][1][0] = None
    return record


def test_a_non_finite_loss_in_an_unused_draw_still_compares(tmp_path):
    a = _write(tmp_path / "a", [_v2(0, _losses([2, 2]))])  # one draw: draw 1 isn't shared
    b = _write(tmp_path / "b", [_null_in_draw_1()])
    assert _overall(vc.compare([("A", a), ("B", b)]))["gap"] == pytest.approx(1.0)


def test_a_non_finite_loss_in_an_unused_draw_still_compares_with_older_records(tmp_path):
    a = _write(tmp_path / "a", [_v1(0, [2, 2])])
    b = _write(tmp_path / "b", [_null_in_draw_1()])
    vc.compare([("A", a), ("B", b)])


def test_a_non_finite_loss_in_a_shared_draw_is_unreadable(tmp_path):
    a = _write(tmp_path / "a", [_v2(0, _losses([2, 2], draws=2))])
    b = _write(tmp_path / "b", [_null_in_draw_1()])
    with pytest.raises(vc.Unreadable, match="step 0 has a non-finite loss"):
        vc.compare([("A", a), ("B", b)])


def _split_with(representatives):
    return _split(representatives=representatives)


@pytest.mark.parametrize("text", [
    "{}", "[]", '{"holdout": "x", "representatives": []}',
    '{"holdout": ["k"], "representatives": [{}], "split_seed": 0}',
    _split_with([]),  # no held-out image described
    _split_with([{"key": key} for key in KEYS]),  # keys only: no caption or size
    _split_with([{"key": key, "caption": "", "target_w": 64, "target_h": 64} for key in KEYS[:-1]]),  # one missing
])
def test_a_split_file_that_is_not_a_split_is_unreadable(tmp_path, text):
    a = _write(tmp_path / "a", [_v2(0, _losses([1, 1]))])
    b = _write(tmp_path / "b", [_v2(0, _losses([1, 1]))], split=text)
    with pytest.raises(vc.Unreadable, match=r"b[\\/]validation_split.json"):
        vc.compare([("A", a), ("B", b)])


def test_runs_of_different_models_are_refused(tmp_path):
    a = _write(tmp_path / "a", [_v2(0, _losses([1, 1]))], split=_split(model="family-a"))
    b = _write(tmp_path / "b", [_v2(0, _losses([1, 1]))], split=_split(model="family-b"))
    with pytest.raises(vc.Refusal, match="family-a.*family-b"):
        vc.compare([("A", a), ("B", b)])


def test_a_run_whose_model_isnt_recorded_compares_with_a_note(tmp_path):
    a = _write(tmp_path / "a", [_v2(0, _losses([1, 1]))], split=_split(model="family-a"))
    b = _write(tmp_path / "b", [_v2(0, _losses([1, 1]))])
    report = vc.compare([("A", a), ("B", b)])
    assert any("doesn't record its model" in note and str(b) in note for note in report["notes"])


def test_runs_of_one_recorded_model_get_no_model_note(tmp_path):
    a = _write(tmp_path / "a", [_v2(0, _losses([1, 1]))], split=_split(model="family-a"))
    b = _write(tmp_path / "b", [_v2(0, _losses([1, 1]))], split=_split(model="family-a"))
    assert not any("model" in note for note in vc.compare([("A", a), ("B", b)])["notes"])


@pytest.mark.parametrize("version", [3, "2", 0])
def test_a_record_of_an_unknown_version_is_unreadable(tmp_path, version):
    a = _write(tmp_path / "a", [_v2(0, _losses([1, 1]))])
    record = _v2(0, _losses([1, 1]))
    record["version"] = version
    b = _write(tmp_path / "b", [record])
    with pytest.raises(vc.Unreadable, match="version"):
        vc.compare([("A", a), ("B", b)])


@pytest.mark.parametrize("field", ["step", "noise_draws", "train_seed"])
def test_a_boolean_where_an_integer_belongs_is_unreadable(tmp_path, field):
    a = _write(tmp_path / "a", [_v2(0, _losses([1, 1]))])
    record = _v2(0, _losses([1, 1]), seed=1)
    record[field] = True
    b = _write(tmp_path / "b", [record])
    with pytest.raises(vc.Unreadable):
        vc.compare([("A", a), ("B", b)])


# ── Verdicts on planted gaps ─────────────────────────────────────────────────


def _seeded(tmp_path, gap, spread, images=6):
    """Two seeds per arm; A is B plus ``gap`` everywhere, its seeds ``spread`` apart."""
    base = _losses([1.0, 0.5], images)
    shifted = lambda d: [[[v + d for v in draw] for draw in image] for image in base]  # noqa: E731
    return _runs(tmp_path, {
        "a1": ("A", [_v2(0, shifted(gap), seed=1, keys=KEYS[:images])]),
        "a2": ("A", [_v2(0, shifted(gap + spread), seed=2, keys=KEYS[:images])]),
        "b1": ("B", [_v2(0, shifted(0.0), seed=1, keys=KEYS[:images])]),
        "b2": ("B", [_v2(0, shifted(spread), seed=2, keys=KEYS[:images])]),
    })


def _wiggle(images, amplitude):
    """Per-image differences that vary, so an interval has width."""
    return [amplitude * ((i % 3) - 1) for i in range(images)]


def test_a_gap_far_beyond_a_small_seed_spread_beats_the_floor(tmp_path):
    runs = _seeded(tmp_path, gap=0.0, spread=0.01)
    # A's images get a gap of 1.0 with some spread, so the interval has width but excludes 0.
    for folder in (tmp_path / "a1", tmp_path / "a2"):
        record = json.loads((folder / vc.METRICS_FILE).read_text(encoding="utf-8"))
        record["image_band_loss"] = [[[v + 1.0 + w for v in draw] for draw in image]
                                     for image, w in zip(record["image_band_loss"], _wiggle(6, 0.05), strict=True)]
        _write(folder, [record])
    report = vc.compare(runs)
    assert report["floor"]["overall"] == pytest.approx(0.01)
    overall = _overall(report)
    assert overall["gap"] == pytest.approx(1.0)  # both arms carry the same seed offset
    assert overall["interval"][0] > 0
    assert overall["verdict"] == vc.BEATS
    assert all(band["verdict"] == vc.BEATS for band in report["pairs"][0]["bands"])


def test_the_same_gap_with_a_larger_seed_spread_is_within_the_floor(tmp_path):
    report = vc.compare(_seeded(tmp_path, gap=1.0, spread=3.0))
    assert report["floor"]["overall"] == pytest.approx(3.0)
    assert _overall(report)["verdict"] == vc.WITHIN


def test_an_interval_straddling_zero_is_within_the_floor(tmp_path):
    base = _losses([1.0, 0.5])
    a = [[[v + w for v in draw] for draw in image] for image, w in zip(base, [0.3, -0.3, 0.2, -0.2, 0.25, -0.24], strict=True)]
    runs = _runs(tmp_path, {
        "a1": ("A", [_v2(0, a, seed=1)]),
        "a2": ("A", [_v2(0, a, seed=2)]),
        "b1": ("B", [_v2(0, base, seed=1)]),
        "b2": ("B", [_v2(0, base, seed=2)]),
    })
    report = vc.compare(runs)
    overall = _overall(report)
    assert overall["interval"][0] < 0 < overall["interval"][1]
    assert report["floor"]["overall"] == 0.0
    assert abs(overall["gap"]) > report["floor"]["overall"]
    assert overall["verdict"] == vc.WITHIN


@pytest.mark.parametrize("digest", ["missing", None, ""])
def test_a_split_file_without_content_digests_gets_no_verdict(tmp_path, digest):
    """Every image described but no content digest (written before digests were recorded): the
    same names may hold other pictures, so the gap is reported and no verdict rests on it."""
    runs = _seeded(tmp_path, gap=1.0, spread=0.01)
    representatives = [{"key": key, "caption": "", "target_w": 64, "target_h": 64} for key in KEYS]
    if digest != "missing":
        representatives = [{**item, "sha256": digest} for item in representatives]
    for _, folder in runs:
        (folder / vc.SPLIT_FILE).write_text(_split_with(representatives), encoding="utf-8")
    overall = _overall(vc.compare(runs))
    assert overall["gap"] == pytest.approx(1.0)
    assert overall["interval"] is None
    assert overall["verdict"].startswith("no verdict: ")
    assert "records no content digest" in overall["verdict"]


def test_four_images_get_no_verdict(tmp_path):
    report = vc.compare(_seeded(tmp_path, gap=1.0, spread=0.01, images=4))
    assert _overall(report)["interval"] is None
    assert _overall(report)["verdict"] == "no verdict: too few held-out images (4) for an interval; at least 5"


def test_five_images_get_an_interval(tmp_path):
    report = vc.compare(_seeded(tmp_path, gap=1.0, spread=0.01, images=5))
    assert _overall(report)["interval"] is not None


def test_one_seed_per_arm_gets_no_verdict(tmp_path):
    runs = _runs(tmp_path, {
        "a": ("A", [_v2(0, _losses([2.0, 1.0]), seed=1)]),
        "b": ("B", [_v2(0, _losses([1.0, 0.5]), seed=1)]),
    })
    report = vc.compare(runs)
    assert report["floor"] is None
    assert _overall(report)["verdict"].startswith("no verdict: no noise floor: arm A has one seed run")


def test_a_band_can_differ_from_the_overall_verdict(tmp_path):
    base = _losses([1.0, 0.5])
    wiggle = _wiggle(6, 0.05)
    a = [[[draw[0] + 2.0 + w, draw[1]] for draw in image] for image, w in zip(base, wiggle, strict=True)]
    runs = _runs(tmp_path, {
        "a1": ("A", [_v2(0, a, seed=1)]),
        "a2": ("A", [_v2(0, [[[draw[0] + 0.01, draw[1] + 0.01] for draw in image] for image in a], seed=2)]),
        "b1": ("B", [_v2(0, base, seed=1)]),
        "b2": ("B", [_v2(0, [[[v + 0.01 for v in draw] for draw in image] for image in base], seed=2)]),
    })
    pair = vc.compare(runs)["pairs"][0]
    assert pair["overall"]["verdict"] == vc.BEATS
    assert pair["bands"][0]["verdict"] == vc.BEATS
    assert pair["bands"][1]["verdict"] == vc.WITHIN


# ── Older records ────────────────────────────────────────────────────────────


def test_older_records_report_gaps_without_intervals_or_verdicts(tmp_path):
    runs = _runs(tmp_path, {
        "uniform": ("uniform", [_v1(0, [1.2, 0.6]), _v1(1999, [1.1, 0.5])]),
        "radc": ("radc", [_v1(0, [1.2, 0.6]), _v1(1999, [1.0, 0.45])]),
    })
    report = vc.compare(runs)
    assert report["step"] == 1999
    overall = _overall(report, "uniform", "radc")
    assert overall["gap"] == pytest.approx(((1.1 - 1.0) + (0.5 - 0.45)) / 2)
    assert overall["interval"] is None
    assert "has no per-image losses (recorded before per-image losses were kept)" in overall["verdict"]
    assert overall["verdict"].startswith("no verdict: no interval: ")
    assert vc.ONE_DRAW in report["notes"]
    assert report["pairs"][0]["measurement_noise"] is None


def test_mixed_versions_compare_draw_zero_aggregates(tmp_path):
    a_v2 = [[[1.0 + i, 2.0 + i], [9.0, 9.0], [9.0, 9.0], [9.0, 9.0]] for i in range(6)]
    b1 = [[[0.5 + i, 1.0], [7.0, 7.0], [7.0, 7.0], [7.0, 7.0]] for i in range(6)]
    b2 = [[[0.7 + i, 1.2], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]] for i in range(6)]
    runs = _runs(tmp_path, {
        "a_old": ("A", [_v1(0, [3.0, 4.0])]),
        "a_new": ("A", [_v2(0, a_v2, seed=1)]),
        "b1": ("B", [_v2(0, b1, seed=1)]),
        "b2": ("B", [_v2(0, b2, seed=2)]),
    })
    report = vc.compare(runs)

    # By hand: each v2 run cut to draw 0 and averaged over images; each arm the mean of its runs.
    a_new = [sum(1.0 + i for i in range(6)) / 6, sum(2.0 + i for i in range(6)) / 6]  # 3.5, 4.5
    arm_a = [(3.0 + a_new[0]) / 2, (4.0 + a_new[1]) / 2]
    b1_agg = [sum(0.5 + i for i in range(6)) / 6, 1.0]
    b2_agg = [sum(0.7 + i for i in range(6)) / 6, 1.2]
    arm_b = [(b1_agg[0] + b2_agg[0]) / 2, (b1_agg[1] + b2_agg[1]) / 2]
    bands = [arm_a[k] - arm_b[k] for k in range(2)]

    pair = report["pairs"][0]
    assert [band["gap"] for band in pair["bands"]] == pytest.approx(bands)
    assert pair["overall"]["gap"] == pytest.approx(sum(bands) / 2)
    assert report["arm_floors"]["A"] is None
    assert report["arm_floors"]["B"]["bands"] == pytest.approx([abs(b1_agg[k] - b2_agg[k]) for k in range(2)])
    assert report["floor"] is None
    assert pair["measurement_noise"] is None
    assert vc.ONE_DRAW in report["notes"]


# ── Draws and measurement noise ──────────────────────────────────────────────


def _two_draw_pair(tmp_path, gaps, images=5):
    b = [[[1.0, 1.0] for _ in gaps] for _ in range(images)]
    a = [[[1.0 + g, 1.0 + g] for g in gaps] for _ in range(images)]
    return _runs(tmp_path, {"a": ("A", [_v2(0, a, keys=KEYS[:images])]),
                            "b": ("B", [_v2(0, b, keys=KEYS[:images])])})


def test_measurement_noise_is_the_sample_deviation_and_flags_a_gap_it_matches(tmp_path):
    noise = vc.compare(_two_draw_pair(tmp_path, [0.2, 1.8]))["pairs"][0]["measurement_noise"]["overall"]
    assert round(noise["sd"], 4) == 1.1314
    assert noise["sd"] == pytest.approx(statistics.stdev([0.2, 1.8]))
    assert (noise["min"], noise["max"]) == pytest.approx((0.2, 1.8))
    assert noise["as_large_as_gap"] is True


def test_measurement_noise_smaller_than_the_gap_is_not_flagged(tmp_path):
    report = vc.compare(_two_draw_pair(tmp_path, [0.9, 1.1]))
    assert _overall(report)["gap"] == pytest.approx(1.0)
    noise = report["pairs"][0]["measurement_noise"]["overall"]
    assert round(noise["sd"], 4) == 0.1414
    assert noise["as_large_as_gap"] is False


def test_one_shared_draw_measures_no_noise(tmp_path):
    report = vc.compare(_two_draw_pair(tmp_path, [1.0]))
    assert report["pairs"][0]["measurement_noise"] is None
    assert vc.ONE_DRAW in report["notes"]


def test_only_the_shared_draws_are_used(tmp_path):
    four = [[[1.0 + 0.1 * d + 0.01 * i, 2.0 - 0.1 * d] for d in range(4)] for i in range(6)]
    two = [[[0.5 + 0.2 * d + 0.02 * i, 1.0 + 0.3 * d] for d in range(2)] for i in range(6)]
    report = vc.compare(_runs(tmp_path / "mixed", {"a": ("A", [_v2(0, four)]), "b": ("B", [_v2(0, two)])}))
    truncated = vc.compare(_runs(tmp_path / "cut", {"a": ("A", [_v2(0, [image[:2] for image in four])]),
                                                    "b": ("B", [_v2(0, two)])}))
    assert report["shared_draws"] == 2
    assert report["pairs"] == truncated["pairs"]


# ── The bootstrap ────────────────────────────────────────────────────────────


def _dump(report):
    return json.dumps(report, sort_keys=True)


def test_the_same_inputs_and_seed_give_identical_json(tmp_path):
    runs = _seeded(tmp_path, gap=0.5, spread=0.1)
    assert _dump(vc.compare(runs, seed=3, resamples=500)) == _dump(vc.compare(runs, seed=3, resamples=500))


def test_another_seed_changes_only_the_intervals(tmp_path):
    base = _losses([1.0, 0.5])
    a = [[[v + 0.4 + w for v in draw] for draw in image] for image, w in zip(base, _wiggle(6, 0.3), strict=True)]
    runs = _runs(tmp_path, {
        "a1": ("A", [_v2(0, a, seed=1)]), "a2": ("A", [_v2(0, _shifted([[v + 0.02 for v in im[0]] for im in a]), seed=2)]),
        "b1": ("B", [_v2(0, base, seed=1)]), "b2": ("B", [_v2(0, base, seed=2)]),
    })
    first, second = vc.compare(runs, seed=0, resamples=300), vc.compare(runs, seed=1, resamples=300)

    def without_intervals(report):
        report = json.loads(_dump(report))
        report.pop("seed")
        for pair in report["pairs"]:
            for row in [pair["overall"], *pair["bands"]]:
                row.pop("interval")
                row.pop("verdict")
        return report

    assert without_intervals(first) == without_intervals(second)
    assert _overall(first)["interval"] != _overall(second)["interval"]


def test_the_interval_is_the_2_5th_and_97_5th_percentile_of_the_resampled_gaps(tmp_path):
    import random

    base = _losses([1.0, 1.0])
    wiggle = [0.9, -0.4, 0.3, 0.1, -0.7, 0.5]
    a = [[[v + w for v in draw] for draw in image] for image, w in zip(base, wiggle, strict=True)]
    runs = _runs(tmp_path, {"a": ("A", [_v2(0, a)]), "b": ("B", [_v2(0, base)])})
    interval = _overall(vc.compare(runs, seed=7, resamples=1000))["interval"]

    # The same resamples, drawn as the module draws them: images with replacement.
    rng = random.Random(7)
    picks = [[rng.randrange(6) for _ in range(6)] for _ in range(1000)]
    gaps = sorted(sum(wiggle[i] for i in pick) / 6 for pick in picks)
    assert interval == pytest.approx([gaps[25], gaps[974]])
    assert gaps[0] < interval[0] and interval[1] < gaps[-1]


def test_equal_per_image_differences_give_the_same_interval_under_any_seed(tmp_path):
    runs = _seeded(tmp_path, gap=0.7, spread=0.01)
    intervals = {tuple(_overall(vc.compare(runs, seed=s, resamples=200))["interval"]) for s in range(5)}
    assert len(intervals) == 1


def test_the_text_report_names_each_pair_and_verdict(tmp_path):
    report = vc.compare(_seeded(tmp_path, gap=1.0, spread=3.0))
    text = vc.format_report(report)
    assert "A - B" in text
    assert vc.WITHIN in text
    assert vc.FLOOR_CAVEAT in text


def test_the_text_states_each_no_verdict_reason_once(tmp_path):
    runs = _runs(tmp_path, {
        "a": ("A", [_v1(0, [1.1, 0.5])]), "b": ("B", [_v1(0, [1.0, 0.4])]), "c": ("C", [_v1(0, [1.2, 0.6])]),
    })
    report = vc.compare(runs)
    text = vc.format_report(report)
    for reason in report["reasons"]:
        assert text.count(reason) == 1
    assert text.count("no verdict (reasons above)") == 3 * (1 + len(CENTRES))
    assert "training seed unknown" in text


# ── One rule, and a light module ─────────────────────────────────────────────


def test_the_split_guard_and_the_comparison_share_one_rule():
    from app.engine.core.pipeline import validation_split

    assert validation_split.same_measurement is vc.same_measurement


_BACKEND = Path(__file__).resolve().parents[2]


def _heavy_modules_after(module):
    code = (
        "import importlib, sys; importlib.import_module(sys.argv[1]); "
        "print(','.join(sorted(m for m in ('torch', 'PIL', 'numpy') if m in sys.modules)))"
    )
    done = subprocess.run([sys.executable, "-c", code, module], cwd=_BACKEND, capture_output=True, text=True,
                          check=True, timeout=300)
    return done.stdout.strip().splitlines()[-1] if done.stdout.strip() else ""


def test_the_comparison_loads_no_torch_pil_or_numpy():
    assert _heavy_modules_after("app.engine.core.validation_compare") == ""


def test_the_import_check_sees_pil_through_the_split_module():
    assert "PIL" in _heavy_modules_after("app.engine.core.pipeline.validation_split").split(",")
