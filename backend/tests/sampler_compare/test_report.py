"""The report: the noise floor and every pair's verdict, over finished runs the guards accept."""
import dataclasses
import json
import os

import pytest

from app.core.sampler_compare import plan, report, runner
from app.engine.core import validation_compare as vc

from . import fake_trainer
from .conftest import DEFINITION

ARMS = {
    "uniform": {"timestep_sampling": "uniform"},
    "logit_normal": {"timestep_sampling": "logit_normal", "logit_normal_mu": 0.0, "logit_normal_sigma": 1.0},
    "radc": {"timestep_sampling": "radc", "radc_start": 0.9, "radc_end": 0.3},
}


def _finished(world, mode="ok", modes=None, **changes):
    c = world.comparison(arms=ARMS, **changes)
    world.place_all(c, mode=mode, modes=modes)
    return c


def _write(world, c, queue_stop=None):
    code = report.write_report(c, queue_stop, runner.prepare(c, world.seams(), copy=False))
    return code, json.loads((c.work_dir / "report.json").read_text(encoding="utf-8"))


@pytest.fixture
def no_compare(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the comparison ran over runs the guards refuse")

    monkeypatch.setattr(report.vc, "compare", refuse)


def test_the_report_names_the_floor_and_every_pairs_verdict(world):
    c = _finished(world)
    code, out = _write(world, c)
    assert code == 0
    assert out["version"] == 1 and out["compare_error"] is None
    assert [(p["a"], p["b"]) for p in out["pairs"]] == [("uniform", "logit_normal"), ("uniform", "radc"),
                                                        ("logit_normal", "radc")]
    for pair in out["pairs"]:
        assert {"gap", "interval", "verdict"} <= set(pair)
        assert pair["verdict"] in (vc.BEATS, vc.WITHIN) or pair["verdict"].startswith("no verdict:")
    assert out["floor"]["overall"] == pytest.approx(fake_trainer.SEED_STEP, abs=1e-12)
    gaps = {(p["a"], p["b"]): p["gap"] for p in out["pairs"]}
    uniform, logit, radc = (fake_trainer.MODE_OFFSETS[m] for m in ("uniform", "logit_normal", "radc"))
    assert gaps[("uniform", "radc")] == pytest.approx(uniform - radc, abs=1e-12)
    assert gaps[("uniform", "logit_normal")] == pytest.approx(uniform - logit, abs=1e-12)
    text = (c.work_dir / "report.txt").read_text(encoding="utf-8").splitlines()
    assert text[0].startswith("Noise floor: 0.00050")
    assert [line.split(":")[0] for line in text[1:4]] == ["uniform - logit_normal", "uniform - radc",
                                                         "logit_normal - radc"]


def test_the_fake_runs_pass_the_real_comparison(world):
    c = _finished(world)
    folders = [(arm, runner.run_dir(c, plan.label(arm, seed), DEFINITION)) for arm, seed in plan.queue(c)]
    result = vc.compare(folders, resamples=100)
    assert result["images"] == 6 and result["shared_draws"] == 4 and len(result["band_centres"]) == 8


def test_one_finished_arm_gives_a_report_that_says_why_nothing_was_compared(world):
    c = world.comparison(arms=ARMS)
    world.place_finished(c, "uniform", 42)
    world.place_finished(c, "uniform", 43)
    code, out = _write(world, c)
    assert code == 1
    assert "1 arm" in out["compare_error"]
    assert [r["label"] for r in out["runs"]] == [plan.label(a, s) for a, s in plan.queue(c)]
    assert [r["finished"] for r in out["runs"]] == [True, False, False, True, False, False]


def test_a_refusal_from_the_comparison_is_in_the_report(world, monkeypatch):
    c = _finished(world)

    def refuse(*args, **kwargs):
        raise vc.Refusal("the runs don't measure the same thing")

    monkeypatch.setattr(report.vc, "compare", refuse)
    code, out = _write(world, c)
    assert code == 1 and out["pairs"] == []
    assert "the runs don't measure the same thing" in out["compare_error"]


@pytest.mark.parametrize(
    ("modes", "reason"),
    [
        ("no_model", "its split names no model"),
        ("other_model", "'flux3-image-klein-base', not 'moss-image-base'"),
        ({"radc_s43": "no_digest"}, "records no content digest"),
    ],
)
def test_the_report_runs_the_split_guard_before_comparing(world, no_compare, modes, reason):
    c = _finished(world, **({"modes": modes} if isinstance(modes, dict) else {"mode": modes}))
    code, out = _write(world, c)
    assert code == 1 and out["pairs"] == []
    assert reason in out["compare_error"]
    assert "_s4" in out["compare_error"]


@pytest.mark.parametrize(("mode", "reason"), [("wrong_split_seed", "split_seed 1, not 0"),
                                              ("wrong_draws", "noise_draws 1, not 4")])
def test_the_report_refuses_a_consistently_wrong_measurement(world, no_compare, mode, reason):
    c = _finished(world, mode=mode)
    code, out = _write(world, c)
    assert code == 1 and out["pairs"] == []
    assert reason in out["compare_error"]


def test_the_report_leaves_out_nothing_silently_for_a_refused_run(world, no_compare):
    c = world.comparison(arms=ARMS)
    for arm, seed in plan.queue(c):
        if (arm, seed) == ("uniform", 43):
            world.place_finished(c, arm, seed, mode="wrong_sampler", accepted=False,
                                 stopped_for="the run's sampler is 'radc', not 'uniform'")
        else:
            world.place_finished(c, arm, seed)
    code, out = _write(world, c)
    assert code == 1 and out["pairs"] == []
    assert "uniform_s43" in out["compare_error"] and "the run's sampler is 'radc', not 'uniform'" in out["compare_error"]
    row = next(r for r in out["runs"] if r["label"] == "uniform_s43")
    assert row["accepted"] is False


def test_the_report_refuses_runs_of_another_identity(world):
    c = _finished(world)
    world.set_base(learning_rate=0.0002)
    code, out = _write(world, c)
    assert code == 1 and out["pairs"] == []
    assert "config.learning_rate" in out["compare_error"]
    world.set_base(learning_rate=0.0001)
    code, out = _write(world, c)
    assert code == 0 and len(out["pairs"]) == 3


def test_each_run_row_carries_its_telemetry_and_effective_sampler(world):
    c = _finished(world)
    record = {"label": "uniform_s42", "arm": "uniform", "seed": 42, "ran": True, "minutes": 12.5,
              "peak_temperature_c": 71.0, "peak_vram_mib": 40000, "peak_power_w": 480.0,
              "min_available_ram_gib": 30.0, "stopped_for": None}
    with open(c.work_dir / "runs.jsonl", "a", encoding="utf-8") as out:
        out.write(json.dumps(record) + "\n")
    prepared = runner.prepare(c, world.seams(), copy=False)
    facts = json.loads(json.dumps(prepared.facts))
    facts["arms"]["uniform"]["source"]["logit_normal_mu"] = "family"
    report.write_report(c, None, dataclasses.replace(prepared, facts=facts))
    out = json.loads((c.work_dir / "report.json").read_text(encoding="utf-8"))
    row = next(r for r in out["runs"] if r["label"] == "uniform_s42")
    assert (row["minutes"], row["peak_temperature_c"], row["peak_vram_mib"], row["peak_power_w"]) == (12.5, 71.0, 40000, 480.0)
    assert row["sampler"]["logit_normal_mu"] == "0.0 from the family"
    assert row["sampler"]["timestep_sampling"] == "uniform from the arm"
    text = (c.work_dir / "report.txt").read_text(encoding="utf-8")
    assert "logit_normal_mu 0.0 from the family" in text
    assert "uniform_s42" in text and "71" in text and "40000" in text and "480" in text


def test_a_stopped_queue_is_the_reports_first_line(world):
    c = _finished(world)
    _write(world, c, queue_stop="radc_s43: the card reads 95 C, over 94 C")
    first = (c.work_dir / "report.txt").read_text(encoding="utf-8").splitlines()[0]
    assert first == "Queue stopped: radc_s43: the card reads 95 C, over 94 C"


def test_the_report_is_written_through_a_fresh_file(world, monkeypatch):
    c = _finished(world)
    _write(world, c)
    earlier = (c.work_dir / "report.json").read_bytes()

    def failing_replace(source, destination):
        raise OSError("disk full")

    prepared = runner.prepare(c, world.seams(), copy=False)
    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        report.write_report(c, "a stop", prepared)
    monkeypatch.undo()
    assert (c.work_dir / "report.json").read_bytes() == earlier
    assert not [p.name for p in c.work_dir.iterdir() if p.name.endswith(".tmp")]


def test_the_floor_caveat_is_quoted_from_the_module(world):
    c = _finished(world)
    _write(world, c)
    assert vc.FLOOR_CAVEAT in (c.work_dir / "report.txt").read_text(encoding="utf-8")
