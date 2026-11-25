"""One run and the queue, against the fake trainer and backend as real child processes."""
import dataclasses
import json
import socket
import time

import psutil
import pytest

from app.core.sampler_compare import isolation, plan, runner
from app.core.sampler_compare.telemetry import Card

from .conftest import DEFINITION, GIB, FakeClock, port_taken, started_pids


def _records(c):
    path = c.work_dir / "runs.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _events(c, label, event, suffix="log"):
    out = []
    for line in (c.work_dir / "logs" / f"{label}.{suffix}").read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("event") == event:
            out.append(record)
    return out


def _run_all(world, c=None, **seams):
    c = c or world.comparison()
    return c, runner.run_queue(c, world.seams(**seams))


def test_a_queue_of_two_arms_and_two_seeds_runs_to_the_end(world):
    c, stop = _run_all(world)
    assert stop is None
    records = _records(c)
    assert [r["label"] for r in records] == ["uniform_s42", "radc_s42", "uniform_s43", "radc_s43"]
    assert all(r["complete"] and r["stopped_for"] is None and r["ran"] for r in records)
    assert world.trainer_starts == 4
    status = json.loads((c.work_dir / "status.json").read_text(encoding="utf-8"))
    assert status["version"] == 1 and status["queue_stop"] is None


def test_the_trainer_and_the_backend_get_the_comparison_port(world):
    c = world.comparison(arms={"uniform": {"timestep_sampling": "uniform"}, "radc": {"timestep_sampling": "radc"}},
                         seeds=[42, 43])
    runner.run_queue(c, world.seams())
    assert _events(c, "uniform_s42", "fake_trainer_started")[0]["port"] == str(c.port)
    assert _events(c, "uniform_s42", "fake_backend_started", "backend.log")[0]["port"] == str(c.port)
    started = _events(c, "uniform_s42", "fake_trainer_started")[0]
    assert started["db"].startswith(str(c.work_dir))
    assert started["offline"] == "1"


def test_the_telemetry_file_has_a_row_before_the_trainer_starts(world):
    c, _ = _run_all(world)
    assert _events(c, "uniform_s42", "fake_trainer_started")[0]["telemetry_lines"] >= 2
    lines = (c.work_dir / "logs" / "uniform_s42.telemetry.csv").read_text(encoding="utf-8").splitlines()
    assert lines[-1].endswith("ended: finished")


def test_a_taken_port_is_refused_and_its_owner_untouched(world):
    owner = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    owner.bind(("127.0.0.1", world.port))
    owner.listen()
    try:
        c, stop = _run_all(world)
        assert "already taken" in stop
        assert world.backend_starts == 0 and world.trainer_starts == 0
        assert port_taken(world.port)
    finally:
        owner.close()


def test_a_power_limit_other_than_the_configured_one_refuses_the_run(world):
    world.card = Card(1000, 40.0, 100.0, 450.0)
    c, stop = _run_all(world)
    assert "the power limit reads 450 W, not 500 W" in stop
    assert world.trainer_starts == 0
    limits = {**json.loads(json.dumps(dataclasses.asdict(c.limits))), "power_limit_w": None}
    c2 = world.comparison(limits=limits, work_dir=str(world.root / "work2"))
    assert runner.run_queue(c2, world.seams()) is None


def test_a_busy_card_is_waited_for_then_refused(world):
    world.card = Card(30000, 40.0, 100.0, 500.0)
    clock = FakeClock(scale=1.0)
    c, stop = _run_all(world, clock=clock.clock, sleep=clock.sleep)
    assert "the card is not idle after 30 min: 30000 MiB in use" in stop
    assert 30 * 60 <= clock.now <= 31 * 60
    assert world.trainer_starts == 0


def _assert_all_gone(c, label):
    pids = started_pids(c.work_dir / "logs" / f"{label}.log", "fake_trainer_started")
    pids += started_pids(c.work_dir / "logs" / f"{label}.backend.log", "fake_backend_started")
    assert pids
    deadline = time.monotonic() + 10
    while any(psutil.pid_exists(p) and psutil.Process(p).status() != psutil.STATUS_ZOMBIE for p in pids):
        assert time.monotonic() < deadline, f"still alive: {pids}"
        time.sleep(0.1)
    assert not port_taken(c.port)


@pytest.mark.parametrize("phase", ["backend_never_answers", "trainer_exit1", "trainer_hangs_hot", "read_card_raises"])
def test_no_child_survives_a_stop_at_any_phase(world, monkeypatch, phase):
    seams = {}
    if phase == "backend_never_answers":
        monkeypatch.setenv("FAKE_BACKEND", "never_answers")
        clock = FakeClock(scale=2000.0)  # a refused connection takes its own seconds on some hosts
        seams = {"clock": clock.clock, "sleep": clock.sleep}
    elif phase == "trainer_exit1":
        world.default_mode = "exit1"
    elif phase == "trainer_hangs_hot":
        world.default_mode = "hang"
        readings = iter([world.card] + [world.card] * 30)
        seams = {"read_card": lambda: next(readings, Card(1000, 99.0, 100.0, 500.0))}
    else:
        calls = {"n": 0}

        def read_card():
            calls["n"] += 1
            if calls["n"] > 3:
                raise RuntimeError("the card reader broke")
            return world.card

        world.default_mode = "hang"
        seams = {"read_card": read_card}
    c = world.comparison()
    if phase == "read_card_raises":
        with pytest.raises(RuntimeError, match="the card reader broke"):
            runner.run_queue(c, world.seams(**seams))
    else:
        stop = runner.run_queue(c, world.seams(**seams))
        assert stop is not None
        if phase == "backend_never_answers":
            assert "did not answer" in stop
        elif phase == "trainer_exit1":
            assert "the trainer exited with 1" in stop
        else:
            assert "the card reads 99 C, over 94 C" in stop
    if phase == "backend_never_answers":
        pids = started_pids(c.work_dir / "logs" / "uniform_s42.backend.log", "fake_backend_started")
        assert pids and not any(psutil.pid_exists(p) for p in pids)
        assert world.trainer_starts == 0
    else:
        _assert_all_gone(c, "uniform_s42")


def test_a_stalled_run_is_stopped(world):
    world.default_mode = "hang"
    clock = FakeClock(scale=200.0)
    c, stop = _run_all(world, clock=clock.clock, sleep=clock.sleep)
    assert "no log output for 15 min" in stop
    _assert_all_gone(c, "uniform_s42")


def test_a_run_with_another_sampler_stops_the_queue(world):
    world.default_mode = "wrong_sampler"
    c, stop = _run_all(world)
    assert stop == "uniform_s42: the run's sampler is 'radc', not 'uniform'"
    assert world.trainer_starts == 1


def test_a_run_whose_split_differs_from_the_first_stops_the_queue(world):
    world.modes["radc_s42"] = "other_split"
    c, stop = _run_all(world)
    assert stop.startswith("radc_s42: ")
    assert "not the same measurement as" in stop
    assert world.trainer_starts == 2


def test_a_split_without_content_digests_stops_the_queue(world):
    world.default_mode = "no_digest"
    c, stop = _run_all(world)
    assert "records no content digest" in stop
    assert not (c.work_dir / "comparison.json").exists()


def test_a_split_naming_another_model_stops_the_queue(world):
    world.default_mode = "other_model"
    c, stop = _run_all(world)
    assert "flux3-image-klein-base" in stop and DEFINITION in stop


def test_a_split_naming_no_model_stops_the_queue(world):
    world.default_mode = "no_model"
    c, stop = _run_all(world)
    assert "its split names no model" in stop


def test_a_skipped_complete_run_gets_the_same_split_check(world):
    c, stop = _run_all(world)
    assert stop is None
    radc =runner.run_dir(c, "radc_s42", DEFINITION)
    content = json.loads((radc / "validation_split.json").read_text(encoding="utf-8"))
    names = sorted(p.name for p in (c.work_dir / isolation.DATASET_FOLDER).glob("*.png"))
    held = names[-len(content["holdout"]):]
    content["holdout"] = [f"Cars/{n}" for n in held]
    for item, name in zip(content["representatives"], held, strict=True):
        item["key"] = f"Cars/{name}"
        item["sha256"] = json.loads((c.work_dir / "data" / "manifest.json").read_text())["files"][name]
    (radc / "validation_split.json").write_text(json.dumps(content), encoding="utf-8")
    starts = world.trainer_starts
    stop = runner.run_queue(c, world.seams())
    assert world.trainer_starts == starts
    assert "radc_s42" in stop and "not the same measurement as" in stop


def test_the_first_run_cannot_replace_the_recorded_measurement(world):
    c, _ = _run_all(world)
    recorded = (c.work_dir / "comparison.json").read_bytes()
    uniform = runner.run_dir(c, "uniform_s42", DEFINITION)
    content = json.loads((uniform / "validation_split.json").read_text(encoding="utf-8"))
    content["global_triggerword"] = "changed"
    (uniform / "validation_split.json").write_text(json.dumps(content), encoding="utf-8")
    stop = runner.run_queue(c, world.seams())
    assert stop is not None and "uniform_s42" in stop
    assert (c.work_dir / "comparison.json").read_bytes() == recorded


def _folders(c):
    return {p.name: sorted(x.name for x in p.iterdir()) for p in (c.work_dir / "runs").iterdir()}


def test_a_changed_reference_setting_refuses_the_finished_runs(world):
    c, stop = _run_all(world)
    assert stop is None
    before = _folders(c)
    world.set_base(learning_rate=0.0002)
    starts = world.trainer_starts
    stop = runner.run_queue(c, world.seams())
    assert world.trainer_starts == starts
    assert "uniform_s42" in stop and "config.learning_rate" in stop
    assert _folders(c) == before


@pytest.mark.parametrize(
    ("change", "key"),
    [
        ({"steps": 1000}, "config.max_train_steps"),
        ({"validation": {"every": 250, "noise_draws": 4, "holdout_fraction": 0.3, "split_seed": 0}},
         "config.validation_every_n_steps"),
        ({"validation": {"every": 500, "noise_draws": 1, "holdout_fraction": 0.3, "split_seed": 0}},
         "config.validation_noise_draws"),
        ({"validation": {"every": 500, "noise_draws": 4, "holdout_fraction": 0.2, "split_seed": 0}},
         "config.validation_holdout_fraction"),
        ({"arms": {"uniform": {"timestep_sampling": "uniform"},
                   "radc": {"timestep_sampling": "radc", "radc_start": 0.9, "radc_end": 0.2}}}, "config.radc_end"),
    ],
)
def test_a_changed_arm_seed_steps_or_validation_refuses_the_finished_run(world, change, key):
    c, stop = _run_all(world)
    assert stop is None
    starts = world.trainer_starts
    changed = world.comparison(**change)
    stop = runner.run_queue(changed, world.seams())
    assert world.trainer_starts == starts
    assert key in stop


def test_a_finished_run_without_an_identity_is_refused(world):
    c = world.comparison()
    world.place_finished(c, "uniform", 42, identity=False)
    stop = runner.run_queue(c, world.seams())
    assert world.trainer_starts == 0
    assert "uniform_s42" in stop and "no recorded identity" in stop


def test_an_unchanged_experiment_reuses_its_finished_runs(world):
    c, stop = _run_all(world)
    assert stop is None
    starts = world.trainer_starts
    assert runner.run_queue(c, world.seams()) is None
    assert world.trainer_starts == starts
    reused = [r for r in _records(c) if r.get("reused")]
    assert [r["label"] for r in reused] == ["uniform_s42", "radc_s42", "uniform_s43", "radc_s43"]
    assert all(r["complete"] for r in reused)


def test_the_identity_is_on_disk_before_the_trainer_starts(world):
    c, _ = _run_all(world)
    assert _events(c, "uniform_s42", "fake_trainer_started")[0]["identity_found"] is True


def test_every_run_recording_the_same_wrong_split_seed_is_refused(world):
    world.default_mode = "wrong_split_seed"
    c, stop = _run_all(world)
    assert "its split records split_seed 1, not 0" in stop
    assert world.trainer_starts == 1
    assert not (c.work_dir / "comparison.json").exists()


def test_every_run_recording_the_same_wrong_draw_count_is_refused(world):
    world.default_mode = "wrong_draws"
    c, stop = _run_all(world)
    assert "its split records noise_draws 1, not 4" in stop
    assert not (c.work_dir / "comparison.json").exists()


@pytest.mark.parametrize(("mode", "reason"), [("wrong_fraction", "holdout_fraction 0.15, not 0.3"),
                                              ("wrong_bands", "timestep_bands 4, not 8")])
def test_a_wrong_holdout_fraction_or_band_count_is_refused(world, mode, reason):
    world.default_mode = mode
    c, stop = _run_all(world)
    assert reason in stop


def test_a_held_out_picture_the_copied_dataset_lacks_is_refused(world):
    world.default_mode = "foreign_picture"
    c, stop = _run_all(world)
    assert "Cars/img00.png" in stop and "is not the copied dataset's picture" in stop


def test_a_metrics_record_with_another_draw_count_is_refused(world):
    world.default_mode = "one_bad_record"
    c, stop = _run_all(world)
    assert "records noise_draws 1, not 4" in stop and "metrics" in stop


def test_the_same_wrong_split_is_refused_before_a_reuse(world):
    c = world.comparison()
    for arm, seed in plan.queue(c):
        world.place_finished(c, arm, seed, mode="wrong_split_seed")
    stop = runner.run_queue(c, world.seams())
    assert world.trainer_starts == 0
    assert "split_seed" in stop


def test_a_run_refused_for_its_sampler_is_never_reused(world):
    world.default_mode = "wrong_sampler"
    c, stop = _run_all(world)
    reason = "the run's sampler is 'radc', not 'uniform'"
    assert reason in stop
    verdict = json.loads((c.work_dir / "logs" / "uniform_s42.verdict.json").read_text(encoding="utf-8"))
    assert verdict["accepted"] is False
    assert runner.is_complete(runner.run_dir(c, "uniform_s42", DEFINITION), c)
    world.default_mode = "ok"
    starts = world.trainer_starts
    stop = runner.run_queue(c, world.seams())
    assert world.trainer_starts == starts and reason in stop
    isolation.write_json(c.work_dir / "logs" / "uniform_s42.verdict.json",
                         {**verdict, "accepted": True, "stopped_for": None})
    stop = runner.run_queue(c, world.seams())
    assert world.trainer_starts == starts and reason in stop


def test_a_run_refused_for_its_applied_seed_is_never_reused(world):
    world.default_mode = "wrong_seed"
    c, stop = _run_all(world)
    assert "the run applied seed 7, not 42" in stop
    world.default_mode = "ok"
    starts = world.trainer_starts
    stop = runner.run_queue(c, world.seams())
    assert world.trainer_starts == starts and "the run applied seed 7, not 42" in stop


def test_a_finished_run_without_a_verdict_is_refused(world):
    c = world.comparison()
    world.place_finished(c, "uniform", 42)
    (c.work_dir / "logs" / "uniform_s42.verdict.json").unlink()
    stop = runner.run_queue(c, world.seams())
    assert world.trainer_starts == 0 and "no recorded verdict" in stop


def test_a_partial_run_is_moved_aside_never_appended_to(world):
    world.default_mode = "partial"
    c, stop = _run_all(world)
    assert "incomplete" in stop
    world.default_mode = "ok"
    assert runner.run_queue(c, world.seams()) is None
    partial = list((c.work_dir / "partial").iterdir())
    assert len(partial) == 1
    moved = sorted(p.name for p in partial[0].iterdir())
    assert runner.run_dir(c, "uniform_s42", DEFINITION).name in moved
    assert "uniform_s42.log" in moved and "uniform_s42.identity.json" in moved
    steps = [json.loads(line)["step"] for line in
             (runner.run_dir(c, "uniform_s42", DEFINITION) / "validation_metrics.jsonl").read_text().splitlines()]
    assert steps == plan.passes(c.steps, c.validation.every)


def test_a_run_that_ends_without_every_pass_is_incomplete(world):
    world.default_mode = "partial"
    c, stop = _run_all(world)
    assert stop == "uniform_s42: incomplete metrics: passes [0, 500] of [0, 500, 1000, 1500, 1999]"
    assert _records(c)[0]["complete"] is False


def test_the_run_applied_its_own_seed(world):
    world.default_mode = "wrong_seed"
    c, stop = _run_all(world)
    assert stop == "uniform_s42: the run applied seed 7, not 42"


def test_the_status_file_carries_the_lowest_ram_between_writes(world):
    readings = iter([50 * GIB, 50 * GIB, 20 * GIB])
    c, stop = _run_all(world, available_ram_bytes=lambda: next(readings, 40 * GIB))
    assert stop is None
    status = json.loads((c.work_dir / "status.json").read_text(encoding="utf-8"))
    assert status["min_available_ram_gib"] == 20.0
