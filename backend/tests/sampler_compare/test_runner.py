"""One run and the queue, against the fake trainer and backend as real child processes. A test
that only needs finished runs on disk places them (``World.place_finished``) instead of training."""
import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from app.core.naming import model_part_from_definition_id
from app.core.sampler_compare import isolation, plan, runner
from app.core.sampler_compare.runner import port_taken
from app.core.sampler_compare.telemetry import Card
from app.engine.core import validation_compare as vc
from app.engine.core.pipeline.validation_pass import is_due

from . import fake_trainer
from .conftest import DEFINITION, GIB, FakeClock, started_pids

QUEUE = ["uniform_s42", "radc_s42", "uniform_s43", "radc_s43"]


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


# ── A whole queue, trained once ─────────────────────────────────────────────


def test_a_queue_of_two_arms_and_two_seeds_runs_to_the_end(ran_queue):
    world, c, stop = ran_queue
    assert stop is None
    records = _records(c)
    assert [r["label"] for r in records] == QUEUE
    assert all(r["complete"] and r["stopped_for"] is None and r["ran"] for r in records)
    assert world.trainer_starts == 4
    status = json.loads((c.work_dir / "status.json").read_text(encoding="utf-8"))
    assert status["version"] == 1 and status["queue_stop"] is None


def test_the_trainer_and_the_backend_get_the_comparison_port(ran_queue):
    _, c, _ = ran_queue
    started = _events(c, "uniform_s42", "fake_trainer_started")[0]
    assert started["port"] == str(c.port)
    assert _events(c, "uniform_s42", "fake_backend_started", "backend.log")[0]["port"] == str(c.port)
    assert started["db"].startswith(str(c.work_dir))
    assert started["offline"] == "1"


def test_the_telemetry_file_has_a_row_before_the_trainer_starts(ran_queue):
    _, c, _ = ran_queue
    assert _events(c, "uniform_s42", "fake_trainer_started")[0]["telemetry_lines"] >= 2
    lines = (c.work_dir / "logs" / "uniform_s42.telemetry.csv").read_text(encoding="utf-8").splitlines()
    assert lines[-1].endswith("ended: finished")


def test_the_identity_is_on_disk_before_the_trainer_starts(ran_queue):
    _, c, _ = ran_queue
    assert _events(c, "uniform_s42", "fake_trainer_started")[0]["identity_found"] is True


# ── Before a run ────────────────────────────────────────────────────────────


def test_the_fake_backend_outlasts_the_runners_port_probe(world, tmp_path):
    """The readiness wait probes the port with a plain bind (``port_taken``) while the backend
    starts; a backend binding in that instant must wait it out, as it does once the probe closes."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", world.port))  # what restart_launcher.port_is_free holds for an instant
    env = {**os.environ, "PORT": str(world.port), "MRLN_DB_PATH": str(world.db)}
    backend = subprocess.Popen([sys.executable, str(Path(fake_trainer.__file__).with_name("fake_backend.py"))],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        time.sleep(0.5)
        assert backend.poll() is None, backend.stdout.read()
        probe.close()
        deadline = time.monotonic() + 30
        while not port_taken(world.port):
            assert backend.poll() is None and time.monotonic() < deadline, backend.stdout.read()
            time.sleep(0.05)
    finally:
        probe.close()
        backend.kill()
        backend.wait(timeout=30)


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
    limits = {**json.loads(json.dumps(vars(c.limits))), "power_limit_w": None}
    assert runner.preflight(world.comparison(limits=limits), world.seams()) is None


def test_a_busy_card_is_waited_for_then_refused(world):
    world.card = Card(30000, 40.0, 100.0, 500.0)
    clock = FakeClock(scale=1.0)
    c, stop = _run_all(world, clock=clock.clock, sleep=clock.sleep)
    assert "the card is not idle after 30 min: 30000 MiB in use" in stop
    assert 30 * 60 <= clock.now <= 31 * 60
    assert world.trainer_starts == 0


# ── A stop leaves nothing running ───────────────────────────────────────────


def _assert_all_gone(c, label, trainer=True):
    pids = started_pids(c.work_dir / "logs" / f"{label}.backend.log", "fake_backend_started")
    if trainer:
        pids += started_pids(c.work_dir / "logs" / f"{label}.log", "fake_trainer_started")
    assert pids
    deadline = time.monotonic() + 10
    while any(psutil.pid_exists(p) and psutil.Process(p).status() != psutil.STATUS_ZOMBIE for p in pids):
        assert time.monotonic() < deadline, f"still alive: {pids}"
        time.sleep(0.1)
    assert not port_taken(c.port)


def test_no_child_survives_a_backend_that_never_answers(world, monkeypatch):
    monkeypatch.setenv("FAKE_BACKEND", "never_answers")
    c, stop = _run_all(world, ready_wait_s=3)  # real seconds: the backend gets to start, then is given up
    assert "did not answer within 0.05 min" in stop
    assert world.trainer_starts == 0
    _assert_all_gone(c, "uniform_s42", trainer=False)


def test_no_child_survives_a_trainer_that_fails(world):
    world.default_mode = "exit1"
    c, stop = _run_all(world)
    assert "the trainer exited with 1" in stop
    _assert_all_gone(c, "uniform_s42")


def test_no_child_survives_a_hot_card(world):
    world.default_mode = "hang"
    readings = iter([world.card] * 31)
    c, stop = _run_all(world, read_card=lambda: next(readings, Card(1000, 99.0, 100.0, 500.0)))
    assert "the card reads 99 C, over 94 C" in stop
    _assert_all_gone(c, "uniform_s42")


def test_no_child_survives_a_card_reader_that_raises(world):
    calls = {"n": 0}

    def read_card():
        calls["n"] += 1
        if calls["n"] > 3:
            raise RuntimeError("the card reader broke")
        return world.card

    world.default_mode = "hang"
    c = world.comparison()
    stop = runner.run_queue(c, world.seams(read_card=read_card))
    assert stop == "uniform_s42: the runner failed: RuntimeError: the card reader broke"
    _assert_all_gone(c, "uniform_s42")


def test_a_run_that_raises_leaves_its_verdict_its_record_and_a_final_status(world, monkeypatch):
    real_copy = runner.isolation.copy_database

    def broken(source, target, *args, **kwargs):
        if Path(target).name == Path(isolation.METADATA_SNAPSHOT).name:  # the comparison's snapshot is taken
            return real_copy(source, target, *args, **kwargs)
        raise RuntimeError("the database copy broke")

    monkeypatch.setattr(runner.isolation, "copy_database", broken)
    c, stop = _run_all(world)
    assert stop == "uniform_s42: the runner failed: RuntimeError: the database copy broke"
    assert world.trainer_starts == 0
    [record] = _records(c)
    assert record["label"] == "uniform_s42" and record["stopped_for"] == stop[len("uniform_s42: "):]
    assert record["error"] == "RuntimeError: the database copy broke"
    verdict = json.loads((c.work_dir / "logs" / "uniform_s42.verdict.json").read_text(encoding="utf-8"))
    assert verdict["accepted"] is False and "the database copy broke" in verdict["stopped_for"]
    status = json.loads((c.work_dir / "status.json").read_text(encoding="utf-8"))
    assert status["running"] is None and status["queue_stop"] == stop


def _grandchild_code(pid_file):
    """A child that starts a sleeping grandchild and writes the grandchild's pid."""
    return ("import pathlib, subprocess, sys, time\n"
            "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            f"pathlib.Path({str(pid_file)!r}).write_text(str(g.pid))\n"
            "time.sleep(120)\n")


def _wait_pid(pid_file):
    deadline = time.monotonic() + 30
    while not (pid_file.exists() and pid_file.read_text().strip()):
        assert time.monotonic() < deadline, "the child started no grandchild"
        time.sleep(0.05)
    return int(pid_file.read_text())


def _tree_of(pid):
    root = psutil.Process(pid)
    return [root, *root.children(recursive=True)]


def _one_cpu():
    """A mask other than this process's own wherever the host has two CPUs."""
    return [psutil.Process().cpu_affinity()[-1]]


def test_every_process_a_child_starts_runs_on_the_configured_cpus(tmp_path):
    before = psutil.Process().cpu_affinity()
    mask = _one_cpu()
    children = runner._Children(mask)
    pid_file = tmp_path / "grandchild.pid"
    with open(tmp_path / "child.log", "w", encoding="utf-8") as log:
        try:
            proc = children.start([sys.executable, "-c", _grandchild_code(pid_file)], dict(os.environ), log)
            grandchild = _wait_pid(pid_file)
            assert psutil.Process().cpu_affinity() == before
            for p in [*_tree_of(proc.pid), *_tree_of(grandchild)]:
                assert p.cpu_affinity() == mask, p
        finally:
            children.stop()


def test_a_child_is_started_already_on_the_configured_cpus(tmp_path, monkeypatch):
    """A pin after the start races a venv launcher's own child: the mask is in place before it."""
    before = psutil.Process().cpu_affinity()
    mask = _one_cpu()
    real_popen, at_start = subprocess.Popen, []

    def popen(*args, **kwargs):
        at_start.append(psutil.Process().cpu_affinity())
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    children = runner._Children(mask)
    with open(tmp_path / "child.log", "w", encoding="utf-8") as log:
        try:
            children.start([sys.executable, "-c", "import time; time.sleep(30)"], dict(os.environ), log)
        finally:
            children.stop()
    assert at_start == [mask]
    assert psutil.Process().cpu_affinity() == before


def test_a_poll_pins_every_process_seen_under_a_child(tmp_path):
    before = psutil.Process().cpu_affinity()
    if len(before) < 2:
        pytest.skip("one CPU: no other mask to set")
    mask = _one_cpu()
    pid_file = tmp_path / "grandchild.pid"
    proc = subprocess.Popen([sys.executable, "-c", _grandchild_code(pid_file)], stdout=subprocess.DEVNULL)
    children = runner._Children(mask)
    children.procs.append(proc)
    try:
        grandchild = _wait_pid(pid_file)
        assert psutil.Process(grandchild).cpu_affinity() == before
        children.note(proc)
        for p in _tree_of(proc.pid):
            assert p.cpu_affinity() == mask, p
    finally:
        children.stop()


def test_a_card_that_cant_be_told_apart_refuses_the_run(world):
    c, stop = _run_all(world, card_problem="2 cards: name one")
    assert stop == "uniform_s42: before the run: 2 cards: name one"
    assert world.backend_starts == 0 and world.trainer_starts == 0


def test_a_progress_line_written_in_two_halves_is_read_once(tmp_path):
    path = tmp_path / "trainer.log"
    path.write_text('{"event": "step_progress", "st', encoding="utf-8")
    with open(path, encoding="utf-8") as reader:
        progress = runner._StepReader(reader)
        progress.poll()
        assert (progress.step, progress.events) == (0, 0)
        with open(path, "a", encoding="utf-8") as out:
            out.write('ep": 5, "step_time": 0.5}\n{"event": "step_progress", "step": 6')
        progress.poll()
        assert (progress.step, progress.step_time, progress.events) == (5, 0.5, 1)
        with open(path, "a", encoding="utf-8") as out:
            out.write(', "step_time": 0.4}\n')
        progress.poll()
        assert (progress.step, progress.step_time, progress.events) == (6, 0.4, 2)


def test_a_stalled_run_is_stopped(world):
    world.default_mode = "hang"
    clock = FakeClock(scale=500.0)
    # The fast clock is for the stall limit; the backend's start keeps its real seconds.
    c, stop = _run_all(world, clock=clock.clock, sleep=clock.sleep, ready_wait_s=1e9)
    assert "no log output for 15 min" in stop
    _assert_all_gone(c, "uniform_s42")


class _Alive:
    returncode = None

    def poll(self):
        return None


def test_the_ready_wait_asks_nothing_before_the_port_is_bound(world, monkeypatch, tmp_path):
    def no_request(*args, **kwargs):
        raise AssertionError("a request went to a port nothing listens on")

    monkeypatch.setattr(runner.urllib.request, "urlopen", no_request)
    clock = FakeClock(scale=1.0)
    seams = world.seams(port_answers=lambda port: False, clock=clock.clock, sleep=clock.sleep, ready_wait_s=60,
                        poll_s=5)
    why = runner._wait_ready(world.comparison(), seams, _Alive(), "Cars", tmp_path / "backend.log")
    assert why == f"the comparison's backend did not answer within 1 min; see {tmp_path / 'backend.log'}"
    assert 60 <= clock.now < 61


def test_the_ready_wait_never_bind_probes_a_started_backend(world, monkeypatch):
    """A bind probe meeting the backend's own bind makes uvicorn's bind fail as in use: once the
    backend is started, its port is only connected to."""
    real_free, started, during = runner.restart_launcher.port_is_free, [], []

    def port_is_free(host, port):
        if started and not started[-1]:
            during.append(port)
        return real_free(host, port)

    def backend_argv(c):
        started.append(False)
        return world.backend_argv(c)

    def trainer_argv(c, definition_id, config):
        started[-1] = True  # the readiness wait is over
        return world.trainer_argv(c, definition_id, config)

    monkeypatch.setattr(runner.restart_launcher, "port_is_free", port_is_free)
    c = world.comparison(seeds=[42, 43])
    stop = runner.run_queue(c, world.seams(backend_argv=backend_argv, trainer_argv=trainer_argv))
    assert stop is None and len(started) == 4
    assert during == []


def test_the_port_wait_runs_on_the_seams_clock(world):
    clock = FakeClock(scale=10.0)
    asked = []
    seams = world.seams(port_taken=lambda port: asked.append(port) or True, clock=clock.clock, sleep=clock.sleep)
    c = world.comparison()
    assert runner._wait_port_free(c, seams) is False
    assert runner.PORT_FREE_WAIT_S <= clock.now < runner.PORT_FREE_WAIT_S + 3
    clock.now, asked[:] = 0.0, []
    free = world.seams(port_taken=lambda port: asked.append(port) or False, clock=clock.clock, sleep=clock.sleep)
    assert runner._wait_port_free(c, free) is True
    assert asked == [c.port] and clock.now == 0.0


# ── What a run measured ─────────────────────────────────────────────────────


def test_a_run_with_another_sampler_stops_the_queue(world):
    world.default_mode = "wrong_sampler"
    c, stop = _run_all(world)
    assert stop == "uniform_s42: the run's sampler is 'radc', not 'uniform'"
    assert world.trainer_starts == 1


def test_a_run_whose_split_differs_from_the_first_stops_the_queue(world):
    c = world.comparison()
    world.place_finished(c, "uniform", 42)
    world.modes["radc_s42"] = "other_split"
    stop = runner.run_queue(c, world.seams())
    assert stop.startswith("radc_s42: ")
    assert "not the same measurement as" in stop
    assert world.trainer_starts == 1


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


def test_a_run_that_ends_without_every_pass_is_incomplete(world):
    world.default_mode = "partial"
    c, stop = _run_all(world)
    assert stop == "uniform_s42: incomplete metrics: passes [0, 500] of [0, 500, 1000, 1500, 1999]"
    assert _records(c)[0]["complete"] is False


def test_the_run_applied_its_own_seed(world):
    world.default_mode = "wrong_seed"
    c, stop = _run_all(world)
    assert stop == "uniform_s42: the run applied seed 7, not 42"


# ── Finished runs on disk ───────────────────────────────────────────────────


def test_a_skipped_complete_run_gets_the_same_split_check(world):
    c = world.comparison()
    world.place_all(c, modes={"radc_s42": "other_split"})
    stop = runner.run_queue(c, world.seams())
    assert world.trainer_starts == 0
    assert "radc_s42" in stop and "not the same measurement as" in stop


def test_the_first_run_cannot_replace_the_recorded_measurement(world):
    c = world.comparison()
    world.place_all(c)
    assert runner.run_queue(c, world.seams()) is None
    recorded = (c.work_dir / "comparison.json").read_bytes()
    uniform = runner.run_dir(c, "uniform_s42", DEFINITION)
    content = json.loads((uniform / "validation_split.json").read_text(encoding="utf-8"))
    content["global_triggerword"] = "changed"
    (uniform / "validation_split.json").write_text(json.dumps(content), encoding="utf-8")
    stop = runner.run_queue(c, world.seams())
    assert stop is not None and "uniform_s42" in stop
    assert (c.work_dir / "comparison.json").read_bytes() == recorded
    assert world.trainer_starts == 0


def _folders(c):
    return {p.name: sorted(x.name for x in p.iterdir()) for p in (c.work_dir / "runs").iterdir()}


def test_a_changed_reference_setting_refuses_the_finished_runs(world):
    c = world.comparison()
    world.place_all(c)
    before = _folders(c)
    world.set_base(learning_rate=0.0002)
    stop = runner.run_queue(c, world.seams())
    assert world.trainer_starts == 0
    assert "uniform_s42" in stop and "config.learning_rate" in stop
    assert _folders(c) == before


def _local_override(folder):
    return {"models": {"overrides": {DEFINITION: {"source_type": "local_diffusers", "local_path": str(folder)}}}}


def _weights(folder, content=b"weights A"):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "transformer.safetensors").write_bytes(content)
    return folder


@pytest.mark.parametrize("change", ["local_path", "weights", "source_type", "local_files_only"])
def test_a_changed_model_source_refuses_the_finished_runs(world, tmp_path, change):
    model_a = _weights(tmp_path / "model-a")
    world.set_settings(_local_override(model_a))
    c = world.comparison()
    world.place_all(c)
    before = _folders(c)
    if change == "local_path":
        changed = _local_override(_weights(tmp_path / "model-b", b"weights B"))
    elif change == "weights":
        _weights(model_a, b"other weights, same folder")
        changed = _local_override(model_a)
    elif change == "source_type":
        changed = {"models": {"overrides": {}}}
    else:
        world.set_settings({"models": {"overrides": {}}})
        world.place_all(c := world.comparison())
        before = _folders(c)
        changed = {"models": {"global_offline_mode": True, "overrides": {}}}
    world.set_settings(changed)
    settings_copy = c.work_dir / isolation.ISOLATED_PATHS["MRLN_SETTINGS_PATH"]
    copied_before = settings_copy.read_bytes() if settings_copy.exists() else None
    stop = runner.run_queue(c, world.seams())
    assert world.trainer_starts == 0
    assert "uniform_s42" in stop and f"model_source.{change}" in stop
    assert _folders(c) == before
    assert (settings_copy.read_bytes() if settings_copy.exists() else None) == copied_before


def _enabled(db_path):
    db = sqlite3.connect(db_path)
    try:
        return db.execute("select count(*) from media_items where enabled=1").fetchone()[0]
    finally:
        db.close()


def test_every_arm_trains_on_one_metadata_snapshot(world):
    real_argv = world.trainer_argv

    def argv(c, definition_id, config):
        if world.trainer_starts == 1:  # after the first arm: the live backend switches a picture off
            world.disable_image("img03.png")
        return real_argv(c, definition_id, config)

    c = world.comparison()
    stop = runner.run_queue(c, world.seams(trainer_argv=argv))
    assert stop is None and world.trainer_starts == 4
    assert _enabled(world.db) == 19
    run_dbs = sorted(p for p in (c.work_dir / "data" / "db").glob("*_s*.db"))
    assert len(run_dbs) == 4
    assert [_enabled(p) for p in run_dbs] == [20, 20, 20, 20]


def test_a_dataset_copy_changed_between_arms_stops_the_queue(world):
    real_argv = world.trainer_argv

    def argv(c, definition_id, config):
        if world.trainer_starts == 0:  # while the first arm trains, its caption is edited in the copy
            (c.work_dir / isolation.DATASET_FOLDER / "img05.txt").write_text("edited", encoding="utf-8")
        return real_argv(c, definition_id, config)

    c = world.comparison()
    stop = runner.run_queue(c, world.seams(trainer_argv=argv))
    assert world.trainer_starts == 1
    assert stop.startswith("radc_s42: before the run: the dataset copy") and "changed img05.txt" in stop


def test_a_resume_trains_on_the_first_metadata_snapshot(world):
    c = world.comparison()
    world.place_all(c, skip={"radc_s43"})
    world.disable_image("img03.png")
    world._prepared = []
    stop = runner.run_queue(c, world.seams())
    assert stop is None and world.trainer_starts == 1
    assert [_enabled(p) for p in (c.work_dir / "data" / "db").glob("radc_s43*.db")] == [20]


def test_a_replaced_metadata_snapshot_refuses_the_finished_runs(world):
    c = world.comparison()
    world.place_all(c)
    before = _folders(c)
    world.disable_image("img03.png")
    world._prepared = []
    (c.work_dir / isolation.METADATA_SNAPSHOT).unlink()
    stop = runner.run_queue(c, world.seams())
    assert world.trainer_starts == 0
    assert "uniform_s42" in stop and "dataset_metadata_digest" in stop
    assert _folders(c) == before


def test_every_refused_finished_run_is_named_before_anything_trains(world):
    c = world.comparison()
    world.place_all(c, skip={"uniform_s42"})
    world.set_base(learning_rate=0.0002)
    stop = runner.run_queue(c, world.seams())
    assert world.trainer_starts == 0
    assert [name for name in QUEUE if f"{name} (run earlier)" in stop] == QUEUE[1:]
    assert stop.count("config.learning_rate") == 3


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
    world.place_all(world.comparison())
    stop = runner.run_queue(world.comparison(**change), world.seams())
    assert world.trainer_starts == 0
    assert key in stop


def test_a_finished_run_without_an_identity_is_refused(world):
    c = world.comparison()
    world.place_finished(c, "uniform", 42, identity=False)
    stop = runner.run_queue(c, world.seams())
    assert world.trainer_starts == 0
    assert "uniform_s42" in stop and "no recorded identity" in stop


def test_an_unchanged_experiment_reuses_its_finished_runs(world):
    c = world.comparison()
    world.place_all(c)
    assert runner.run_queue(c, world.seams()) is None
    assert world.trainer_starts == 0
    reused = [r for r in _records(c) if r.get("reused")]
    assert [r["label"] for r in reused] == QUEUE
    assert all(r["complete"] for r in reused)


def test_the_same_wrong_split_is_refused_before_a_reuse(world):
    c = world.comparison()
    world.place_all(c, mode="wrong_split_seed")
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
    world.place_all(c, skip={"uniform_s42"})
    assert runner.run_queue(c, world.seams()) is None
    assert world.trainer_starts == 2
    partial = list((c.work_dir / "partial").iterdir())
    assert len(partial) == 1
    moved = sorted(p.name for p in partial[0].iterdir())
    assert runner.run_dir(c, "uniform_s42", DEFINITION).name in moved
    assert "uniform_s42.log" in moved and "uniform_s42.identity.json" in moved
    steps = [json.loads(line)["step"] for line in
             (runner.run_dir(c, "uniform_s42", DEFINITION) / "validation_metrics.jsonl").read_text().splitlines()]
    assert steps == plan.passes(c.steps, c.validation.every)


def _work_files(c):
    """Every path under the work folder -> (bytes, mtime) of a file, or None for a folder; the lock
    file another process holds can't be read on Windows: its size and mtime stand for it."""
    out = {}
    for path in sorted(c.work_dir.rglob("*")):
        if path.name == runner.LOCK_FILE:
            content = (path.stat().st_size, path.stat().st_mtime_ns)
        else:
            content = (path.read_bytes(), path.stat().st_mtime_ns) if path.is_file() else None
        out[path.relative_to(c.work_dir).as_posix()] = content
    return out


def _partial_on_disk(world):
    """A comparison whose first run is partial on disk, as an active run looks from outside."""
    world.default_mode = "partial"
    c, stop = _run_all(world)
    assert "incomplete" in stop
    world.default_mode = "ok"
    world.trainer_starts = world.backend_starts = 0
    return c


def _hold_lock(c, tmp_path):
    """A real child holding ``c``'s work-folder lock until it is killed."""
    code = ("import sys, time; sys.path.insert(0, sys.argv[1])\n"
            "from app.core.sampler_compare import plan, runner\n"
            "with runner.work_lock(plan.load_comparison(sys.argv[2])):\n"
            "    print('locked', flush=True); time.sleep(120)\n")
    child = subprocess.Popen([sys.executable, "-c", code, str(plan.BACKEND), str(tmp_path / "comparison.json")],
                             stdout=subprocess.PIPE, text=True, cwd=tmp_path)
    assert child.stdout.readline().strip() == "locked"
    return child


def test_a_second_launch_while_the_first_holds_the_work_folder_moves_nothing(world, tmp_path):
    c = _partial_on_disk(world)
    child = _hold_lock(c, tmp_path)
    try:
        before = _work_files(c)
        stop = runner.run_queue(c, world.seams())
        assert "another comparison is running in" in stop and str(c.work_dir) in stop
        assert world.trainer_starts == 0 and world.backend_starts == 0
        assert _work_files(c) == before
    finally:
        child.kill()
        child.wait(timeout=30)
    deadline = time.monotonic() + 30  # Windows drops a killed process's locks soon after, not at once
    while runner.lock_problem(c) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert runner.run_queue(c, world.seams()) is None  # the lock goes with its process


def test_a_taken_port_is_checked_before_a_partial_run_is_moved(world):
    c = _partial_on_disk(world)
    folder = runner.run_dir(c, "uniform_s42", DEFINITION)
    before = sorted(p.name for p in folder.iterdir())
    owner = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    owner.bind(("127.0.0.1", world.port))
    owner.listen()
    try:
        stop = runner.run_queue(c, world.seams())
    finally:
        owner.close()
    assert "already taken" in stop
    assert sorted(p.name for p in folder.iterdir()) == before
    assert list((c.work_dir / "partial").iterdir()) == []


def test_the_status_file_carries_the_lowest_ram_between_writes(world):
    c = world.comparison()
    world.place_all(c, skip={"uniform_s42"})
    readings = iter([50 * GIB, 50 * GIB, 20 * GIB])
    stop = runner.run_queue(c, world.seams(available_ram_bytes=lambda: next(readings, 40 * GIB)))
    assert stop is None and world.trainer_starts == 1
    status = json.loads((c.work_dir / "status.json").read_text(encoding="utf-8"))
    assert status["min_available_ram_gib"] == 20.0


# ── Facts the runner copies from the product ────────────────────────────────


def test_the_fake_trainer_runs_the_products_passes():
    config = {"max_train_steps": 2000, "validation_every_n_steps": 500}
    assert fake_trainer.passes(config) == [s for s in range(2000) if is_due(s, 500, 2000)]


def test_the_run_folder_is_the_one_the_trainer_writes(world):
    c = world.comparison()
    names = plan.run_names(c, "radc", 42)
    expected = runner.BACKEND / "app" / "engine" / "core" / "pipeline" / "pipeline_optimization.py"
    source = expected.read_text(encoding="utf-8")
    assert re.search(r'run_name = f"\{lora_name\}_\{model_part\}"', source)
    assert "os.path.join(output_root, run_name)" in source
    folder = runner.run_dir(c, "radc_s42", DEFINITION)
    assert folder == Path(names["output_dir"]) / f"{names['lora_name']}_{model_part_from_definition_id(DEFINITION)}"


def _split(**changes):
    split = {"representatives": [{"key": "Cars/a.png", "path": "D:/x/a.png", "sha256": "1" * 64}],
             "holdout": ["Cars/a.png"], "split_seed": 0, "global_triggerword": "", "noise_draws": 4}
    return json.dumps({**split, **changes})


@pytest.mark.parametrize(
    "changes",
    [{}, {"noise_draws": 1}, {"representatives": [{"key": "Cars/a.png", "path": "E:/moved/a.png", "sha256": "1" * 64}]},
     {"split_seed": 1}, {"holdout": ["Cars/b.png"]}, {"global_triggerword": "ohwx"},
     {"representatives": [{"key": "Cars/a.png", "path": "D:/x/a.png", "sha256": "2" * 64}]}],
    ids=["same", "draws", "moved", "seed", "holdout", "trigger", "digest"],
)
def test_the_recorded_measurement_digest_agrees_with_same_measurement(changes):
    earlier, content = _split(), _split(**changes)
    assert (runner._measured_digest(earlier) == runner._measured_digest(content)) == \
        vc.same_measurement(earlier, content)
