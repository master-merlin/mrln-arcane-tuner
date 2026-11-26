"""`backend/scripts/compare_samplers.py`: the dry run, the report-only mode, the exit codes."""
import json
import socket
import subprocess
import sys

import pytest

from app.core.sampler_compare import plan
from app.core.sampler_compare.telemetry import Card

from .conftest import write_config

SCRIPT = plan.BACKEND / "scripts" / "compare_samplers.py"
EXAMPLE = plan.BACKEND / "scripts" / "compare_samplers.example.json"
sys.path.insert(0, str(SCRIPT.parent))
import compare_samplers as cli  # noqa: E402


def _config(world, **changes):
    changes.setdefault("port", world.port)
    return str(write_config(world.root, **changes))


def _no_children(world):
    def refuse(*args, **kwargs):
        raise AssertionError("a dry run started a child")

    return world.seams(backend_argv=refuse, trainer_argv=refuse)


def test_the_example_config_loads(world, capsys):
    c = plan.load_comparison(EXAMPLE)
    assert c.arms == {
        "uniform": {"timestep_sampling": "uniform"},
        "logit_normal": {"timestep_sampling": "logit_normal", "logit_normal_mu": 0.0, "logit_normal_sigma": 1.0},
        "radc": {"timestep_sampling": "radc", "radc_start": 0.9, "radc_end": 0.3, "radc_width": 0.5,
                 "radc_res_influence": 0.15, "radc_uniform_prob": 0.0},
        "family_default": {},
    }
    example = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    copy = world.root / "example.json"
    copy.write_text(json.dumps({**example, "work_dir": str(world.root / "work"), "port": world.port}),
                    encoding="utf-8")
    assert cli.main([str(copy), "--dry-run"], seams=_no_children(world)) == 1
    out = capsys.readouterr().out
    assert "no job" in out and example["reference_job"] in out


def test_a_dry_run_starts_no_process(world, capsys):
    assert cli.main([_config(world), "--dry-run"], seams=_no_children(world)) == 0
    out = capsys.readouterr().out
    assert "uniform_s42, radc_s42, uniform_s43, radc_s43" in out
    assert "radc: timestep_sampling radc from the arm" in out
    assert f"port {world.port}: free" in out
    assert "copy plan:" in out and str(world.dataset) in out
    assert out.rstrip().endswith("dry run: OK")
    assert world.backend_starts == 0 and world.trainer_starts == 0


def test_a_dry_run_names_every_problem_and_exits_one(world, capsys):
    world.card = Card(1000, 40.0, 100.0, 450.0)
    owner = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    owner.bind(("127.0.0.1", world.port))
    owner.listen()
    try:
        assert cli.main([_config(world), "--dry-run"], seams=_no_children(world)) == 1
    finally:
        owner.close()
    out = capsys.readouterr().out
    assert f"port {world.port} is already taken" in out
    assert "the power limit reads 450 W, not 500 W" in out
    assert out.rstrip().endswith("dry run: PROBLEMS above")


def test_a_dry_run_names_a_missing_reference_job(world, capsys):
    assert cli.main([_config(world, reference_job="no-such-job"), "--dry-run"], seams=_no_children(world)) == 1
    out = capsys.readouterr().out
    assert "no job" in out and "no-such-job" in out


def test_a_dry_run_refuses_a_reference_job_of_two_datasets(world, capsys):
    two = {**world.base, "datasets": [{"dataset_name": "Cars"}, {"dataset_name": "Other"}]}
    world.write_db(extra_jobs={"job-2": two})
    assert cli.main([_config(world, reference_job="job-2"), "--dry-run"], seams=_no_children(world)) == 1
    assert "2 dataset entries, not one" in capsys.readouterr().out


def test_a_dry_run_says_whether_downloads_are_allowed(world, capsys):
    cli.main([_config(world), "--dry-run"], seams=_no_children(world))
    assert "downloads off" in capsys.readouterr().out
    cli.main([_config(world, allow_downloads=True), "--dry-run"], seams=_no_children(world))
    out = capsys.readouterr().out
    assert "downloads on" in out and "the shared weight cache" in out and "is written to" in out


def test_a_dry_run_names_finished_runs_it_would_refuse(world, capsys):
    path = _config(world)
    c = plan.load_comparison(path)
    for arm, seed in plan.queue(c):
        world.place_finished(c, arm, seed)
    world.set_base(learning_rate=0.0002)
    assert cli.main([path, "--dry-run"], seams=_no_children(world)) == 1
    out = capsys.readouterr().out
    assert "uniform_s42" in out and "config.learning_rate" in out


def test_a_bad_config_exits_two_with_the_reason(world, capsys):
    bad = world.root / "bad.json"
    bad.write_text(json.dumps({"version": 1}), encoding="utf-8")
    assert cli.main([str(bad)], seams=_no_children(world)) == 2
    assert "lacks the key 'name'" in capsys.readouterr().err


def test_report_only_trains_nothing(world):
    path = _config(world)
    c = plan.load_comparison(path)
    for arm, seed in plan.queue(c):
        world.place_finished(c, arm, seed)
    assert cli.main([path, "--report-only"], seams=_no_children(world)) == 0
    out = json.loads((c.work_dir / "report.json").read_text(encoding="utf-8"))
    assert len(out["pairs"]) == 1 and out["queue_stop"] is None


def test_the_exit_code_is_one_when_the_queue_stopped(world):
    world.card = Card(1000, 40.0, 100.0, 450.0)
    path = _config(world)
    assert cli.main([path], seams=world.seams()) == 1
    c = plan.load_comparison(path)
    out = json.loads((c.work_dir / "report.json").read_text(encoding="utf-8"))
    assert "the power limit reads 450 W" in out["queue_stop"]
    assert world.trainer_starts == 0


@pytest.mark.parametrize("flags", [["--dry-run"], ["--dry-run", "--report-only"]])
def test_the_script_runs_from_any_working_directory(tmp_path, flags):
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    done = subprocess.run([sys.executable, str(SCRIPT), str(bad), *flags], cwd=tmp_path,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 2, done.stderr
    assert "Traceback" not in done.stderr
