"""`backend/scripts/compare_validation_runs.py`: exit codes, the JSON file, the default step."""
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import compare_validation_runs as cli  # noqa: E402

from app.engine.core import validation_compare as vc  # noqa: E402

CENTRES = [0.25, 0.75]
KEYS = [f"ds/img{i}.png" for i in range(6)]
SPLIT = json.dumps({"holdout": KEYS, "representatives": [{"key": k, "sha256": k} for k in KEYS],
                    "split_seed": 0, "global_triggerword": ""})


def _record(step, offset=0.0, seed=None):
    losses = [[[1.0 + offset + 0.01 * i, 0.5 + offset]] for i in range(len(KEYS))]
    band_loss = [sum(image[0][b] for image in losses) / len(losses) for b in range(2)]
    return {"step": step, "images": len(KEYS), "band_centres": CENTRES, "band_loss": band_loss,
            "mean": sum(band_loss) / 2, "worst_band": 0, "seconds": 1.0, "version": 2,
            "image_keys": KEYS, "noise_draws": 1, "train_seed": seed, "image_band_loss": losses}


def _folder(tmp_path, name, steps, offset=0.0):
    folder = tmp_path / name
    folder.mkdir()
    (folder / vc.METRICS_FILE).write_text("".join(json.dumps(_record(s, offset)) + "\n" for s in steps), encoding="utf-8")
    (folder / vc.SPLIT_FILE).write_text(SPLIT, encoding="utf-8")
    return folder


def _args(*pairs, extra=()):
    return [arg for arm, folder in pairs for arg in ("--run", f"{arm}={folder}")] + list(extra)


def test_a_report_exits_zero_and_prints_each_pair(tmp_path, capsys):
    a, b = _folder(tmp_path, "a", [0, 500]), _folder(tmp_path, "b", [0, 500], offset=0.2)
    assert cli.main(_args(("uniform", a), ("radc", b))) == 0
    out = capsys.readouterr().out
    assert "Validation comparison at step 500" in out
    assert "uniform - radc" in out


def test_a_refusal_exits_two_with_the_reason(tmp_path, capsys):
    a, b = _folder(tmp_path, "a", [0]), _folder(tmp_path, "b", [0])
    assert cli.main(_args(("uniform", a), ("uniform", b))) == 2
    assert "at least two arms" in capsys.readouterr().err


def test_an_unreadable_folder_exits_one_naming_the_path(tmp_path, capsys):
    a = _folder(tmp_path, "a", [0])
    missing = tmp_path / "missing"
    assert cli.main(_args(("uniform", a), ("radc", missing))) == 1
    err = capsys.readouterr().err
    assert str(missing / vc.SPLIT_FILE) in err


def test_the_json_file_holds_the_modules_report(tmp_path):
    a, b = _folder(tmp_path, "a", [0, 500]), _folder(tmp_path, "b", [0, 500], offset=0.2)
    target = tmp_path / "report.json"
    assert cli.main(_args(("uniform", a), ("radc", b), extra=["--json", str(target), "--resamples", "50"])) == 0
    expected = vc.compare([("uniform", a), ("radc", b)], resamples=50)
    assert json.loads(target.read_text(encoding="utf-8")) == json.loads(json.dumps(expected))


def test_the_json_file_is_written_atomically(tmp_path, monkeypatch):
    a, b = _folder(tmp_path, "a", [0]), _folder(tmp_path, "b", [0], offset=0.2)
    target = tmp_path / "report.json"
    target.write_text("earlier report", encoding="utf-8")
    replaced = []

    def failing_replace(source, destination):
        replaced.append((Path(source).parent, Path(destination)))
        raise OSError("disk full")

    monkeypatch.setattr(cli.os, "replace", failing_replace)
    assert cli.main(_args(("uniform", a), ("radc", b), extra=["--json", str(target)])) == 1
    assert replaced == [(tmp_path, target)]
    assert target.read_text(encoding="utf-8") == "earlier report"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a", "b", "report.json"]


def test_the_default_step_is_the_last_every_run_has(tmp_path, capsys):
    a, b = _folder(tmp_path, "a", [0, 500, 1000]), _folder(tmp_path, "b", [0, 500])
    target = tmp_path / "report.json"
    assert cli.main(_args(("uniform", a), ("radc", b), extra=["--json", str(target)])) == 0
    assert json.loads(target.read_text(encoding="utf-8"))["step"] == 500


def test_a_given_step_is_the_one_compared(tmp_path):
    a, b = _folder(tmp_path, "a", [0, 500, 1000]), _folder(tmp_path, "b", [0, 500])
    target = tmp_path / "report.json"
    assert cli.main(_args(("uniform", a), ("radc", b), extra=["--step", "0", "--json", str(target)])) == 0
    assert json.loads(target.read_text(encoding="utf-8"))["step"] == 0


@pytest.mark.parametrize(
    ("steps", "where"),
    [
        ([0, 500, 1000, 500], "line 4 repeats step 500"),
        ([0, 1000, 1999, 1500, 1600], "line 4 goes back to step 1500 after step 1999"),
    ],
)
def test_a_history_that_repeats_or_goes_back_exits_two(tmp_path, capsys, steps, where):
    a, b = _folder(tmp_path, "a", steps), _folder(tmp_path, "b", [0, 500])
    assert cli.main(_args(("uniform", a), ("radc", b))) == 2
    err = capsys.readouterr().err
    assert f"{a / vc.METRICS_FILE}: {where}" in err
    assert "resume into a fresh output folder" in err


def test_a_run_without_an_arm_name_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--run", str(tmp_path)])
    assert stopped.value.code == 2
