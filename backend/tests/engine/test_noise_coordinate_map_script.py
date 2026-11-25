"""`backend/scripts/noise_coordinate_map.py`: the printed table, the JSON file, the exit codes."""
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import noise_coordinate_map as cli  # noqa: E402

from app.engine.core.noise_map import MapOptions, build_report, parse_sampler  # noqa: E402

pytestmark = pytest.mark.xdist_group("noise_map_registry")

SHORT = ["--steps", "64", "--phases", "2"]


def test_a_report_exits_zero_and_prints_both_models(capsys):
    code = cli.main(["--model", "flux3-image-klein-base", "--model", "moss-image-base",
                     "--sampler", "uniform", "--sampler", "default", *SHORT])
    out = capsys.readouterr().out
    assert code == 0
    assert "flux3-image-klein-base" in out and "moss-image-base" in out


def test_an_unknown_model_exits_two_naming_it(capsys):
    assert cli.main(["--model", "nope", *SHORT]) == 2
    assert "nope" in capsys.readouterr().err


def test_a_bad_sampler_exits_two_naming_the_bad_part(capsys):
    assert cli.main(["--model", "moss-image-base", "--sampler", "radc:centre=1", *SHORT]) == 2
    assert "centre" in capsys.readouterr().err


def test_nothing_mapped_exits_two_with_the_reasons(capsys):
    assert cli.main(["--model", "sdxl_base_1.0", "--sampler", "uniform", *SHORT]) == 2
    captured = capsys.readouterr()
    assert "Not mapped:" in captured.out and "flow-matching" in captured.out
    assert captured.err == "No row was mapped\n"


@pytest.mark.parametrize("bands", ["0", "1"])
def test_a_band_count_validation_refuses_exits_two(capsys, bands):
    assert cli.main(["--model", "moss-image-base", "--sampler", "uniform", "--bands", bands, *SHORT]) == 2
    assert "bands" in capsys.readouterr().err


def test_fewer_than_one_latent_frame_exits_two(capsys):
    assert cli.main(["--model", "moss-image-base", "--sampler", "uniform", "--latent-frames", "0", *SHORT]) == 2
    assert "latent_frames" in capsys.readouterr().err


def test_the_options_default_to_the_modules_defaults():
    args = cli._parser().parse_args([])
    defaults = MapOptions()
    assert (args.steps, args.phases, args.draws_per_step, args.bands, args.seed, args.latent_downscale,
            args.latent_frames) == (defaults.steps, defaults.phases, defaults.draws_per_step, defaults.bands,
                                    defaults.seed, defaults.latent_downscale, defaults.latent_frames)


def test_the_json_file_holds_the_modules_report(tmp_path):
    target = tmp_path / "map.json"
    code = cli.main(["--model", "moss-image-base", "--sampler", "uniform", "--resolution", "512",
                     "--json", str(target), *SHORT])
    assert code == 0
    expected = build_report(["moss-image-base"], [parse_sampler("uniform")], [512], steps=64, phases=2,
                            draws_per_step=4, bands=None, seed=0, latent_downscale=None, latent_frames=1)
    assert json.loads(target.read_text(encoding="utf-8")) == json.loads(json.dumps(expected))


def test_an_unwritable_json_path_exits_one(tmp_path, capsys):
    target = tmp_path / "missing" / "map.json"
    assert cli.main(["--model", "moss-image-base", "--sampler", "uniform", "--json", str(target), *SHORT]) == 1
    assert "map.json" in capsys.readouterr().err
