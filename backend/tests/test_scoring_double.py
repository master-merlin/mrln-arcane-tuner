"""The backend suite scores through a double of the `hpsv2` library.

`backend/conftest.py` installs `tests/support/fake_hpsv2.py` in every test
process, so a scan stays real down to the library call and no test loads the
production model. The tests here hold that in place.
"""
import importlib
import importlib.metadata as md
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from tests.support import fake_hpsv2


def _real_hpsv2_modules() -> list[str]:
    return [name for name in sys.modules if name.startswith("hpsv2.")]


def test_every_test_process_gets_the_double():
    assert _real_hpsv2_modules() == []
    assert importlib.import_module("hpsv2") is fake_hpsv2
    assert _real_hpsv2_modules() == []


def test_score_image_goes_through_the_real_plugin(tmp_path):
    from app.core.scoring.scoring_service import ScoringService

    image = tmp_path / "a.png"
    Image.new("RGB", (8, 8), "red").save(image)

    ScoringService.reset_instance()
    try:
        score = ScoringService.get_instance().score_image(
            str(image), "hpsv2", {"hps_version": "v2.1", "prompt": "a red square"}
        )
        assert score == fake_hpsv2.FAKE_SCORE
        assert ScoringService._active_model_id == "hpsv2"
    finally:
        ScoringService.reset_instance()


def _parameters(signature: inspect.Signature) -> list[list[str]]:
    return [[p.name, p.kind.name, repr(p.default)] for p in signature.parameters.values()]


_SIGNATURE_PROBE = """
import inspect, json, hpsv2
print(json.dumps([[p.name, p.kind.name, repr(p.default)]
                  for p in inspect.signature(hpsv2.score).parameters.values()]))
"""


def test_double_matches_the_real_signature():
    """In a subprocess, which has no conftest, so it sees the real library.

    Importing it loads no weights. Kinds count: `HPSv2Model` passes
    `hps_version` by keyword.
    """
    try:
        md.version("hpsv2")
    except md.PackageNotFoundError:
        pytest.skip("hpsv2 is not installed in this environment")

    result = subprocess.run(
        [sys.executable, "-c", _SIGNATURE_PROBE],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    real = json.loads(result.stdout.strip().splitlines()[-1])
    assert _parameters(inspect.signature(fake_hpsv2.score)) == real


_REAL_SCORE_PROBE = """
import sys
from app.core.scoring.scoring_service import ScoringService
print(ScoringService.get_instance().score_image(
    sys.argv[1], "hpsv2", {"hps_version": "v2.1", "prompt": "a red square"}))
"""


@pytest.mark.xdist_group("gpu")
@pytest.mark.skipif(
    os.environ.get("MRLN_REAL_SCORER") != "1",
    reason="loads the real model (about 15 GB of commit, weights may download): "
    "set MRLN_REAL_SCORER=1 to run it",
)
def test_real_model_scores_one_image(tmp_path):
    image = tmp_path / "a.png"
    Image.new("RGB", (64, 64), "red").save(image)

    result = subprocess.run(
        [sys.executable, "-c", _REAL_SCORE_PROBE, str(image)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert math.isfinite(float(result.stdout.strip().splitlines()[-1]))
