"""The backend suite scores through a double of the `hpsv2` library.

`backend/conftest.py` installs `tests/support/fake_hpsv2.py` in every test
process, so a scan stays real down to the library call and no test loads the
production model. The tests here hold that in place.
"""
import importlib
import sys

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
