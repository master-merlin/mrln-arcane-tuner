"""Stands in for the `hpsv2` library in every backend test process.

One real score costs about 15 GB of commit and 3.77 GB on the GPU, and the model
stays alive in the library's module. Tests that scan a dataset need a score, not
that model. `FAKE_SCORE` differs from the 0.0 `HPSv2Model.score` returns for an
empty result. `test_scoring_double.py` holds the signature to the real library's.
"""
FAKE_SCORE = 1.0


def score(imgs_path, prompt, hps_version="v2.0"):
    return [FAKE_SCORE]
