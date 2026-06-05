"""Regression: `tqdm_class` must NOT be passed to ``hf_hub_download``.

huggingface_hub >= 0.36 dropped ``tqdm_class`` from ``hf_hub_download`` (only
``snapshot_download`` still accepts it). Passing it raises:

    TypeError: hf_hub_download() got an unexpected keyword argument 'tqdm_class'

For SAM 3 this surfaced as a misleading "accept the SAM 3 license" error and
broke every mass-masking item; for the engine model resolver it would abort
single-file downloads. ``with_progress`` still emits coarse start/complete
events, so the only loss is the per-chunk download bar.

These tests mock ``hf_hub_download`` at each callsite and assert ``tqdm_class``
is absent from every real (non ``local_files_only``) download call.
"""

from unittest.mock import patch

import pytest


def test_sam3_resolve_bpe_omits_tqdm_class():
    from app.core.masking.models import sam3

    captured: dict = {}

    def fake_dl(*_args, **kwargs):
        captured["kwargs"] = kwargs
        # Short-circuit before the gzip/cache write — we only care about kwargs.
        raise RuntimeError("stop after capturing kwargs")

    with patch("huggingface_hub.hf_hub_download", side_effect=fake_dl):
        with pytest.raises(FileNotFoundError):
            sam3._resolve_bpe_path()

    assert "kwargs" in captured, "hf_hub_download was never called"
    assert "tqdm_class" not in captured["kwargs"], (
        f"hf_hub_download received forbidden kwarg `tqdm_class`: "
        f"kwargs={sorted(captured['kwargs'])}"
    )


def test_model_resolver_file_download_omits_tqdm_class():
    from app.engine.utils import model_utils

    calls: list[dict] = []

    def fake_dl(*_args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("not in local cache")
        return "/tmp/merges.txt"

    with patch.object(model_utils, "hf_hub_download", side_effect=fake_dl):
        model_utils.ModelPathResolver._resolve_hf("huggingface:facebook/sam3:merges.txt")

    real_calls = [k for k in calls if not k.get("local_files_only")]
    assert real_calls, "real (non-local) hf_hub_download was never called"
    for kwargs in real_calls:
        assert "tqdm_class" not in kwargs, (
            f"hf_hub_download received forbidden kwarg `tqdm_class`: "
            f"kwargs={sorted(kwargs)}"
        )
