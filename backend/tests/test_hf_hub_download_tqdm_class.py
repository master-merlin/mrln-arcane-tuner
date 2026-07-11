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
    """The engine resolver's real online ``hf_hub_download`` callsite now
    lives in the stall guard's child worker (``hf_fetch_worker`` — the
    resolver routes online downloads through the killable guard), so THAT is
    where the forbidden-kwarg regression must be pinned."""
    from app.engine.utils import hf_fetch_worker

    calls: list[dict] = []

    def fake_dl(*_args, **kwargs):
        calls.append(kwargs)
        return "/tmp/merges.txt"

    with patch.object(hf_fetch_worker, "hf_hub_download", side_effect=fake_dl):
        hf_fetch_worker.run_download(
            {"repo_id": "facebook/sam3", "filename": "merges.txt"},
        )

    assert calls, "hf_hub_download was never called"
    for kwargs in calls:
        assert "tqdm_class" not in kwargs, (
            f"hf_hub_download received forbidden kwarg `tqdm_class`: "
            f"kwargs={sorted(kwargs)}"
        )


def test_model_resolver_offline_file_download_omits_tqdm_class():
    """The resolver's remaining IN-PROCESS ``hf_hub_download`` call (the
    ``local_files_only`` cache-only path) must not pass ``tqdm_class``
    either."""
    from app.engine.utils import model_utils

    calls: list[dict] = []

    def fake_dl(*_args, **kwargs):
        calls.append(kwargs)
        raise FileNotFoundError("not in local cache")

    with patch.object(model_utils, "hf_hub_download", side_effect=fake_dl):
        with pytest.raises(FileNotFoundError):
            model_utils.ModelPathResolver._resolve_hf(
                "huggingface:facebook/sam3:merges.txt", local_files_only=True,
            )

    assert calls, "hf_hub_download was never called"
    for kwargs in calls:
        assert "tqdm_class" not in kwargs, (
            f"hf_hub_download received forbidden kwarg `tqdm_class`: "
            f"kwargs={sorted(kwargs)}"
        )
