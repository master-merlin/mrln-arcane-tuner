"""Regression tests for the HF download-progress tqdm class.

Root cause of "'functools.partial' object has no attribute 'get_lock'":
``snapshot_download`` fetches files concurrently via
``tqdm.contrib.concurrent.thread_map`` → ``ensure_lock(tqdm_class)`` →
``tqdm_class.get_lock()`` (a classmethod). A ``functools.partial`` wrapper
hides inherited classmethods, so the download aborts. The fix binds the WS
metadata with a real subclass via ``make_progress_tqdm`` instead.
"""
import functools

import pytest
from tqdm.contrib.concurrent import ensure_lock

from app.api.events.download_progress import WSProgressTqdm, make_progress_tqdm


def test_make_progress_tqdm_returns_subclass_with_get_lock():
    cls = make_progress_tqdm(source="hf", model_id="org/repo", category="training")
    assert issubclass(cls, WSProgressTqdm)
    # get_lock is the classmethod huggingface_hub calls during concurrent
    # snapshot downloads — it must be reachable on the bound class.
    lock = cls.get_lock()
    assert lock is not None


def test_bound_class_survives_hf_concurrent_ensure_lock():
    # Faithful reproduction of snapshot_download's concurrent path.
    cls = make_progress_tqdm(source="hf", model_id="org/repo", category="training")
    with ensure_lock(cls) as lk:
        assert lk is not None


def test_partial_wrapper_would_break_hf_concurrent_lock():
    # Documents the original bug: a functools.partial has no get_lock, so the
    # exact call snapshot_download makes raises AttributeError.
    bad = functools.partial(
        WSProgressTqdm, source="hf", model_id="org/repo", category="training"
    )
    assert not hasattr(bad, "get_lock")
    with pytest.raises(AttributeError):
        with ensure_lock(bad):
            pass


def test_bound_class_binds_metadata_and_instantiates():
    # HF instantiates the class with its own kwargs (no source/model_id/
    # category); the subclass must supply the bound metadata and construct
    # without an event loop (emits are no-ops in subprocess/test context).
    cls = make_progress_tqdm(source="hf", model_id="org/repo", category="training")
    bar = cls(total=100, disable=True)  # disable=True: no terminal output in tests
    try:
        assert bar._meta_source == "hf"
        assert bar._meta_model_id == "org/repo"
        assert bar._meta_category == "training"
        bar.update(10)  # must not raise even with no app loop captured
    finally:
        bar.close()
