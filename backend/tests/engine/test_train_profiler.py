"""Inert-when-off guarantee + config parsing for the train-loop profiler hooks.

The profiler instrumentation in ``PipelineTrainMixin`` MUST be byte-identical to
the old loop when ``profile_steps`` is unset — these are the safety tests that
protect every normal training run. The armed path is exercised lightly (config
parse + dir creation) without starting a real profiler, to keep the test GPU-free
and fast.
"""

import os
from contextlib import nullcontext

import structlog

from app.engine.core.pipeline.pipeline_train import PipelineTrainMixin


class _CkptStub:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir


class _Trainer(PipelineTrainMixin):
    """Minimal carrier of the profiler helpers (mixin has no __init__)."""

    def __init__(self, config: dict, output_dir: str) -> None:
        self.config = config
        self.logger = structlog.get_logger("test")
        self.checkpoint_manager = _CkptStub(output_dir)


def test_profiling_inert_when_unset(tmp_path):
    t = _Trainer({}, str(tmp_path))
    t._maybe_init_profiling()
    assert t._profiler is None
    assert t._profiling_active == 0
    # region is a true no-op context
    with t._prof_region("x"):
        pass
    # begin/end are no-ops when not armed
    t._profiling_maybe_begin(0)
    assert t._profiler is None
    assert t._profiling_maybe_end(0) is False
    assert t._profiling_maybe_end(10_000) is False


def test_prof_region_is_nullcontext_when_off(tmp_path):
    t = _Trainer({}, str(tmp_path))
    t._maybe_init_profiling()
    assert isinstance(t._prof_region("data_prep"), type(nullcontext()))


def test_profiling_armed_parses_config_and_makes_dir(tmp_path):
    pdir = tmp_path / "prof"
    t = _Trainer(
        {"profile_steps": 8, "profile_warmup": 3, "profile_dir": str(pdir)},
        str(tmp_path),
    )
    t._maybe_init_profiling()
    assert t._profiling_active == 8
    assert t._profiling_warmup == 3
    assert os.path.isdir(str(pdir))
    # Profiler is created lazily AFTER warmup steps — not at init, and not
    # before the warmup boundary.
    assert t._profiler is None
    t._profiling_maybe_begin(0)
    assert t._profiler is None  # step 0 < warmup 3 → still not started
