"""Inert-when-off guarantee + live region timing for the train-loop profiler.

The profiler hooks in ``PipelineTrainMixin`` MUST be byte-identical to the old
loop when ``profile_steps`` is unset — these are the safety tests that protect
every normal training run. The live path (wall + CUDA-event region timing) is
exercised on CPU (CUDA events are guarded by ``torch.cuda.is_available()``), so
the test is GPU-agnostic.
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
    assert t._profiling_active == 0
    assert t._profiling_live is False
    # region is a true no-op context
    with t._prof_region("x"):
        pass
    # begin/end are no-ops when not armed
    t._profiling_maybe_begin(0)
    assert t._profiling_live is False
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
    # Goes live only AFTER the warmup boundary — not at init, not before warmup.
    assert t._profiling_live is False
    t._profiling_maybe_begin(0)
    assert t._profiling_live is False  # step 0 < warmup 3


def test_live_region_timing_accumulates_and_writes_report(tmp_path):
    pdir = tmp_path / "prof"
    t = _Trainer(
        {"profile_steps": 2, "profile_warmup": 0, "profile_dir": str(pdir)},
        str(tmp_path),
    )
    t._maybe_init_profiling()
    t._profiling_maybe_begin(0)  # warmup 0 → live immediately
    assert t._profiling_live is True
    # A live region is a real timer (not the no-op context) and accumulates.
    assert not isinstance(t._prof_region("data_prep"), type(nullcontext()))
    with t._prof_region("data_prep"):
        pass
    with t._prof_region("forward_loss"):
        pass
    assert t._region_count["data_prep"] == 1
    assert "forward_loss" in t._region_wall
    # Window: stop when step+1 >= warmup(0)+active(2) == 2.
    assert t._profiling_maybe_end(0) is False  # 0+1=1 < 2
    assert t._profiling_maybe_end(1) is True   # 1+1=2 >= 2 → writes report
    assert (pdir / "profile_summary.txt").exists()
    assert t._profiling_live is False
