"""``config['seed']`` (when set) must actually seed python/torch/CUDA.

``saver_base.py:149`` maps ``seed`` -> ``ss_seed`` in the saved LoRA metadata,
advertising reproducibility, but nothing previously called
``random.seed``/``torch.manual_seed`` anywhere in the pipeline — runs were
irreproducible while the metadata claimed otherwise.

``PipelineBaseMixin._apply_run_seed`` (called first thing in ``setup()``,
before any RNG consumer draws) closes that gap. Seeding is OPT-IN: an unset
seed must leave today's nondeterministic behavior completely untouched — no
raise, and no perturbation of global ``random``/``torch`` state.
"""

from __future__ import annotations

import random

import structlog
import torch

from app.engine.core.pipeline.pipeline_base import PipelineBaseMixin


class _Harness(PipelineBaseMixin):
    """Concrete shell exposing only ``_apply_run_seed`` — no real family,
    driver, or definition wiring needed for this unit."""

    def _setup_family(self):  # pragma: no cover - abstract stub
        pass

    async def setup(self):  # pragma: no cover - abstract stub
        pass

    async def load_model(self):  # pragma: no cover - abstract stub
        pass

    async def prepare_data(self):  # pragma: no cover - abstract stub
        pass

    async def train(self):  # pragma: no cover - abstract stub
        pass


def _make(seed=None) -> _Harness:
    t = object.__new__(_Harness)
    t.config = {} if seed is None else {"seed": seed}
    t.logger = structlog.get_logger("test_run_seed_applied")
    return t


def _draw() -> tuple[float, list[float]]:
    return (random.random(), torch.rand(3).tolist())


def test_configured_seed_yields_identical_sequences_across_setups():
    """Two independent 'setups' with the same configured seed must draw the
    exact same random/torch sequence afterward."""
    t1 = _make(seed=1234)
    t1._apply_run_seed()
    seq1 = _draw()

    t2 = _make(seed=1234)
    t2._apply_run_seed()
    seq2 = _draw()

    assert seq1 == seq2


def test_different_configured_seeds_yield_different_sequences():
    """Sanity check that the seed is actually driving the draw (not a
    coincidental match / no-op)."""
    t1 = _make(seed=1234)
    t1._apply_run_seed()
    seq1 = _draw()

    t2 = _make(seed=5678)
    t2._apply_run_seed()
    seq2 = _draw()

    assert seq1 != seq2


def test_no_seed_key_does_not_raise_and_does_not_touch_global_rng():
    """OPT-IN guarantee: an unset seed must be a pure no-op on global RNG
    state — proven by comparing against a baseline that never calls
    ``_apply_run_seed`` at all from the same known starting state."""
    random.seed(2026)
    torch.manual_seed(2026)
    baseline = _draw()

    random.seed(2026)
    torch.manual_seed(2026)
    t = _make(seed=None)
    t._apply_run_seed()  # must not raise
    after = _draw()

    assert after == baseline


def test_empty_string_seed_is_also_treated_as_unset():
    """An empty-string seed (e.g. from an unfilled form field) must behave
    identically to a missing seed key — no seeding, no raise."""
    random.seed(2026)
    torch.manual_seed(2026)
    baseline = _draw()

    random.seed(2026)
    torch.manual_seed(2026)
    t = _make(seed="")
    t._apply_run_seed()  # must not raise
    after = _draw()

    assert after == baseline


def test_zero_is_a_valid_configured_seed_not_treated_as_unset():
    """seed=0 is falsy but a legitimate, explicit seed request — must NOT be
    coalesced into the unset/no-op path."""
    random.seed(999)
    torch.manual_seed(999)
    unseeded_baseline = _draw()

    t1 = _make(seed=0)
    t1._apply_run_seed()
    seq1 = _draw()

    t2 = _make(seed=0)
    t2._apply_run_seed()
    seq2 = _draw()

    assert seq1 == seq2
    assert seq1 != unseeded_baseline
