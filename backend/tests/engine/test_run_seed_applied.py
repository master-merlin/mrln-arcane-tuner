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
from structlog.testing import capture_logs

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


# ── Malformed seed (Finding 1, W2.T7 review): must never crash setup() ──────
#
# `config['seed']` is an untyped dict entry -- no Pydantic-typed run-level
# `seed` field exists -- so a stray value from a legacy job config or a
# hand-edited/imported template is a reachable input. Seeding is an opt-in
# reproducibility nicety; it must never be the reason a training run cannot
# start. A malformed seed must warn and proceed UNSEEDED, exactly as if no
# seed were configured.


def test_non_numeric_string_seed_warns_and_proceeds_unseeded():
    random.seed(2026)
    torch.manual_seed(2026)
    baseline = _draw()

    random.seed(2026)
    torch.manual_seed(2026)
    t = _make(seed="abc")
    with capture_logs() as logs:
        t._apply_run_seed()  # must not raise
    after = _draw()

    assert after == baseline  # proceeded unseeded, RNG state untouched
    assert any(
        e.get("event") == "run_seed_invalid_ignored"
        and e.get("configured_seed") == "abc"
        for e in logs
    ), f"expected a run_seed_invalid_ignored warning naming 'abc', got {logs}"


def test_whitespace_only_string_seed_warns_and_proceeds_unseeded():
    random.seed(2026)
    torch.manual_seed(2026)
    baseline = _draw()

    random.seed(2026)
    torch.manual_seed(2026)
    t = _make(seed="   ")
    with capture_logs() as logs:
        t._apply_run_seed()  # must not raise
    after = _draw()

    assert after == baseline
    assert any(e.get("event") == "run_seed_invalid_ignored" for e in logs)


def test_float_shaped_string_seed_is_rejected_not_truncated():
    """`"12.5"` is REJECTED (warn + unseeded), not truncated to 12.

    Decision: rejecting is safer and more predictable than silently
    truncating -- a float-shaped string is far more likely to be a form/JSON
    round-trip mistake (or a different field's value leaking in) than a
    deliberate request for seed 12. Truncation would also mean two visibly
    different configured values (`"12.5"` and `"12.9"`) silently collapse to
    the identical seed, which is a worse surprise than an unseeded run.
    """
    random.seed(2026)
    torch.manual_seed(2026)
    baseline = _draw()

    random.seed(2026)
    torch.manual_seed(2026)
    t = _make(seed="12.5")
    with capture_logs() as logs:
        t._apply_run_seed()  # must not raise
    after = _draw()

    assert after == baseline
    assert any(
        e.get("event") == "run_seed_invalid_ignored"
        and e.get("configured_seed") == "12.5"
        for e in logs
    )


def test_numeric_string_seed_seeds_deterministically():
    """A numeric string (e.g. round-tripped through JSON/YAML config) is a
    normal, valid seed shape -- must seed identically to the equivalent int."""
    t1 = _make(seed="1234")
    t1._apply_run_seed()
    seq1 = _draw()

    t2 = _make(seed=1234)
    t2._apply_run_seed()
    seq2 = _draw()

    assert seq1 == seq2
