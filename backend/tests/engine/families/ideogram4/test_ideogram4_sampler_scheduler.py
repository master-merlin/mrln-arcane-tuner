"""Ideogram 4 sampler scheduler cache-key tests.

Regression: ``_get_scheduler`` only rebuilt when ``num_steps`` changed, but
``_Ideogram4FlowSchedule.mean`` is resolution-dependent (``mu + 0.5*log(area /
512^2)``) — ``self._height``/``self._width`` are re-stashed per prompt by
``_create_initial_noise``. Two prompts with the SAME step count but DIFFERENT
resolutions (the common case) served prompt 2+ the FIRST prompt's sigma
spacing. The code comment at :173 already promised "rebuild if the stashed
resolution changed"; the check was never written.
"""

from __future__ import annotations

from app.engine.models.families.ideogram4.sampler import IdeogramV4Sampler


def _bare_sampler(height: int, width: int) -> IdeogramV4Sampler:
    """Construct a sampler without running ``__init__`` (no pipeline needed —
    ``_get_scheduler`` only touches ``_scheduler``/``_height``/``_width``)."""
    s = object.__new__(IdeogramV4Sampler)
    s._scheduler = None
    s._height = height
    s._width = width
    return s


def test_scheduler_rebuilds_when_resolution_changes_at_same_step_count():
    s = _bare_sampler(512, 512)
    sched_512 = s._get_scheduler(30)

    s._height, s._width = 1024, 1024
    sched_1024 = s._get_scheduler(30)

    assert sched_512 is not sched_1024, (
        "same object served across a resolution change — stale sigma spacing "
        "for every prompt after the first"
    )
    assert sched_512.mean != sched_1024.mean
    assert sched_512.flow_times() != sched_1024.flow_times()


def test_scheduler_is_reused_when_neither_steps_nor_resolution_change():
    s = _bare_sampler(512, 512)
    sched_a = s._get_scheduler(30)
    sched_b = s._get_scheduler(30)
    assert sched_a is sched_b  # no unnecessary rebuild


def test_scheduler_still_rebuilds_on_step_count_change_alone():
    s = _bare_sampler(512, 512)
    sched_a = s._get_scheduler(20)
    sched_b = s._get_scheduler(30)
    assert sched_a is not sched_b
    assert sched_b.num_steps == 30


def test_scheduler_mean_matches_direct_construction_per_resolution():
    """Sanity: the rebuilt scheduler's mean matches a fresh instance built
    directly for that resolution (no drift from stale mu leaking through)."""
    from app.engine.models.families.ideogram4.sampler import _Ideogram4FlowSchedule

    s = _bare_sampler(512, 512)
    s._get_scheduler(30)
    s._height, s._width = 768, 1024
    rebuilt = s._get_scheduler(30)

    direct = _Ideogram4FlowSchedule(num_steps=30, height=768, width=1024)
    assert rebuilt.mean == direct.mean
