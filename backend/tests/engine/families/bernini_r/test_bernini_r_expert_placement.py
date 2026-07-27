"""Bernini-R 14B dual-expert BLOCK-SWAP-AWARE start placement.

Mirrors ``test_wan22_expert_placement.py``'s block-swap-aware coverage
(wave-3 review finding, fix follow-up to Task W3.T2): ``place_experts_for_start()``
and the ``_set_active`` device guard must NOT bulk-``.to(device)`` an expert
whose deep blocks are under active ``BlockSwappingManager`` management
(``driver.block_swap_active_expert``) — that would force every swapped block
back onto GPU at once, defeating the swap. The non-swapped expert must still
be placed, preserving the original W3.T2 fix (first router flip must not land
on a CPU-resident model).

Bernini-R never had a dedicated placement test file before this fix — this
module covers only the new block-swap-aware behavior (the base placement
behavior mirrors wan22's, already covered there and by
``test_bernini_r_driver.py`` / ``test_bernini_r_trainer.py`` for
non-placement concerns).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from app.engine.models.families.bernini_r.driver import BerniniRDriver
from app.engine.models.families.wan22.expert_router import HIGH, LOW


class _Defn:
    architecture_params = {
        "mode": "t2v",
        "dual_expert": True,
        "switch_dit_boundary": 0.875,
    }
    lora_targetable_modules: list[str] = []


class _RecordingLogger:
    def __init__(self):
        self.infos: list[tuple[str, dict]] = []

    def warning(self, event, **kw):
        pass

    def info(self, event=None, **kw):
        self.infos.append((event, kw))


class _DeviceRecorder(nn.Module):
    """A module whose ``.to()`` records the requested device instead of
    moving (so we can observe placement without a real GPU)."""

    def __init__(self):
        super().__init__()
        self.to_calls: list[str] = []
        self.p = nn.Parameter(torch.zeros(2), requires_grad=True)

    def to(self, *args, **kwargs):  # noqa: A003 - shadow intentional
        target = args[0] if args else kwargs.get("device")
        self.to_calls.append(str(target))
        return self


def _make_dual_driver() -> BerniniRDriver:
    driver = BerniniRDriver(_Defn(), torch.device("cuda:0"))
    assert driver.is_dual
    driver.configure_swap_mode("resident")
    return driver


class TestBlockSwapAwareStartPlacement:
    def test_place_experts_for_start_skips_the_block_swapped_expert(self):
        driver = _make_dual_driver()
        high, low = _DeviceRecorder(), _DeviceRecorder()
        driver.transformer_high = high
        driver.transformer_low = low
        driver._active_expert = HIGH
        driver.block_swap_active_expert = HIGH
        rec = _RecordingLogger()
        driver.logger = rec

        driver.place_experts_for_start()

        assert high.to_calls == [], "block-swap-managed expert must not be bulk-moved"
        assert low.to_calls and low.to_calls[-1] == str(driver.device), (
            "the non-swapped expert must still be placed"
        )
        events = [ev for ev, _ in rec.infos]
        assert "expert_block_swap_placement_skipped" in events

    def test_place_experts_for_start_places_both_when_block_swap_inactive(self):
        """Regression: default (``block_swap_active_expert is None``) still
        places both experts (the original W3.T2 fix, unperturbed)."""
        driver = _make_dual_driver()
        high, low = _DeviceRecorder(), _DeviceRecorder()
        driver.transformer_high = high
        driver.transformer_low = low
        driver._active_expert = HIGH
        assert driver.block_swap_active_expert is None

        driver.place_experts_for_start()

        assert high.to_calls and high.to_calls[-1] == str(driver.device)
        assert low.to_calls and low.to_calls[-1] == str(driver.device)

    def test_set_active_guard_skips_the_block_swapped_expert(self):
        driver = _make_dual_driver()
        high, low = _DeviceRecorder(), _DeviceRecorder()
        driver.transformer_high = high
        driver.transformer_low = low
        driver._active_expert = HIGH
        driver.block_swap_active_expert = LOW
        rec = _RecordingLogger()
        driver.logger = rec

        driver._set_active(LOW)

        assert low.to_calls == [], (
            "the _set_active guard must not bulk-move a block-swap-managed "
            "expert even when it becomes active mid-run"
        )
        events = [ev for ev, _ in rec.infos]
        assert "expert_block_swap_placement_skipped" in events

    def test_set_active_guard_unaffected_when_a_different_expert_is_swapped(self):
        driver = _make_dual_driver()
        high, low = _DeviceRecorder(), _DeviceRecorder()
        driver.transformer_high = high
        driver.transformer_low = low
        driver._active_expert = HIGH
        driver.block_swap_active_expert = HIGH  # LOW is unaffected

        driver._set_active(LOW)

        assert low.to_calls and low.to_calls[-1] == str(driver.device)
