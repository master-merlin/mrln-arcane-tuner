"""Bernini-R 14B TRAINER wiring for dual-expert start placement (Wave-3
whole-branch review, should-fix Finding 2 — "bernini's half of T2 is
unpinned").

``test_bernini_r_expert_placement.py`` exercises only the DRIVER methods
(``place_experts_for_start`` / ``_set_active``) directly, by its own
docstring. Nothing in the suite asserted that ``BerniniRTrainer.
_configure_optimization`` actually CALLS ``driver.place_experts_for_start()``
or sets ``driver.block_swap_active_expert`` before doing so — the two lines
added by Task W3.T2 / the wave-3 block-swap-aware follow-up
(``backend/app/engine/models/families/bernini_r/trainer.py``, in
``_configure_optimization``):

    if getattr(self, "_block_swap_managers", None):
        driver.block_swap_active_expert = driver.active_expert
    driver.place_experts_for_start()

Deleting those two lines left the WHOLE existing bernini_r test suite green
(confirmed manually before this file existed) — the original defect (the
first router flip landing on a CPU-resident low expert when sampling is
disabled) would silently return on any refactor. This module mirrors
``TestConfigureOptimizationPlacesExperts`` and
``TestConfigureOptimizationBlockSwapHandoff`` from
``backend/tests/engine/families/wan22/test_wan22_expert_placement.py``,
adapted to :class:`BerniniRTrainer` / :class:`BerniniRDriver`. The
``_make_recorder_trainer`` harness is ported directly from that file — same
shape, same fake-wiring approach (``_DeviceRecorder`` modules whose ``.to()``
records the requested device instead of moving, so placement is observable
without a real GPU).
"""

from __future__ import annotations

import structlog
import torch
import torch.nn as nn

from app.engine.models.families.bernini_r.driver import BerniniRDriver
from app.engine.models.families.bernini_r.trainer import BerniniRTrainer
from app.engine.models.families.wan22.expert_router import HIGH, ExpertRouter


# ── Fixtures ───────────────────────────────────────────────────────────────


class _Defn:
    architecture_params = {
        "mode": "t2v",
        "dual_expert": True,
        "switch_dit_boundary": 0.875,
    }
    lora_targetable_modules: list[str] = []


def _base_config(swap_mode="resident", sample_every_n_steps=0):
    return {
        "network_rank": 4,
        "network_alpha": 4,
        "timestep_sampling": "uniform",
        "expert_switch_interval": 1,
        "expert_swap_mode": swap_mode,
        "mixed_precision": "bf16",
        "optimizer_type": "AdamW",
        "learning_rate": 1e-4,
        "seed": 0,
        "sample_every_n_steps": sample_every_n_steps,
    }


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


def _make_recorder_trainer(swap_mode="resident", sample_every_n_steps=0):
    """Fake-wired dual trainer with device-recorder experts — isolates the
    PLACEMENT effect from PEFT wrapping (which needs real Linear layers).
    Ported from ``test_wan22_expert_placement.py``'s harness of the same
    name."""
    t = object.__new__(BerniniRTrainer)
    t.logger = structlog.get_logger("test")
    t.device = torch.device("cpu")
    t.config = _base_config(
        swap_mode=swap_mode, sample_every_n_steps=sample_every_n_steps
    )
    t.components = {}
    driver = BerniniRDriver(_Defn(), t.device)
    assert driver.is_dual
    high, low = _DeviceRecorder(), _DeviceRecorder()
    driver.transformer_high = high
    driver.transformer_low = low
    driver.transformer = high
    driver._active_expert = HIGH
    driver.configure_swap_mode(swap_mode)
    t.driver = driver
    t.transformer = high
    router = ExpertRouter(
        boundary=driver.boundary,
        switch_interval=1,
        timestep_cfg=t.config,
        seed=0,
        mc_samples=20_000,
    )
    t.expert_router = router
    driver.router = router  # bypass set_router — avoid setup-time .to() noise
    # Clear any placement recorded by construction so the assertions below
    # attribute calls solely to _configure_optimization.
    high.to_calls.clear()
    low.to_calls.clear()
    return t, high, low


# ── (a) _configure_optimization actually calls place_experts_for_start ─────


class TestConfigureOptimizationPlacesExperts:
    """Mirrors wan22's class of the same name: pins that the TRAINER — not
    just the driver in isolation — wires ``place_experts_for_start()`` into
    the real optimizer-configuration hook."""

    def test_place_experts_for_start_is_called(self, monkeypatch):
        """Zero coverage today of the TRAINER call site — only the driver's
        own methods were exercised directly by
        ``test_bernini_r_expert_placement.py``."""
        t, high, low = _make_recorder_trainer()

        calls: list[int] = []
        monkeypatch.setattr(
            t.driver, "place_experts_for_start", lambda: calls.append(1)
        )
        t._configure_optimization(max_train_steps=10)

        assert calls == [1], "place_experts_for_start() was never called"

    def test_configure_optimization_places_both_experts(self):
        """The ONLY prior placement mechanism was the step-0 sampler's
        device-ensure loop — skipped when ``sample_every_n_steps=0``. With
        that OFF, both experts must still land on the device via
        ``_configure_optimization`` alone (end-to-end, no monkeypatched
        driver method — the real ``place_experts_for_start`` runs)."""
        t, high, low = _make_recorder_trainer(sample_every_n_steps=0)

        t._configure_optimization(max_train_steps=10)

        assert high.to_calls, "high expert never placed for start"
        assert low.to_calls, "deferred low expert never placed for start"
        assert high.to_calls[-1] == str(t.device)
        assert low.to_calls[-1] == str(t.device)

    def test_swap_mode_places_only_active_on_device(self):
        """In ``swap`` mode, placement puts the ACTIVE expert on-device and
        the inactive one on CPU."""
        t, high, low = _make_recorder_trainer(swap_mode="swap")

        t._configure_optimization(max_train_steps=10)

        assert high.to_calls and high.to_calls[-1] == str(t.device)
        assert low.to_calls and low.to_calls[-1] == "cpu"


# ── (b) Block-swap handoff — the trainer is the only thing that can see
#        ``_block_swap_managers`` (lives on the pipeline/trainer) ──────────


class TestConfigureOptimizationBlockSwapHandoff:
    """Mirrors wan22's class of the same name: the trainer must hand the
    ACTIVE expert's name to the driver (``block_swap_active_expert``) so
    placement can skip it — this is exactly the pair of lines Finding 2
    says nothing pinned at the trainer level."""

    def test_hands_off_active_expert_when_block_swap_managers_exist(self):
        t, high, low = _make_recorder_trainer()
        # Simulate _configure_block_swapping() having swapped some blocks of
        # the (currently active) primary model.
        t._block_swap_managers = [object()]

        t._configure_optimization(max_train_steps=10)

        assert t.driver.block_swap_active_expert == HIGH
        assert high.to_calls == [], (
            "block-swap-managed HIGH expert must not be bulk-moved by "
            "_configure_optimization's placement call"
        )
        assert low.to_calls and low.to_calls[-1] == str(t.device), (
            "the deferred low expert must still be placed"
        )

    def test_no_managers_leaves_flag_none_and_places_both(self):
        t, high, low = _make_recorder_trainer()

        t._configure_optimization(max_train_steps=10)

        assert t.driver.block_swap_active_expert is None
        assert high.to_calls and low.to_calls
