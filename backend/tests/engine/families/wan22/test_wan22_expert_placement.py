"""WAN 2.2 dual-expert START PLACEMENT wiring (Task W3.T2).

``place_experts_for_start()`` existed with ZERO callers anywhere in the
backend. The only thing that ever moved the DEFERRED low expert onto the
training device was the step-0 baseline SAMPLER's ``_ensure_transformer_on_device``
loop (``sampler.py``) — skipped whenever ``sample_every_n_steps: 0``,
``sample_before_training: false``, step-0 sampling raises (swallowed), or
block swapping is active. Any of those left the low expert CPU-resident until
the first router flip hit it mid-forward: a device-mismatch ``RuntimeError``.

This module pins:

* ``_configure_optimization`` (dual path) calls ``driver.place_experts_for_start()``
  AFTER PEFT + the optimizer exist — independent of whether a sampler runs.
* Both experts actually land on the device when sampling is disabled.
* ``configure_swap_mode("auto")`` warns (``expert_swap_auto_unimplemented``)
  and resolves to ``"resident"`` — there never was a resident-vs-swap VRAM
  probe.
* ``_set_active``'s device guard (a belt-and-suspenders safety net) moves a
  CPU-resident expert onto the device in ``resident`` mode, rather than
  handing back a stale-device model as the new primary.
"""

from __future__ import annotations

import structlog
import torch
import torch.nn as nn

from app.engine.models.families.wan22.driver import Wan22Driver
from app.engine.models.families.wan22.expert_router import HIGH, LOW, ExpertRouter
from app.engine.models.families.wan22.trainer import Wan22Trainer


# ── Fixtures ───────────────────────────────────────────────────────────────


class _TinyBlock(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.attn1 = nn.Module()
        self.attn1.to_q = nn.Linear(dim, dim)
        self.attn1.to_k = nn.Linear(dim, dim)
        self.attn1.to_v = nn.Linear(dim, dim)


class _TinyWan(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock(dim)])


class _Defn:
    architecture_params = {"mode": "t2v", "moe.boundary_ratio": 0.875}
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


def _make_trainer(swap_mode="resident"):
    """Real ``_TinyWan`` experts (LoRA-targetable) — mirrors the dual-adapter
    fixture so ``_apply_peft`` + ``_configure_optimization`` run for real."""
    t = object.__new__(Wan22Trainer)
    t.logger = structlog.get_logger("test")
    t.device = torch.device("cpu")
    t.config = _base_config(swap_mode=swap_mode)
    t.components = {}
    driver = Wan22Driver(_Defn(), t.device)
    driver.assign_components({"unet": _TinyWan(), "unet_low": _TinyWan()})
    t.driver = driver
    t.transformer = driver.get_primary_model()
    router = ExpertRouter(
        boundary=driver.boundary,
        switch_interval=1,
        timestep_cfg=t.config,
        seed=0,
        mc_samples=20_000,
    )
    t.expert_router = router
    driver.set_router(router)
    driver.configure_swap_mode(swap_mode)
    return t


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
    PLACEMENT effect from PEFT wrapping (which needs real Linear layers)."""
    t = object.__new__(Wan22Trainer)
    t.logger = structlog.get_logger("test")
    t.device = torch.device("cpu")
    t.config = _base_config(
        swap_mode=swap_mode, sample_every_n_steps=sample_every_n_steps
    )
    t.components = {}
    driver = Wan22Driver(_Defn(), t.device)
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


# ── (a) _configure_optimization calls place_experts_for_start ─────────────


class TestConfigureOptimizationPlacesExperts:
    def test_place_experts_for_start_is_called(self, monkeypatch):
        """Zero callers today — `_configure_optimization` must invoke it."""
        t = _make_trainer()
        t._apply_peft()

        calls: list[int] = []
        monkeypatch.setattr(
            t.driver, "place_experts_for_start", lambda: calls.append(1)
        )
        t._configure_optimization(max_train_steps=10)

        assert calls == [1], "place_experts_for_start() was never called"

    def test_placement_happens_after_optimizer_exists(self, monkeypatch):
        """Hook-order requirement: placement runs AFTER PEFT + optimizer, not
        before (mirrors the deferred-expert-load hook-order reasoning)."""
        t = _make_trainer()
        t._apply_peft()

        seen_optimizer_ready: list[bool] = []

        def _spy():
            seen_optimizer_ready.append(getattr(t, "optimizer", None) is not None)

        monkeypatch.setattr(t.driver, "place_experts_for_start", _spy)
        t._configure_optimization(max_train_steps=10)

        assert seen_optimizer_ready == [True], (
            "placement ran before the optimizer existed"
        )


# ── (a) Both experts actually land on the device without sampling ─────────


class TestBothExpertsPlacedWithoutSampling:
    def test_configure_optimization_places_both_experts(self):
        """The ONLY prior placement mechanism was the step-0 sampler's
        device-ensure loop — skipped when `sample_every_n_steps=0`. With that
        OFF, both experts must still land on the device via
        `_configure_optimization` alone."""
        t, high, low = _make_recorder_trainer(sample_every_n_steps=0)

        t._configure_optimization(max_train_steps=10)

        assert high.to_calls, "high expert never placed for start"
        assert low.to_calls, "deferred low expert never placed for start"
        assert high.to_calls[-1] == str(t.device)
        assert low.to_calls[-1] == str(t.device)

    def test_swap_mode_places_only_active_on_device(self):
        """In ``swap`` mode, placement puts the ACTIVE expert on-device and
        the inactive one on CPU (place_experts_for_start's own swap branch —
        exercised here now that it's actually reachable)."""
        t, high, low = _make_recorder_trainer(swap_mode="swap")

        t._configure_optimization(max_train_steps=10)

        assert high.to_calls and high.to_calls[-1] == str(t.device)
        assert low.to_calls and low.to_calls[-1] == "cpu"


# ── (b) configure_swap_mode("auto") warns and resolves to resident ────────


class _RecordingLogger:
    def __init__(self):
        self.warnings: list[tuple[str, dict]] = []
        self.infos: list[tuple[str, dict]] = []

    def warning(self, event, **kw):
        self.warnings.append((event, kw))

    def info(self, event=None, **kw):
        self.infos.append((event, kw))


class TestAutoSwapModeStopsLying:
    def test_auto_resolves_to_resident(self):
        driver = Wan22Driver(_Defn(), torch.device("cpu"))
        driver.configure_swap_mode("auto")
        assert driver.swap_mode == "resident"

    def test_auto_warns_once(self):
        driver = Wan22Driver(_Defn(), torch.device("cpu"))
        rec = _RecordingLogger()
        driver.logger = rec
        driver.configure_swap_mode("auto")
        events = [ev for ev, _ in rec.warnings]
        assert "expert_swap_auto_unimplemented" in events

    def test_resident_and_swap_do_not_warn(self):
        driver = Wan22Driver(_Defn(), torch.device("cpu"))
        rec = _RecordingLogger()
        driver.logger = rec
        driver.configure_swap_mode("resident")
        driver.configure_swap_mode("swap")
        assert rec.warnings == []


# ── (c) _set_active device guard (resident mode safety net) ───────────────


class TestSetActiveDeviceGuard:
    def test_moves_cpu_resident_expert_in_resident_mode(self):
        """A CPU-resident expert flipped active in resident mode must be
        placed on the device — the safety net alongside
        `place_experts_for_start`."""
        driver = Wan22Driver(_Defn(), torch.device("cuda:0"))
        driver.configure_swap_mode("resident")
        high, low = _DeviceRecorder(), _DeviceRecorder()
        driver.transformer_high = high
        driver.transformer_low = low
        driver._active_expert = HIGH

        driver._set_active(LOW)

        assert low.to_calls, "CPU-resident expert was never placed on the device"
        assert low.to_calls[-1] == str(driver.device)

    def test_swap_mode_does_not_trigger_the_guard(self):
        """In ``swap`` mode placement is `_swap_to`'s job; the guard must stay
        out of the way (no double-move / conflicting semantics)."""
        driver = Wan22Driver(_Defn(), torch.device("cuda:0"))
        driver.configure_swap_mode("swap")
        high, low = _DeviceRecorder(), _DeviceRecorder()
        driver.transformer_high = high
        driver.transformer_low = low
        driver._active_expert = HIGH

        driver._set_active(LOW)

        assert low.to_calls == [], "guard must not fire outside resident mode"

    def test_already_placed_expert_is_a_noop(self):
        """No spurious `.to()` call when the expert is already on-device —
        the guard only acts on an actual mismatch."""
        driver = Wan22Driver(_Defn(), torch.device("cpu"))
        driver.configure_swap_mode("resident")
        high, low = _DeviceRecorder(), _DeviceRecorder()
        driver.transformer_high = high
        driver.transformer_low = low
        driver._active_expert = HIGH

        driver._set_active(LOW)  # low's real param is already on "cpu"

        assert low.to_calls == [], "guard fired even though device matched"


# ── (d) Block-swap-aware placement (Wave-3 review finding) ────────────────
#
# `place_experts_for_start()` / `_set_active`'s bulk `.to(device)` must NOT
# fire for an expert whose blocks are under active `BlockSwappingManager`
# management (`driver.block_swap_active_expert`) — that would force every
# swapped block back onto GPU at once, defeating the swap. The OTHER
# (non-swapped) expert must still be placed, preserving the original W3.T2
# fix for the non-block-swap case.


class TestBlockSwapAwareStartPlacement:
    def test_place_experts_for_start_skips_the_block_swapped_expert(self):
        driver = Wan22Driver(_Defn(), torch.device("cuda:0"))
        driver.configure_swap_mode("resident")
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
        """Regression: default (``block_swap_active_expert is None``) is
        byte-identical to the original W3.T2 fix — both experts placed."""
        driver = Wan22Driver(_Defn(), torch.device("cuda:0"))
        driver.configure_swap_mode("resident")
        high, low = _DeviceRecorder(), _DeviceRecorder()
        driver.transformer_high = high
        driver.transformer_low = low
        driver._active_expert = HIGH
        assert driver.block_swap_active_expert is None

        driver.place_experts_for_start()

        assert high.to_calls and high.to_calls[-1] == str(driver.device)
        assert low.to_calls and low.to_calls[-1] == str(driver.device)

    def test_set_active_guard_skips_the_block_swapped_expert(self):
        driver = Wan22Driver(_Defn(), torch.device("cuda:0"))
        driver.configure_swap_mode("resident")
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
        driver = Wan22Driver(_Defn(), torch.device("cuda:0"))
        driver.configure_swap_mode("resident")
        high, low = _DeviceRecorder(), _DeviceRecorder()
        driver.transformer_high = high
        driver.transformer_low = low
        driver._active_expert = HIGH
        driver.block_swap_active_expert = HIGH  # LOW is unaffected

        driver._set_active(LOW)

        assert low.to_calls and low.to_calls[-1] == str(driver.device)


class TestConfigureOptimizationBlockSwapHandoff:
    """The trainer is the only thing that can see ``_block_swap_managers``
    (it lives on the pipeline/trainer, set by the generic
    ``_configure_block_swapping()`` mixin BEFORE ``_configure_optimization``
    runs) — it must hand the ACTIVE expert's name to the driver so placement
    can skip it."""

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
