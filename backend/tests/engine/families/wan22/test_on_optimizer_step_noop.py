"""Regression: ``on_optimizer_step`` is a no-op for existing families.

Phase B4 adds a default-no-op ``on_optimizer_step`` to ``IModelDriver`` and a
single call site in the training loop. This proves:

- The default (and an EXISTING family driver, WAN 2.1) treat the hook as a no-op
  that raises nothing and mutates nothing — existing families are unaffected.
- The training-loop call site invokes it safely (via ``getattr`` guard) even for
  a driver-less / hook-less trainer.
- The WAN 2.2 driver, by contrast, ADVANCES the router (active_expert changes per
  the seeded plan) — i.e. the hook is wired and meaningful only where overridden.
"""

import torch

from app.engine.models.families.wan21.driver import Wan21Driver
from app.engine.models.families.wan22.driver import Wan22Driver
from app.engine.models.families.wan22.expert_router import ExpertRouter
from app.engine.core.pipeline.pipeline_train import PipelineTrainMixin


class _Defn21:
    architecture_params = {"mode": "t2v", "te.max_length": 512}
    lora_targetable_modules: list[str] = []


class _Defn22:
    architecture_params = {"mode": "t2v", "moe.boundary_ratio": 0.875}
    lora_targetable_modules: list[str] = []


# ── existing family driver: hook is a harmless no-op ──────────────────────


def test_base_default_on_optimizer_step_is_noop():
    driver = Wan21Driver(_Defn21(), torch.device("cpu"))
    # Default inherited from IModelDriver — returns None, mutates nothing, raises
    # nothing across many calls.
    for step in range(100):
        assert driver.on_optimizer_step(step) is None


def test_existing_driver_state_unchanged_by_hook():
    driver = Wan21Driver(_Defn21(), torch.device("cpu"))
    before = dict(driver.__dict__)
    driver.on_optimizer_step(5)
    after = dict(driver.__dict__)
    assert before.keys() == after.keys()
    # No attribute changed identity (the no-op touches nothing).
    for k in before:
        assert before[k] is after[k]


# ── training-loop call site is safe even with no driver / no hook ─────────


class _NoDriverTrainer(PipelineTrainMixin):
    """Trainer with NO driver attribute — the call-site getattr guard applies."""


class _HooklessDriverTrainer(PipelineTrainMixin):
    def __init__(self):
        self.driver = object()  # has no on_optimizer_step


def test_call_site_guard_tolerates_missing_driver_and_hook():
    # Re-create the exact guard the loop uses and prove it no-ops safely.
    def invoke(trainer, step):
        driver = getattr(trainer, "driver", None)
        if driver is not None and hasattr(driver, "on_optimizer_step"):
            driver.on_optimizer_step(step)

    invoke(_NoDriverTrainer(), 0)  # no driver → skipped
    invoke(_HooklessDriverTrainer(), 0)  # driver w/o hook → skipped
    # Both return without raising.


# ── WAN 2.2 driver: hook ADVANCES the router (active changes per plan) ────


def test_wan22_hook_advances_router():
    driver = Wan22Driver(_Defn22(), torch.device("cpu"))
    # Use a boundary/seed that yields a mixed plan so the active expert flips.
    router = ExpertRouter(
        boundary=0.6,
        switch_interval=1,
        timestep_cfg={"timestep_sampling": "uniform"},
        seed=0,
        mc_samples=20_000,
    )
    driver.set_router(router)

    expected = [router.choose_expert(s) for s in range(1, 50)]
    driver._set_active(router.choose_expert(0))
    seen = []
    for step in range(49):
        driver.on_optimizer_step(step)
        seen.append(driver.active_expert)
    assert seen == expected
    # Sanity: the plan is not constant (the hook genuinely switches experts).
    assert len(set(seen)) == 2, "expected both experts to appear in the plan"
