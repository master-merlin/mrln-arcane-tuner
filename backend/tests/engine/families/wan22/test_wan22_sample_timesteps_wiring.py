"""WAN 2.2 dual-expert — REAL-path ``sample_timesteps`` wiring (recon-found bug).

SYMPTOM (recon, dead-dispatch class): ``Wan22Driver.sample_timesteps``
(``driver.py:132-148``) routes to ``ExpertRouter.sample_timesteps_for(active_expert,
...)``, which draws timesteps **truncated to the active expert's boundary range**
(high expert → ``t >= boundary``; low expert → ``t < boundary``). This truncation
is the whole point of the WAN 2.2 Mixture-of-Experts split: the high-noise expert
must only ever see high timesteps and the low-noise expert only low timesteps
(``expert_router.py`` docstring — the per-expert conditioning keeps the run's
*marginal* timestep distribution unbiased while each expert specializes).

ROOT CAUSE: the REAL training loop (``pipeline_train.py:531``) calls the TRAINER's
``self.sample_timesteps(batch, latents)`` family hook (MRO-resolved), NOT
``driver.sample_timesteps``. ``Wan22Trainer`` did NOT override ``sample_timesteps``,
so the call resolved to ``PipelineBaseMixin.sample_timesteps``
(``pipeline_base.py:108-127``), the **full-range** flow-match sampler with NO
knowledge of the expert boundary or the router. On a real dual-expert run the
active expert IS switched per step (``driver.on_optimizer_step`` — that hook IS
wired), but every expert was fed timesteps from the FULL ``[0, 1000]`` range,
directly contradicting the MoE design: the high expert was trained on low-noise
timesteps and vice-versa. The driver's router truncation was dead code on the
real path (the same dead-dispatch class as k5 I2V frame-0 / boogu scheduler).

Every existing ``sample_timesteps`` test calls ``driver.sample_timesteps`` (or the
router) directly — never through ``trainer.sample_timesteps`` — so the MRO gap was
invisible to unit tests (the historical boogu/k5 bug shape).

THE FIX, ORIGINALLY (``trainer.py``): ``Wan22Trainer.sample_timesteps`` overrode
the ``PipelineBaseMixin`` hook to delegate to
``self.driver.sample_timesteps(batch, self.device, self.config,
latents=latents)`` — mirroring ``boogu_image``'s convention-delegation
precedent.

THE FIX, NOW (W5.T10): that per-family override was deleted as
auto-delegation-identical — ``PipelineBaseMixin.sample_timesteps``
(``pipeline_base.py:176-231``, ``_driver_hook_override`` /
``core/hook_dispatch.py``) auto-delegates to ``driver.sample_timesteps`` with
the EXACT SAME arguments whenever the driver meaningfully overrides the hook
(``Wan22Driver.sample_timesteps`` does — router-truncated MoE sampling), so
the redundant trainer method was pure duplication. ``Wan22Trainer`` no longer
defines ``sample_timesteps`` at all; the structural mechanism is what wires
this now, not a per-family method to keep in sync.

These tests walk the REAL call ``pipeline_train.py`` makes on every step
(``trainer.sample_timesteps(batch_size, latents)``) with a REAL driver + REAL
``ExpertRouter`` and assert the timesteps land in the active expert's range.
Nothing here calls ``driver.sample_timesteps`` directly.
"""

from __future__ import annotations

import torch

from app.engine.core.pipeline.pipeline_base import PipelineBaseMixin
from app.engine.models.families.wan22.driver import Wan22Driver
from app.engine.models.families.wan22.expert_router import ExpertRouter
from app.engine.models.families.wan22.trainer import Wan22Trainer

BOUNDARY_FRAC = 0.875
BOUNDARY_SCALED = BOUNDARY_FRAC * 1000.0


class _Defn:
    architecture_params = {
        "mode": "t2v",
        "te.max_length": 512,
        "moe.boundary_ratio": BOUNDARY_FRAC,
    }
    lora_targetable_modules: list[str] = []


def _trainer(pinned_expert: str) -> Wan22Trainer:
    """REAL Wan22Trainer + REAL driver + REAL (pinned) ExpertRouter, no model load.

    Mirrors the precision-contract tests' ``_make_driver`` construction and the
    kandinsky5 wiring test's ``object.__new__`` trainer stub. ``set_router`` syncs
    ``driver._active_expert`` to the pinned expert — exactly the state the real
    ``pipeline_train`` loop holds when it calls ``sample_timesteps``.
    """
    t = object.__new__(Wan22Trainer)
    t.device = torch.device("cpu")
    t.config = {"timestep_sampling": "logit_normal"}
    driver = Wan22Driver(_Defn(), torch.device("cpu"))
    router = ExpertRouter(
        boundary=BOUNDARY_FRAC,
        switch_interval=1,
        timestep_cfg=t.config,
        seed=0,
        pinned_expert=pinned_expert,
    )
    driver.set_router(router)
    t.driver = driver
    return t


def test_wan22_trainer_does_not_shadow_sample_timesteps() -> None:
    """W5.T10: the per-family override is GONE — Wan22Trainer must resolve
    ``sample_timesteps`` to PipelineBaseMixin (auto-delegation), not shadow
    it with a redundant same-behavior method. If this ever fails, a future
    edit reintroduced the exact duplication this task removed."""
    assert "sample_timesteps" not in vars(Wan22Trainer)
    assert Wan22Trainer.sample_timesteps is PipelineBaseMixin.sample_timesteps


class TestRealPathExpertTimestepTruncation:
    """Load-bearing regression — FAILS RED without the driver override being
    auto-delegated (``PipelineBaseMixin._driver_hook_override``,
    ``pipeline_base.py:176-231``)."""

    def test_high_expert_timesteps_stay_in_high_range(self) -> None:
        t = _trainer("high")
        assert t.driver.active_expert == "high"  # sanity: router synced
        # THE REAL CALL — exactly what pipeline_train.py:531 makes.
        ts = t.sample_timesteps(128, None)
        assert torch.all(ts >= BOUNDARY_SCALED), (
            "high-noise expert must ONLY see timesteps >= boundary through the "
            "REAL trainer.sample_timesteps() call — got some below boundary, "
            "which means the base full-range sampler ran instead of the driver's "
            "router truncation (dead-dispatch: expert specialization broken)."
        )

    def test_low_expert_timesteps_stay_in_low_range(self) -> None:
        t = _trainer("low")
        assert t.driver.active_expert == "low"
        ts = t.sample_timesteps(128, None)
        assert torch.all(ts < BOUNDARY_SCALED), (
            "low-noise expert must ONLY see timesteps < boundary through the "
            "REAL trainer.sample_timesteps() call."
        )

    def test_trainer_hook_matches_driver_router_range(self) -> None:
        """The trainer hook must reach the SAME expert-aware path as the driver —
        i.e. it is not the base sampler (which would ignore the boundary)."""
        t = _trainer("high")
        # Base (un-overridden) sampler would draw across the full range; with 256
        # logit-normal draws at boundary 0.875 it is astronomically unlikely to
        # land ALL of them >= boundary by chance, so this pins the wiring.
        ts = t.sample_timesteps(256, None)
        assert ts.shape == (256,)
        assert torch.all(ts >= BOUNDARY_SCALED)
