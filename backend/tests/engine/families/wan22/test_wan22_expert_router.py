"""WAN 2.2 ExpertRouter tests — THE point of phase B4.

These pin the routing math that makes single-run dual-expert training UNBIASED:

1. ``p_high`` ≈ the analytic mass above the boundary under the configured
   timestep distribution (Monte-Carlo tolerance).
2. Per-expert truncated samples are STRICTLY in the expert's range.
3. Over many steps the high/low frequency ≈ ``p_high`` even with
   ``switch_interval > 1`` (hysteresis preserves long-run frequency).
4. Per-optimizer-step coherence: the same expert is chosen for every micro-batch
   query within one step (and across a switch_interval block).
5. Deterministic under a fixed seed.
6. The MARGINAL timestep distribution (mixing per-expert truncated samples by
   p_high) matches the untruncated distribution — i.e. the routing is unbiased.
"""

import math

import pytest
import torch

from app.engine.models.families.wan22.expert_router import HIGH, LOW, ExpertRouter
from app.engine.strategies.timestep_sampling import TimestepSampler

CPU = torch.device("cpu")


# ── 1. p_high ≈ analytic mass above boundary ──────────────────────────────


def test_phigh_uniform_matches_analytic_mass():
    # Uniform U(0,1)→[0,1000]; P(t >= 800) = 0.2 analytically.
    router = ExpertRouter(
        boundary=0.8,
        switch_interval=1,
        timestep_cfg={"timestep_sampling": "uniform"},
        seed=0,
        mc_samples=200_000,
    )
    assert abs(router.p_high - 0.2) < 0.01, router.p_high


def test_phigh_logit_normal_matches_analytic_mass():
    # logit_normal mu=0,sigma=1: t = sigmoid(N(0,1)); P(t >= boundary) =
    # P(N <= -logit(boundary))... compute the analytic mass and compare.
    boundary = 0.875
    cfg = {"timestep_sampling": "logit_normal"}
    router = ExpertRouter(
        boundary=boundary,
        switch_interval=1,
        timestep_cfg=cfg,
        seed=1,
        mc_samples=400_000,
    )
    # P(sigmoid(Z) >= b) = P(Z >= logit(b)) = 1 - Phi(logit(b)).
    logit_b = math.log(boundary / (1.0 - boundary))
    phi = 0.5 * (1.0 + math.erf(logit_b / math.sqrt(2.0)))
    analytic = 1.0 - phi
    assert abs(router.p_high - analytic) < 0.01, (router.p_high, analytic)


# ── 2. truncated samples strictly in expert range ─────────────────────────


def test_truncated_samples_in_high_range():
    router = ExpertRouter(
        boundary=0.8,
        switch_interval=1,
        timestep_cfg={"timestep_sampling": "uniform"},
        seed=2,
        mc_samples=10_000,
    )
    cfg = {"timestep_sampling": "uniform"}
    hi = router.sample_timesteps_for(HIGH, 4096, CPU, cfg)
    assert bool((hi >= router.boundary_scaled).all()), hi.min().item()
    lo = router.sample_timesteps_for(LOW, 4096, CPU, cfg)
    assert bool((lo < router.boundary_scaled).all()), lo.max().item()
    # And in [0, 1000].
    assert bool((hi <= 1000.0).all()) and bool((lo >= 0.0).all())


# ── 3. long-run frequency ≈ p_high (with hysteresis) ──────────────────────


def test_step_frequency_matches_phigh_switch_interval_1():
    router = ExpertRouter(
        boundary=0.7,
        switch_interval=1,
        timestep_cfg={"timestep_sampling": "uniform"},
        seed=3,
        mc_samples=200_000,
    )
    n = 20_000
    highs = sum(1 for s in range(n) if router.choose_expert(s) == HIGH)
    freq = highs / n
    # p_high ≈ 0.3; binomial std over 20k ≈ 0.0032 → 4σ ≈ 0.013.
    assert abs(freq - router.p_high) < 0.02, (freq, router.p_high)


def test_step_frequency_matches_phigh_switch_interval_8():
    router = ExpertRouter(
        boundary=0.7,
        switch_interval=8,
        timestep_cfg={"timestep_sampling": "uniform"},
        seed=4,
        mc_samples=200_000,
    )
    n = 40_000  # 5000 blocks of 8 → frequency still ≈ p_high.
    highs = sum(1 for s in range(n) if router.choose_expert(s) == HIGH)
    freq = highs / n
    assert abs(freq - router.p_high) < 0.025, (freq, router.p_high)


# ── 4. per-step / per-block coherence ─────────────────────────────────────


def test_per_block_coherence_switch_interval():
    router = ExpertRouter(
        boundary=0.6,
        switch_interval=5,
        timestep_cfg={"timestep_sampling": "uniform"},
        seed=5,
        mc_samples=50_000,
    )
    # All steps inside one switch_interval block share one decision.
    for block in range(50):
        decisions = {router.choose_expert(block * 5 + offset) for offset in range(5)}
        assert len(decisions) == 1, (block, decisions)


def test_same_step_queried_twice_is_coherent():
    router = ExpertRouter(
        boundary=0.6,
        switch_interval=1,
        timestep_cfg={"timestep_sampling": "uniform"},
        seed=6,
        mc_samples=50_000,
    )
    for s in range(200):
        assert router.choose_expert(s) == router.choose_expert(s)


# ── 5. determinism under fixed seed ───────────────────────────────────────


def test_deterministic_under_seed():
    def plan(seed):
        r = ExpertRouter(
            boundary=0.65,
            switch_interval=3,
            timestep_cfg={"timestep_sampling": "uniform"},
            seed=seed,
            mc_samples=50_000,
        )
        return [r.choose_expert(s) for s in range(500)]

    assert plan(42) == plan(42)
    # Different seed → (very likely) different plan.
    assert plan(42) != plan(99)


def test_state_roundtrip_resumes_plan():
    r1 = ExpertRouter(
        boundary=0.65,
        switch_interval=2,
        timestep_cfg={"timestep_sampling": "uniform"},
        seed=7,
        mc_samples=50_000,
    )
    # Advance a few steps, snapshot, then continue.
    [r1.choose_expert(s) for s in range(10)]
    state = r1.state_dict()
    continued = [r1.choose_expert(s) for s in range(10, 30)]

    r2 = ExpertRouter(
        boundary=0.65,
        switch_interval=2,
        timestep_cfg={"timestep_sampling": "uniform"},
        seed=999,
        mc_samples=50_000,
    )
    r2.load_state_dict(state)
    resumed = [r2.choose_expert(s) for s in range(10, 30)]
    assert resumed == continued


# ── 6. unbiased marginal: mixing per-expert truncated by p_high == full ───


def test_marginal_timestep_distribution_is_unbiased():
    """Mixing the two truncated distributions by p_high recovers the full one.

    We compare CDFs of (a) untruncated samples and (b) a p_high-weighted mix of
    high/low truncated samples. They must coincide (law of total probability).
    """
    boundary = 0.7
    cfg = {"timestep_sampling": "logit_normal"}
    router = ExpertRouter(
        boundary=boundary,
        switch_interval=1,
        timestep_cfg=cfg,
        seed=8,
        mc_samples=400_000,
    )
    p = router.p_high
    n = 200_000

    torch.manual_seed(123)
    from app.engine.strategies.timestep_sampling import TimestepSampler

    full = TimestepSampler.sample_scaled("logit_normal", n, CPU, cfg, scale=1000.0)

    n_high = int(round(p * n))
    n_low = n - n_high
    hi = router.sample_timesteps_for(HIGH, n_high, CPU, cfg)
    lo = router.sample_timesteps_for(LOW, n_low, CPU, cfg)
    mix = torch.cat([hi, lo])

    # Compare empirical CDFs at several quantile probe points.
    probes = torch.linspace(50.0, 950.0, 19)
    for t in probes:
        cdf_full = float((full <= t).float().mean())
        cdf_mix = float((mix <= t).float().mean())
        assert abs(cdf_full - cdf_mix) < 0.02, (t.item(), cdf_full, cdf_mix)


# ── 7. optional `timestep_draw` override (Task W3.T3) is a no-op unless ──
#      supplied — wan22 never passes it, so its p_high estimate must stay
#      byte-identical to the pre-T3 TimestepSampler-only computation.


def test_phigh_byte_identical_without_timestep_draw():
    """wan22's ExpertRouter never passes `timestep_draw` — adding the optional
    parameter (for bernini_r's real per-step distribution, see
    BerniniRTrainer._build_router) must not perturb wan22's own estimate by
    even a float ULP. Reproduces the internal RNG-seeding convention
    independently and compares exactly."""
    cfg = {"timestep_sampling": "logit_normal"}
    seed = 55
    router = ExpertRouter(
        boundary=0.875,
        switch_interval=1,
        timestep_cfg=cfg,
        seed=seed,
        mc_samples=80_000,
    )

    prev_state = torch.random.get_rng_state()
    try:
        torch.random.manual_seed(seed + 1)
        ref = TimestepSampler.sample_scaled(
            "logit_normal",
            80_000,
            CPU,
            cfg,
            scale=1000.0,
        )
    finally:
        torch.random.set_rng_state(prev_state)
    ref_p = min(max(float((ref >= 875.0).float().mean().item()), 1e-6), 1.0 - 1e-6)

    assert router.p_high == ref_p


def test_timestep_draw_overrides_the_estimate_when_supplied():
    """When `timestep_draw` IS supplied, the estimate must come from it, not
    the generic TimestepSampler — a directly-observable contract check
    independent of any specific family's formula."""
    calls: list[int] = []

    def _all_high(n: int) -> torch.Tensor:
        calls.append(n)
        return torch.full((n,), 999.0)

    router = ExpertRouter(
        boundary=0.875,
        switch_interval=1,
        timestep_cfg={"timestep_sampling": "uniform"},  # would give p_high≈0.125
        seed=0,
        mc_samples=1000,
        timestep_draw=_all_high,
    )
    assert calls == [1000]
    # All draws are >= boundary → p_high clamps to the "never 1.0" guard.
    assert router.p_high == pytest.approx(1.0 - 1e-6)
