"""An expert's timesteps keep the configured distribution's own shape inside its band.

Both dual-expert samplers (WAN 2.2's ``ExpertRouter``, Bernini-R's ``_sample_band``) redraw
into the band through ``TimestepSampler.draw_in_band``. Each "exact" value below comes from
the distribution's formula in this file, never from the code under test.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from types import SimpleNamespace

import pytest
import torch

from app.engine.models.families.bernini_r.trainer import BerniniRTrainer
from app.engine.models.families.wan22.expert_router import HIGH, LOW, ExpertRouter
from app.engine.strategies.timestep_sampling import TimestepSampler

CPU = torch.device("cpu")
BOUNDARY = 875.0
EDGE = {HIGH: 875.0, LOW: 874.0}
EDGE_DRAWS = 2000
# The shape's percentiles: 2000 draws put the 90th percentile's standard error at about 1.2
# timesteps, too close to the bound of 2; 40000 put it at about 0.27.
SHAPE_BATCH = 2000
SHAPE_ROUNDS = 20
SHAPE_TOLERANCE = 2.0
# ``timestep_shift`` 5: the warp 5t / (1 + 4t) adds ln 5 to the logit of t.
BERNINI_SHIFT = 5.0
NORMAL = NormalDist()
# What a mode needs beyond its name: ``fixed_sigmas`` refuses to draw without a list. These levels
# put mass in both bands (with or without Bernini's warp) and none on either edge.
_MODE_CONFIG = {"fixed_sigmas": {"fixed_sigmas": [0.95, 0.9, 0.5, 0.25]}}


class _Log:
    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict]] = []

    def warning(self, event: str, **kw) -> None:
        self.warnings.append((event, kw))

    def info(self, event: str, **kw) -> None:
        pass


@pytest.fixture(autouse=True)
def _restore_the_cpu_stream():
    with torch.random.fork_rng(devices=[]):
        yield


def _wan(cfg: dict) -> ExpertRouter:
    router = ExpertRouter(BOUNDARY / 1000.0, 1, cfg, pinned_expert=HIGH)
    router.logger = _Log()
    return router


def _bernini(cfg: dict) -> BerniniRTrainer:
    trainer = object.__new__(BerniniRTrainer)
    trainer.device = CPU
    trainer.config = cfg
    trainer.driver = SimpleNamespace(boundary_timestep=BOUNDARY)
    trainer.logger = _Log()
    return trainer


def _sampler(family: str, cfg: dict):
    """``(draw(expert, n, progress), log)`` of the family's real band sampler."""
    if family == "wan22":
        router = _wan(cfg)
        return (
            lambda expert, n, progress=0.0: router.sample_timesteps_for(
                expert, n, CPU, cfg, progress=progress
            ),
            router.logger,
        )
    trainer = _bernini(cfg)
    return (
        lambda expert, n, progress=0.0: trainer._sample_band(expert, n, progress=progress),
        trainer.logger,
    )


# ── The distributions' formulas ──────────────────────────────────────────


def _radc_curve(center: float, width: float) -> list[tuple[int, float]]:
    """``(i, weight)`` of the curve's 1000 bins; bin ``i`` is ``t = i / 1000``."""
    sigma = max(width * 500.0, 1.0)
    return [(i, math.exp(-0.5 * ((i - center * 1000.0) / sigma) ** 2)) for i in range(1, 1001)]


def _radc_edge_share(center: float, width: float, expert: str) -> float:
    """What the truncated curve itself puts on the band's edge bin."""
    curve = _radc_curve(center, width)
    band = [w for i, w in curve if (i >= BOUNDARY if expert == HIGH else i < BOUNDARY)]
    edge = next(w for i, w in curve if i == EDGE[expert])
    return edge / sum(band)


def _discrete_stats(values_weights: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Mean, 10th and 90th percentile of a discrete distribution sorted by value."""
    total = sum(w for _, w in values_weights)
    mean = sum(v * w for v, w in values_weights) / total
    cum = 0.0
    p10 = p90 = None
    for v, w in values_weights:
        cum += w / total
        if p10 is None and cum >= 0.1:
            p10 = v
        if p90 is None and cum >= 0.9:
            p90 = v
    return mean, p10, p90


def _truncated_logistic_normal(mu: float, lo: float) -> tuple[float, float, float]:
    """Mean, p10, p90 (x1000) of ``sigmoid(N(mu, 1))`` truncated to ``t >= lo``.

    From the normal CDF: a slot's quantile ``q`` in the band is the normal's quantile
    ``Phi(a) + q * (1 - Phi(a))``. The mean is a midpoint rule over those quantiles.
    """
    a = math.log(lo / (1.0 - lo)) - mu
    base = NORMAL.cdf(a)

    def at(q: float) -> float:
        z = NORMAL.inv_cdf(base + q * (1.0 - base))
        return 1000.0 / (1.0 + math.exp(-(z + mu)))

    steps = 20_000
    mean = sum(at((k + 0.5) / steps) for k in range(steps)) / steps
    return mean, at(0.1), at(0.9)


def _assert_shape(draws: torch.Tensor, exact: tuple[float, float, float]) -> None:
    mean, p10, p90 = exact
    got = torch.quantile(draws, torch.tensor([0.1, 0.9]))
    assert abs(draws.mean().item() - mean) <= SHAPE_TOLERANCE, (draws.mean().item(), mean)
    assert abs(got[0].item() - p10) <= SHAPE_TOLERANCE, (got[0].item(), p10)
    assert abs(got[1].item() - p90) <= SHAPE_TOLERANCE, (got[1].item(), p90)


# ── draw_in_band itself ──────────────────────────────────────────────────


class _Stub:
    """A draw counting its candidates; ``value(i)`` gives the ``i``-th candidate ever drawn."""

    def __init__(self, value=lambda i: 0.0) -> None:
        self.sizes: list[int] = []
        self.value = value
        self.drawn = 0

    def __call__(self, k: int) -> torch.Tensor:
        self.sizes.append(k)
        out = torch.tensor([self.value(self.drawn + j) for j in range(k)], dtype=torch.float32)
        self.drawn += k
        return out


def _never(k: int, sizes: list[int]) -> torch.Tensor:
    sizes.append(k)
    return torch.zeros(k)


def test_a_draw_that_fills_at_once_is_one_draw_of_what_is_missing():
    stub = _Stub()
    out, filled = TimestepSampler.draw_in_band(stub, 4, CPU, lambda t: t == 0.0)
    assert stub.sizes == [4]
    assert filled.tolist() == [True] * 4
    assert out.tolist() == [0.0] * 4


@pytest.mark.parametrize("mode", TimestepSampler.SUPPORTED_MODES)
def test_a_draw_that_fills_at_once_consumes_the_stream_as_one_plain_draw(mode):
    cfg = {"timestep_sampling": mode, **_MODE_CONFIG.get(mode, {})}

    def draw(k: int) -> torch.Tensor:
        return TimestepSampler.sample_scaled(mode, k, CPU, cfg, progress=0.5)

    torch.manual_seed(5)
    plain = draw(8)
    after_plain = torch.rand(4)
    torch.manual_seed(5)
    out, filled = TimestepSampler.draw_in_band(draw, 8, CPU, lambda t: t >= 0.0)
    assert bool(filled.all())
    assert torch.equal(out, plain)
    assert torch.equal(torch.rand(4), after_plain)


@pytest.mark.parametrize("family", ["wan22", "bernini_r"])
def test_a_band_holding_the_whole_distribution_changes_no_stream(family):
    """Through the real samplers: a boundary of 0 fills every high draw at once."""
    cfg = {"timestep_sampling": "uniform", "timestep_shift": 1.0}
    if family == "wan22":
        router = ExpertRouter(0.0, 1, cfg, pinned_expert=HIGH)
        band = lambda: router.sample_timesteps_for(HIGH, 8, CPU, cfg)  # noqa: E731
    else:
        trainer = _bernini(cfg)
        trainer.driver = SimpleNamespace(boundary_timestep=0.0)
        band = lambda: trainer._sample_band(HIGH, 8, progress=0.0)  # noqa: E731
    torch.manual_seed(9)
    plain = TimestepSampler.sample_scaled("uniform", 8, CPU, cfg)
    after_plain = torch.rand(4)
    torch.manual_seed(9)
    assert torch.equal(band(), plain)
    assert torch.equal(torch.rand(4), after_plain)


def test_slots_fill_in_order_with_the_first_accepted_and_surplus_is_dropped():
    # Every third candidate is in the band; its value says which candidate it was.
    stub = _Stub(value=lambda i: float(i) if i % 3 == 0 else -1.0)
    out, filled = TimestepSampler.draw_in_band(stub, 4, CPU, lambda t: t >= 0.0)
    # Round 1 draws 4 (candidates 0..3): accepts 0, 3. Round 2 draws 8 (4..11): accepts
    # 6, 9, and drops nothing yet. Filled at 4 accepted.
    assert stub.sizes == [4, 8]
    assert out.tolist() == [0.0, 3.0, 6.0, 9.0]
    assert filled.all()

    # A later round finding more than is missing keeps only the first ones.
    stub = _Stub(value=lambda i: float(i) if i >= 2 else -1.0)
    out, filled = TimestepSampler.draw_in_band(stub, 4, CPU, lambda t: t >= 0.0)
    assert stub.sizes == [4, 8]
    assert out.tolist() == [2.0, 3.0, 4.0, 5.0]
    assert filled.all()


def test_a_band_found_part_way_reports_the_rest_as_unfilled():
    stub = _Stub(value=lambda i: 7.0 if i == 1 else 0.0)
    out, filled = TimestepSampler.draw_in_band(stub, 3, CPU, lambda t: t > 0.0)
    assert filled.tolist() == [True, False, False]
    assert out[0].item() == 7.0
    assert sum(stub.sizes) == 1_000_000


def _expected_sizes(n: int) -> list[int]:
    """The contract in numbers: ``n``, then doubling to 65536 a round, 1,000,000 in all."""
    sizes = [n]
    while sum(sizes) < 1_000_000:
        sizes.append(min(sizes[-1] * 2, 65536, 1_000_000 - sum(sizes)))
    return sizes


@pytest.mark.parametrize("n", [1, 4, 7, 2000])
def test_no_mass_doubles_up_to_the_round_limit_and_stops_at_the_candidate_limit(n):
    sizes: list[int] = []
    _, filled = TimestepSampler.draw_in_band(
        lambda k: _never(k, sizes), n, CPU, lambda t: t > 0.0
    )
    assert not bool(filled.any())
    assert sizes == _expected_sizes(n)
    assert sum(sizes) == 1_000_000
    assert max(sizes) <= 65536


def test_the_batch_of_four_spends_its_candidates_exactly_like_this():
    sizes: list[int] = []
    TimestepSampler.draw_in_band(lambda k: _never(k, sizes), 4, CPU, lambda t: t > 0.0)
    assert sizes == [4 * 2**i for i in range(15)] + [65536] * 13 + [16964]
    assert len(sizes) == 29


def test_a_batch_above_the_round_limit_still_starts_with_all_of_it():
    sizes: list[int] = []
    TimestepSampler.draw_in_band(lambda k: _never(k, sizes), 70_000, CPU, lambda t: t > 0.0)
    assert sizes[0] == 70_000
    assert all(s <= 65536 for s in sizes[1:])
    assert sum(sizes) == 1_000_000


def test_an_empty_batch_draws_nothing():
    stub = _Stub()
    out, filled = TimestepSampler.draw_in_band(stub, 0, CPU, lambda t: t > 0.0)
    assert stub.sizes == []
    assert out.shape == (0,) and filled.shape == (0,)


# ── Through the real samplers: nothing piles up on the edge ──────────────

_CASES = [(m, 0.0) for m in TimestepSampler.SUPPORTED_MODES if m != "radc"] + [
    ("radc", 0.0),
    ("radc", 0.5),
    ("radc", 1.0),
]


def _edge_limit(family: str, mode: str, progress: float, expert: str) -> float:
    """1.5%, except where WAN's ``radc`` grid itself puts more than that near the edge.

    On WAN's grid the bin at the edge is a real value: its own share is about 0.9% at
    progress 0, 1.2% at 0.5, and 1.5% at 1.0 on the high expert, so 2000 draws scatter around
    it by about 0.27 points: the formula's share and four standard errors bound it there.
    """
    if family == "wan22" and mode == "radc":
        share = _radc_edge_share(0.8 + (0.2 - 0.8) * progress, 0.5, expert)
        return max(0.015, share + 4.0 * math.sqrt(share * (1.0 - share) / EDGE_DRAWS))
    return 0.015


@pytest.mark.parametrize("family", ["wan22", "bernini_r"])
@pytest.mark.parametrize("expert", [HIGH, LOW])
@pytest.mark.parametrize("mode,progress", _CASES)
def test_draws_stay_in_the_band_and_do_not_pile_up_on_its_edge(family, expert, mode, progress):
    sample, log = _sampler(family, {"timestep_sampling": mode, **_MODE_CONFIG.get(mode, {})})
    torch.manual_seed(1)
    t = sample(expert, EDGE_DRAWS, progress=progress)
    if expert == HIGH:
        assert float(t.min()) >= BOUNDARY and float(t.max()) <= 1000.0
    else:
        assert float(t.max()) < BOUNDARY and float(t.min()) >= 0.0
    share = (t == EDGE[expert]).float().mean().item()
    assert share <= _edge_limit(family, mode, progress, expert), share
    assert log.warnings == []


def test_the_shift_warp_of_a_thin_low_band_on_bernini_no_longer_clamps():
    """``model_shift`` at WAN's fixed 12 with the warp: 0.6% of low draws were on 874."""
    sample, log = _sampler("bernini_r", {"timestep_sampling": "model_shift", "model_shift_fixed": 12.0})
    torch.manual_seed(1)
    t = sample(LOW, EDGE_DRAWS)
    assert (t == EDGE[LOW]).float().mean().item() == 0.0
    assert log.warnings == []


# ── Through the real samplers: the band's shape ──────────────────────────


def _many(sample, expert: str, progress: float = 0.0) -> torch.Tensor:
    return torch.cat([sample(expert, SHAPE_BATCH, progress=progress) for _ in range(SHAPE_ROUNDS)])


def test_wan_logit_normal_is_the_truncated_distribution():
    sample, log = _sampler("wan22", {"timestep_sampling": "logit_normal"})
    torch.manual_seed(21)
    _assert_shape(_many(sample, HIGH), _truncated_logistic_normal(0.0, BOUNDARY / 1000.0))
    assert log.warnings == []


def test_bernini_logit_normal_is_the_truncated_warped_distribution():
    sample, log = _sampler("bernini_r", {"timestep_sampling": "logit_normal"})
    torch.manual_seed(22)
    _assert_shape(
        _many(sample, HIGH),
        _truncated_logistic_normal(math.log(BERNINI_SHIFT), BOUNDARY / 1000.0),
    )
    assert log.warnings == []


def test_wan_radc_at_the_last_step_is_the_truncated_curve():
    sample, log = _sampler("wan22", {"timestep_sampling": "radc"})
    torch.manual_seed(23)
    curve = [(float(i), w) for i, w in _radc_curve(0.2, 0.5) if i >= BOUNDARY]
    _assert_shape(_many(sample, HIGH, progress=1.0), _discrete_stats(curve))
    assert log.warnings == []


def test_bernini_radc_at_the_last_step_is_the_truncated_warped_curve():
    sample, log = _sampler("bernini_r", {"timestep_sampling": "radc"})
    torch.manual_seed(24)
    warped = [
        (1000.0 * BERNINI_SHIFT * (i / 1000.0) / (1.0 + (BERNINI_SHIFT - 1.0) * i / 1000.0), w)
        for i, w in _radc_curve(0.2, 0.5)
    ]
    curve = [(v, w) for v, w in warped if v >= BOUNDARY]
    _assert_shape(_many(sample, HIGH, progress=1.0), _discrete_stats(curve))
    assert log.warnings == []


# ── A band with no mass ends on the edge, warns, doesn't hang ────────────


@pytest.mark.parametrize("family", ["wan22", "bernini_r"])
def test_radc_tail_with_no_representable_mass_ends_on_the_edge(family):
    """``radc_width`` 0.1 at the last step: the float32 CDF is flat long before 875."""
    cfg = {"timestep_sampling": "radc", "radc_width": 0.1}
    sample, log = _sampler(family, cfg)
    torch.manual_seed(31)
    t = sample(HIGH, 4, progress=1.0)
    assert t.tolist() == [BOUNDARY] * 4
    assert len(log.warnings) == 1
    assert log.warnings[0][1]["unfilled"] == 4


@pytest.mark.parametrize("family", ["wan22", "bernini_r"])
@pytest.mark.parametrize("expert", [HIGH, LOW])
def test_a_draw_that_never_lands_in_the_band_stops_at_the_limit(family, expert, monkeypatch):
    sizes: list[int] = []
    outside = 0.0 if expert == HIGH else 1000.0

    def never(k: int) -> torch.Tensor:
        sizes.append(k)
        return torch.full((k,), outside)

    cfg = {"timestep_sampling": "uniform"}
    if family == "wan22":
        router = _wan(cfg)
        monkeypatch.setattr(TimestepSampler, "sample_scaled", staticmethod(lambda _m, k, *a, **kw: never(k)))
        t = router.sample_timesteps_for(expert, 4, CPU, cfg)
        log = router.logger
    else:
        trainer = _bernini(cfg)
        monkeypatch.setattr(trainer, "_warped_draw", lambda k, *a, **kw: never(k), raising=False)
        t = trainer._sample_band(expert, 4, progress=0.0)
        log = trainer.logger
    assert sum(sizes) == 1_000_000
    assert t.tolist() == [EDGE[expert]] * 4
    assert len(log.warnings) == 1
    assert log.warnings[0][1]["unfilled"] == 4


@pytest.mark.parametrize("family", ["wan22", "bernini_r"])
def test_only_the_unfilled_slots_go_to_the_edge(family, monkeypatch):
    drawn = {"n": 0}

    def rare(k: int) -> torch.Tensor:
        out = torch.zeros(k)
        if drawn["n"] == 0:
            out[1] = 950.0
        drawn["n"] += k
        return out

    cfg = {"timestep_sampling": "uniform"}
    if family == "wan22":
        router = _wan(cfg)
        monkeypatch.setattr(TimestepSampler, "sample_scaled", staticmethod(lambda _m, k, *a, **kw: rare(k)))
        t = router.sample_timesteps_for(HIGH, 3, CPU, cfg)
        log = router.logger
    else:
        trainer = _bernini(cfg)
        monkeypatch.setattr(trainer, "_warped_draw", lambda k, *a, **kw: rare(k), raising=False)
        t = trainer._sample_band(HIGH, 3, progress=0.0)
        log = trainer.logger
    assert t.tolist() == [950.0, BOUNDARY, BOUNDARY]
    assert log.warnings[0][1]["unfilled"] == 2


# ── The router's choices don't read these draws ──────────────────────────


@pytest.mark.parametrize("family", ["wan22", "bernini_r"])
@pytest.mark.parametrize("mode", ["logit_normal", "radc"])
def test_p_high_and_the_seeds_decisions_are_unchanged(family, mode):
    cfg = {"timestep_sampling": mode}
    seed, mc = 3, 20_000
    trainer = _bernini(cfg)
    draw = None
    if family == "bernini_r":
        draw = lambda n, progress=0.0, latents=None: trainer._warped_draw(  # noqa: E731
            n, CPU, latents=latents, progress=progress
        )
    router = ExpertRouter(BOUNDARY / 1000.0, 1, cfg, seed=seed, mc_samples=mc, timestep_draw=draw)

    # What a router computes, written out: the estimate on its own seed, the Bernoulli on another.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed + 1)
        if draw is None:
            samples = TimestepSampler.sample_scaled(mode, mc, CPU, cfg)
        else:
            samples = draw(mc)
    p_high = (samples >= BOUNDARY).float().mean().item()
    assert router.p_high == min(max(p_high, 1e-6), 1.0 - 1e-6)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    expected = [HIGH if torch.rand((), generator=gen).item() < router.p_high else LOW for _ in range(32)]

    # Band draws between the decisions, of every length, move neither.
    got = []
    for step in range(32):
        got.append(router.choose_expert(step))
        if family == "wan22":
            router.sample_timesteps_for(got[-1], 4, CPU, cfg, progress=step / 32)
        else:
            trainer._sample_band(got[-1], 4, progress=step / 32)
    assert got == expected
