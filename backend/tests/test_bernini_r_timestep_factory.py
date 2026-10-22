"""LANE-168 task 2: bernini_r takes its timesteps from the shared ``TimestepSampler``.

The configured ``timestep_sampling`` mode is the BASE draw (the flux3 pattern); the family applies
only its declared transforms on top: its pinned cosine ``mode`` over a factory ``uniform`` draw, and
the ``timestep_shift`` warp on every mode. The 14B band sampler and the ``ExpertRouter`` p_high
estimate read that one warped distribution (the router at the current progress and one reference
latent shape).

CPU only, no weights: the trainer is a bare ``BerniniRTrainer`` around a real dual driver.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from app.engine.models.families.bernini_r.driver import BerniniRDriver
from app.engine.models.families.bernini_r.trainer import BerniniRTrainer
from app.engine.strategies.timestep_sampling import TimestepSampler

CPU = torch.device("cpu")
PRESETS = {"timestep_sampling": "mode", "mode_scale": 1.29, "timestep_shift": 5.0}


class _Recorder:
    """A logger double that records ``(level, event, fields)`` -- avoids structlog caching traps."""

    def __init__(self):
        self.events: list[tuple[str, str, dict]] = []

    def info(self, event, **kw):
        self.events.append(("info", event, kw))

    def warning(self, event, **kw):
        self.events.append(("warning", event, kw))

    def debug(self, event, **kw):
        self.events.append(("debug", event, kw))


class _DualDefn:
    architecture_params = {
        "mode": "t2v",
        "te.max_length": 512,
        "dual_expert": True,
        "switch_dit_boundary": 0.875,
        "moe.boundary_ratio": 0.875,
    }
    lora_targetable_modules: list[str] = []
    defaults: dict = {"num_frames": 81}


def _trainer(config=None, *, step=30, max_steps=60) -> BerniniRTrainer:
    t = object.__new__(BerniniRTrainer)
    t.device = CPU
    t.config = dict(config or {})
    t.logger = _Recorder()
    t.global_step = step
    t.max_train_steps = max_steps
    return t


def _dual(config, *, step=0, max_steps=100) -> BerniniRTrainer:
    t = _trainer(
        {"expert_switch_interval": 1, "expert_swap_mode": "resident", "expert_mode": "both", "seed": 0, **config},
        step=step,
        max_steps=max_steps,
    )
    t.definition = _DualDefn()
    t.driver = BerniniRDriver(_DualDefn(), CPU)
    t.expert_mode = "both"
    t._build_router()
    return t


def _warp(t01: torch.Tensor, shift: float) -> torch.Tensor:
    return (shift * t01 / (1.0 + (shift - 1.0) * t01) * 1000.0).clamp(0, 1000)


@pytest.fixture
def spy(monkeypatch):
    calls: list[SimpleNamespace] = []
    real = TimestepSampler.sample

    def sample(mode, bs, device, config, *, latents=None, progress=0.0):
        calls.append(SimpleNamespace(mode=mode, bs=bs, device=device, config=config, latents=latents, progress=progress))
        return real(mode, bs, device, config, latents=latents, progress=progress)

    monkeypatch.setattr(TimestepSampler, "sample", staticmethod(sample))
    return calls


# ── scenario 1: the configured mode is the base draw, the shift warp is on top ──


def test_radc_is_the_factory_draw_with_progress_then_the_shift_warp(spy):
    cfg = {"timestep_sampling": "radc", "radc_start": 0.8, "timestep_shift": 5.0}
    t = _trainer(cfg)
    torch.manual_seed(0)
    got = t.sample_timesteps(4)

    assert [(c.mode, c.bs, c.progress) for c in spy] == [("radc", 4, 0.5)]
    assert spy[0].device == CPU and spy[0].config.get("radc_start") == 0.8
    torch.manual_seed(0)
    base = TimestepSampler.sample("radc", 4, CPU, cfg, progress=0.5)
    assert torch.equal(got, _warp(base, 5.0))


# ── scenario 2: the presets stay byte-identical ──


def test_presets_keep_the_cosine_mode_over_a_factory_uniform_draw(spy):
    t = _trainer(PRESETS)
    torch.manual_seed(0)
    got = t.sample_timesteps(4096)

    assert [c.mode for c in spy] == ["uniform"]
    torch.manual_seed(0)
    want = BerniniRTrainer._mode_shift_timesteps(torch.rand(4096), 1.29, 5.0)
    assert torch.equal(got, want)


def test_no_configured_mode_is_the_family_default_mode():
    a, b = _trainer({"mode_scale": 1.29, "timestep_shift": 5.0}), _trainer(PRESETS)
    torch.manual_seed(1)
    x = a.sample_timesteps(64)
    torch.manual_seed(1)
    y = b.sample_timesteps(64)
    assert torch.equal(x, y)


# ── scenario 3: latents + patchify factor reach the resolution-aware modes ──


def _draw(cfg, shape):
    t = _trainer({"flux_shift_patchify_factor": 2, "timestep_shift": 1.0, **cfg})
    torch.manual_seed(0)
    return t.sample_timesteps(512, latents=torch.zeros(*shape))


@pytest.mark.parametrize(
    "cfg",
    [{"timestep_sampling": "flux_shift"}, {"timestep_sampling": "radc", "radc_res_influence": 1.0}],
)
def test_resolution_aware_modes_see_the_batch_latents(cfg):
    small, large = _draw(cfg, (2, 16, 32, 32)), _draw(cfg, (2, 16, 64, 64))
    assert not torch.equal(small, large), f"{cfg}: latents were not forwarded to the factory"


def test_radc_without_res_influence_ignores_the_resolution():
    cfg = {"timestep_sampling": "radc", "radc_res_influence": 0.0}
    assert torch.equal(_draw(cfg, (2, 16, 32, 32)), _draw(cfg, (2, 16, 64, 64)))


# ── once-per-job evidence ──


def test_first_draw_logs_the_mode_once_and_nothing_is_replaced():
    t = _trainer({"timestep_sampling": "radc", "radc_res_influence": 0.0})
    t.sample_timesteps(2)
    t.sample_timesteps(2)
    events = [e for e in t.logger.events if e[1] == "timestep_sampling_mode"]
    assert len(events) == 1
    _, _, kw = events[0]
    assert kw["family"] == "bernini_r" and kw["configured"] == "radc" and kw["effective"] == "radc"
    assert kw["progress_forwarded"] is True and kw["res_influence"] == 0.0
    assert "routing_reference_shape" in kw
    assert not [e for e in t.logger.events if e[0] == "warning"]


# ── 14B: band split and router read ONE warped distribution at the current progress ──


def _band_mass(cfg, progress, shape, n=8192):
    torch.manual_seed(0)
    base = TimestepSampler.sample(
        cfg["timestep_sampling"], n, CPU, cfg, latents=torch.zeros(*shape), progress=progress
    )
    return float((_warp(base, cfg["timestep_shift"]) >= 875.0).float().mean())


RADC14 = {
    "timestep_sampling": "radc",
    "radc_res_influence": 1.0,
    "timestep_shift": 5.0,
    "flux_shift_patchify_factor": 2,
    "num_frames": 1,
}


def _shape(res, frames=1):
    return (1, 16, (frames - 1) // 4 + 1, res // 8, res // 8)


def test_router_p_high_follows_progress():
    # 1024 px: the saturated token count makes the resolution cross-term monotone in progress
    # (at 512 px with these defaults the progress and resolution terms cancel exactly at 0.9).
    t = _dual({**RADC14, "resolutions": [1024]})
    lo, hi = t.expert_router.estimate_p_high(progress=0.0), t.expert_router.estimate_p_high(progress=0.9)
    assert abs(lo - hi) > 0.02, (lo, hi)
    for p, got in ((0.0, lo), (0.9, hi)):
        assert got == pytest.approx(_band_mass(RADC14, p, _shape(1024)), abs=0.02)


def test_router_p_high_follows_the_reference_resolution():
    p512 = _dual({**RADC14, "resolutions": [512]}).expert_router.estimate_p_high(progress=0.5)
    p1024 = _dual({**RADC14, "resolutions": [1024]}).expert_router.estimate_p_high(progress=0.5)
    assert abs(p512 - p1024) > 0.02, (p512, p1024)
    assert p512 == pytest.approx(_band_mass(RADC14, 0.5, _shape(512)), abs=0.02)
    assert p1024 == pytest.approx(_band_mass(RADC14, 0.5, _shape(1024)), abs=0.02)


def test_video_shaped_reference_saturates_the_token_count():
    cfg = {**RADC14, "num_frames": 81}
    a = _dual({**cfg, "resolutions": [512]}).expert_router.estimate_p_high(progress=0.5)
    b = _dual({**cfg, "resolutions": [1024]}).expert_router.estimate_p_high(progress=0.5)
    assert a == b  # both shapes are past the 4096-token saturation


def test_reference_shape_comes_from_resolution_frames_and_vae_factors():
    t = _dual({**RADC14, "resolutions": [512], "num_frames": 81})
    assert tuple(t._routing_reference_shape()) == (1, 16, 21, 64, 64)
    t2 = _dual({"resolutions": [1024]})  # frames from the definition default (81)
    assert tuple(t2._routing_reference_shape()) == (1, 16, 21, 128, 128)


def test_mode_preset_probability_is_the_static_value_at_every_progress():
    router = _dual({**PRESETS, "resolutions": [512]}).expert_router
    values = {round(router.estimate_p_high(progress=p), 6) for p in (0.0, 0.5, 0.9)}
    assert len(values) == 1
    assert 0.25 <= router.p_high <= 0.34 and router.p_high == pytest.approx(values.pop())


def test_router_reestimates_at_each_switch_interval_from_the_trainer_progress():
    t = _dual({**RADC14, "resolutions": [1024]}, step=0, max_steps=100)
    router = t.expert_router
    first = router.p_high
    t.global_step = 90
    router.choose_expert(5)  # a fresh interval block
    assert router.p_high != first
    assert router.p_high == pytest.approx(router.estimate_p_high(progress=0.9), abs=1e-9)


def _rng_snapshot():
    cuda = [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    return torch.get_rng_state().clone(), cuda


def test_an_interval_estimate_leaves_the_global_cpu_and_cuda_rng_untouched():
    # verify 1.01: a global manual_seed in the estimate reseeded every CUDA generator.
    t = _dual({**RADC14, "resolutions": [1024]}, step=0, max_steps=100)
    t.global_step = 40
    # Advance the training RNG first (a training step drew), so a reseed cannot coincide with it.
    torch.rand(8, device=torch.device("cuda") if torch.cuda.is_available() else CPU)
    cpu_before, cuda_before = _rng_snapshot()
    t.expert_router.choose_expert(3)  # a fresh interval block -> a re-estimate
    cpu_after, cuda_after = _rng_snapshot()
    assert torch.equal(cpu_before, cpu_after)
    assert len(cuda_before) == len(cuda_after)
    assert all(torch.equal(a, b) for a, b in zip(cuda_before, cuda_after, strict=True))


def test_training_draws_after_consecutive_interval_switches_are_not_repeated():
    dev = torch.device("cuda") if torch.cuda.is_available() else CPU
    t = _dual({**RADC14, "resolutions": [1024]}, step=0, max_steps=100)
    router = t.expert_router
    t.global_step = 10
    router.choose_expert(1)
    first = torch.rand(16, device=dev)
    t.global_step = 20
    router.choose_expert(2)
    second = torch.rand(16, device=dev)
    assert not torch.equal(first, second), "the interval estimate reseeded the training RNG"


def test_band_sampler_draws_from_the_factory_at_the_current_progress(spy):
    t = _dual({**RADC14, "resolutions": [512]}, step=50, max_steps=100)
    spy.clear()
    out = t._sample_band("high", 16, latents=torch.zeros(2, 16, 1, 64, 64))
    assert bool((out >= 875.0).all())
    assert spy and all(c.mode == "radc" and c.progress == 0.5 for c in spy)
