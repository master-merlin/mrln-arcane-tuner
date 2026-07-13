"""OmniGen2 timestep-scale + time-convention pins (the family's silent-
LoRA-killer battery).

Facts pinned here (citations in ``families/omnigen2/driver.py`` §3-4 and
the vendored scheduler's header):

1. REVERSED clock: t=0 pure noise, t=1 clean image (checked FIRST — no
   "standard flow-match" assumption): the vendored scheduler's timesteps
   ASCEND from 0, and the perfect-velocity round-trip only closes with
   ``x_t = (1-t)*noise + t*x0`` / ``v = x0 - noise``.
2. Timestep SCALE: the transformer consumes RAW ``[0, 1)`` t and applies
   its own ``timestep_scale=1000`` inside the ``Timesteps`` proj — family
   code must never multiply/divide by 1000 (flow-match timestep-scale
   gotcha).
3. Training-time shift: upstream's ``lognorm + do_shift +
   dynamic_time_shift(v1)`` draw == ``sigma = sigmoid(N(0,1) + mu)`` with
   ``mu = lin(256->0.5, 4096->1.15)(patch_tokens)`` — the house
   ``flux_shift`` mode; the driver returns ``1 - sigma``.
4. Inference-time dynamic shift: ``t' = t / (m - m*t + t)`` with
   ``m = sqrt(num_tokens)/40`` — algebraically a flux-style
   ``sigma' = m*sigma / (m*sigma + 1 - sigma)`` in sigma-space (m = 3.2 at
   1024x1024).
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import torch


def _make_driver(arch=None):
    from app.engine.models.families.omnigen2.driver import OmniGen2Driver

    definition = MagicMock()
    definition.family = "omnigen2"
    definition.id = "omnigen2-test"
    definition.lora_targetable_modules = []
    definition.architecture_params = arch or {}
    return OmniGen2Driver(definition, torch.device("cpu"))


def _make_scheduler():
    from app.engine.models.families.omnigen2.vendor.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )

    return FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, dynamic_time_shift=True,
    )


# ── 1. Reversed clock + perfect-velocity round-trip ─────────────────────────


def test_scheduler_timesteps_ascend_from_zero():
    """The vendored scheduler walks t UPWARD from 0 (noise) toward 1
    (image) — the opposite of stock diffusers' descending sigmas. This is
    the checked-first 'reversal quirk' the brief demands evidence for."""
    s = _make_scheduler()
    s.set_timesteps(8, device="cpu")  # no num_tokens -> unshifted grid
    ts = s.timesteps.tolist()
    assert ts[0] == 0.0
    assert all(b > a for a, b in zip(ts, ts[1:])), f"not ascending: {ts}"
    assert ts[-1] < 1.0
    # set_timesteps appends the synthetic terminal 1.0 used by step().
    assert float(s._timesteps[-1]) == 1.0


def test_perfect_velocity_round_trip_recovers_x0():
    """Running the vendored scheduler from pure noise with the ORACLE
    velocity (driver.compute_target = x0 - noise) must land exactly on x0.
    Forward Euler over v = x0 - noise telescopes: x(1) = noise + Σdt·v =
    x0 — but ONLY if add_noise/compute_target/scheduler all share the same
    (inverted) convention. A sign or direction error diverges wildly."""
    drv = _make_driver()
    s = _make_scheduler()

    torch.manual_seed(3)
    x0 = torch.randn(1, 4, 8, 8)
    noise = torch.randn(1, 4, 8, 8)

    s.set_timesteps(12, device="cpu", num_tokens=8 * 8)
    latents = noise.clone()
    v = drv.compute_target(x0, noise, torch.tensor([0.0]))  # x0 - noise
    for t in s.timesteps:
        latents = s.step(v, t, latents, return_dict=False)[0]

    assert torch.allclose(latents, x0, atol=1e-5), (
        "perfect-velocity walk must recover clean latents exactly"
    )


def test_add_noise_matches_convention_endpoints():
    """add_noise: t=0 -> pure noise, t=1 -> clean latents (INVERTED
    convention); intermediate t follows x_t = (1-t)*noise + t*x0. And the
    (add_noise, compute_target) pair is consistent: x_{t2} - x_{t1} ==
    (t2 - t1) * target."""
    drv = _make_driver()
    x0 = torch.randn(2, 4, 4, 4)
    noise = torch.randn(2, 4, 4, 4)

    at_0 = drv.add_noise(x0, noise, torch.tensor([0.0, 0.0]))
    at_1 = drv.add_noise(x0, noise, torch.tensor([1.0, 1.0]))
    assert torch.allclose(at_0, noise)
    assert torch.allclose(at_1, x0)

    t1, t2 = 0.3, 0.8
    x_t1 = drv.add_noise(x0, noise, torch.tensor([t1, t1]))
    x_t2 = drv.add_noise(x0, noise, torch.tensor([t2, t2]))
    target = drv.compute_target(x0, noise, torch.tensor([t1, t1]))
    assert torch.allclose(x_t2 - x_t1, (t2 - t1) * target, atol=1e-6)


# ── 2. Timestep scale ────────────────────────────────────────────────────────


def test_transformer_applies_timestep_scale_internally():
    """The vendored Lumina2CombinedTimestepCaptionEmbedding passes
    ``timestep_scale`` as the ``Timesteps`` proj's ``scale`` — the
    transformer multiplies the [0, 1) input by 1000 INTERNALLY; callers
    must never do it."""
    from app.engine.models.families.omnigen2.vendor.models.transformers.block_lumina2 import (
        Lumina2CombinedTimestepCaptionEmbedding,
    )

    embed = Lumina2CombinedTimestepCaptionEmbedding(
        hidden_size=24, text_feat_dim=12, norm_eps=1e-5, timestep_scale=1000.0,
    )
    assert embed.time_proj.scale == 1000.0

    # A scaled clock at t and a unit clock at 1000*t produce IDENTICAL
    # sinusoidal projections — the definitional meaning of the scale.
    embed_unit = Lumina2CombinedTimestepCaptionEmbedding(
        hidden_size=24, text_feat_dim=12, norm_eps=1e-5, timestep_scale=1.0,
    )
    t = torch.tensor([0.37])
    proj_scaled = embed.time_proj(t)
    proj_unit = embed_unit.time_proj(t * 1000.0)
    # fp32 op-order differs (scale*(t·f) vs (1000t)·f) — allow float slack.
    assert torch.allclose(proj_scaled, proj_unit, rtol=1e-3, atol=1e-4)


def test_no_thousand_scaling_in_family_source():
    """Source guard: neither the driver nor the samplers multiply or divide
    timesteps by 1000 (the flow-match timestep-scale gotcha). The only 1000
    in play is the transformer's own config value."""
    import inspect

    from app.engine.models.families.omnigen2 import driver, sampler, sampler_edit

    for mod in (driver, sampler, sampler_edit):
        src = inspect.getsource(mod)
        assert "/ 1000" not in src and "* 1000" not in src, (
            f"{mod.__name__} scales timesteps by 1000 — the transformer's "
            "timestep_scale config already does this internally"
        )


# ── 3. Training-time shift ───────────────────────────────────────────────────


def test_sample_timesteps_native_range_and_noise_bias():
    """Driver timesteps are native [0, 1] (t = 1 - sigma). With the default
    flux_shift mode at 1024px-class resolution (patch tokens 4096 -> mu =
    1.15), sigma is biased TOWARD noise (mean sigmoid(N(0,1)+1.15) ≈ 0.72)
    so native t must be biased toward 0 — pinning both the shift and the
    direction of the 1-sigma conversion."""
    drv = _make_driver()
    torch.manual_seed(0)
    latents = torch.randn(4, 16, 128, 128)  # 1024px @ vae_sf 8

    ts = drv.sample_timesteps(4096, torch.device("cpu"), {}, latents=latents)
    assert ts.shape == (4096,)
    assert float(ts.min()) >= 0.0 and float(ts.max()) <= 1.0

    # E[sigmoid(N(0,1) + 1.15)] ≈ 0.72 -> E[native t] ≈ 0.28.
    mean_t = float(ts.mean())
    assert 0.20 < mean_t < 0.36, (
        f"native-t mean {mean_t:.3f} outside the shifted-lognorm band — "
        "either the flux_shift seeding or the 1-sigma conversion broke"
    )


def test_sample_timesteps_seeds_upstream_shift_constants():
    """The seeded flux_shift defaults are upstream's own lin_function
    constants (0.5 -> 1.15 over 256 -> 4096 patch tokens, patchify 2) —
    and a user-supplied override must win."""
    from unittest.mock import patch

    drv = _make_driver()
    captured = {}

    def _spy(mode, bs, device, cfg, latents=None, progress=0.0):
        captured.update(mode=mode, cfg=cfg)
        return torch.full((bs,), 0.5)

    with patch(
        "app.engine.strategies.timestep_sampling.TimestepSampler.sample",
        side_effect=_spy,
    ):
        drv.sample_timesteps(2, torch.device("cpu"), {})
        assert captured["mode"] == "flux_shift"
        assert captured["cfg"]["flux_shift_base"] == 0.5
        assert captured["cfg"]["flux_shift_max"] == 1.15
        assert captured["cfg"]["flux_shift_patchify_factor"] == 2

        drv.sample_timesteps(
            2, torch.device("cpu"),
            {"timestep_sampling": "uniform", "flux_shift_max": 3.0},
        )
        assert captured["mode"] == "uniform"
        assert captured["cfg"]["flux_shift_max"] == 3.0


# ── 4. Inference-time dynamic shift ──────────────────────────────────────────


def test_scheduler_dynamic_shift_matches_formula_and_sigma_space_identity():
    """set_timesteps(num_tokens=128*128): t' = t/(m - m*t + t) with
    m = sqrt(16384)/40 = 3.2 (the upstream comment's own example), AND the
    equivalent sigma-space statement sigma' = m*sigma/(m*sigma + 1 - sigma)
    — proving the vendored shift is a flux-style static shift with shift
    factor m, applied in the reversed clock."""
    s = _make_scheduler()
    n = 8
    s.set_timesteps(n, device="cpu", num_tokens=128 * 128)
    m = math.sqrt(128 * 128) / 40
    assert abs(m - 3.2) < 1e-9

    raw = [i / n for i in range(n)]
    for t_raw, t_shifted in zip(raw, s.timesteps.tolist()):
        expected = t_raw / (m - m * t_raw + t_raw)
        assert abs(t_shifted - expected) < 1e-6

        # sigma-space identity: 1 - t' == m*(1-t) / (m*(1-t) + t)
        sigma = 1.0 - t_raw
        sigma_shifted = m * sigma / (m * sigma + 1.0 - sigma)
        assert abs((1.0 - t_shifted) - sigma_shifted) < 1e-6


def test_scheduler_no_num_tokens_is_unshifted():
    """Without num_tokens the dynamic shift is a no-op (upstream guard:
    ``if self.config.dynamic_time_shift and num_tokens is not None``)."""
    s = _make_scheduler()
    n = 8
    s.set_timesteps(n, device="cpu")
    for i, t in enumerate(s.timesteps.tolist()):
        assert abs(t - i / n) < 1e-6
