"""Classifier-free guidance in the WAN in-training preview sampler.

The WAN 2.1/2.2 preview ``denoise`` accepted ``guidance_scale`` but never used
it — every in-training sample was effectively CFG=1, massively under-showing the
LoRA vs ComfyUI (cfg~3.5). These tests pin true CFG: a conditional +
unconditional (negative-prompt) forward combined on the velocity in fp32
(``v = v_u + s*(v_c - v_u)``); ``guidance_scale<=1`` keeps the single forward
(byte-identical to before).

Mirrors the LTX-2 CFG tests (commit 0b47bb5): a velocity field that equals the
prompt embedding's mean makes the conditional vs unconditional velocities
distinct, so the CFG combine can be pinned to an exact endpoint.
"""

from __future__ import annotations

import torch

from app.engine.models.families.wan21.sampler import Wan21Sampler
from app.engine.models.families.wan22.sampler import Wan22Sampler


class _EmbVelWan(torch.nn.Module):
    """Velocity = uniform field equal to ``encoder_hidden_states.mean()``.

    Makes the velocity depend on the PROMPT, so a conditional vs unconditional
    embedding produce different velocities — which lets us pin the CFG combine
    ``v = v_uncond + scale * (v_cond - v_uncond)`` to an exact endpoint. An fp32
    param is registered FIRST (like the real WAN ``scale_shift_table``) so the
    sampler's dtype probe works.
    """

    def __init__(self, channels: int = 16, log: list | None = None, name: str = "") -> None:
        super().__init__()
        self.scale_shift_table = torch.nn.Parameter(torch.zeros(2, channels))
        self.log = log if log is not None else []
        self.name = name

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_image=None,
        return_dict=False,
    ):
        m = encoder_hidden_states.float().mean()
        self.log.append((self.name, round(float(m), 4)))
        v = torch.zeros_like(hidden_states, dtype=torch.float32) + m
        return (v.to(hidden_states.dtype),)


def _cond_emb(val: float = 1.0, ch: int = 16) -> torch.Tensor:
    return torch.full((1, 4, ch), float(val), dtype=torch.float32)


# ── WAN 2.1 (shared base denoise) ─────────────────────────────────────────


def _wan21_sampler(model: _EmbVelWan, neg_val: float = 0.25) -> Wan21Sampler:
    """WAN 2.1 sampler whose uncond (negative) embedding is a ``neg_val`` field."""

    class _Driver:
        def __init__(self, m):
            self._m = m

        def get_primary_model(self):
            return self._m

    class _Pipeline:
        def __init__(self, m):
            self.config = {"sample_num_frames": 5}
            self.device = torch.device("cpu")
            self.autocast_dtype = torch.bfloat16
            self.driver = _Driver(m)

        # encode_prompt(neg) routes through the trainer's cached encode_text;
        # here the uncond embedding is a constant field distinct from the cond.
        def encode_text(self, caps, dtype):
            return _cond_emb(neg_val)

    return Wan21Sampler(_Pipeline(model))


def test_wan21_denoise_no_cfg_is_single_conditional_forward():
    """guidance_scale<=1: one forward/step, velocity = the conditional only."""
    log: list = []
    model = _EmbVelWan(log=log)
    s = _wan21_sampler(model)
    noise = torch.zeros(1, 16, 2, 4, 4)

    out = s.denoise(noise, _cond_emb(1.0), num_steps=2, guidance_scale=1.0, seed=0)

    assert len(log) == 2  # one forward per step, no uncond pass
    # x0=0, Σdt=-1, constant v=cond.mean()=1.0 → endpoint all -1.0.
    assert torch.allclose(out, torch.full_like(out, -1.0), atol=1e-4)


def test_wan21_denoise_cfg_runs_uncond_and_combines_velocity():
    """guidance_scale>1: cond+uncond forwards, combined v = u + s*(c-u)."""
    log: list = []
    model = _EmbVelWan(log=log)
    s = _wan21_sampler(model, neg_val=0.25)
    noise = torch.zeros(1, 16, 2, 4, 4)

    out = s.denoise(noise, _cond_emb(1.0), num_steps=2, guidance_scale=3.0, seed=0)

    assert len(log) == 4  # two forwards per step (conditional + unconditional)
    means = [m for _, m in log]
    assert any(abs(m - 1.0) < 1e-4 for m in means)   # conditional used
    assert any(abs(m - 0.25) < 1e-4 for m in means)  # unconditional used
    # v = 0.25 + 3*(1.0-0.25) = 2.5 (constant) → endpoint all -2.5.
    assert torch.allclose(out, torch.full_like(out, -2.5), atol=1e-4)


# ── WAN 2.2 (dual-expert override) ─────────────────────────────────────────


def _wan22_sampler(high, low, boundary: float, neg_val: float = 0.25) -> Wan22Sampler:
    class _Driver:
        def __init__(self):
            self.boundary = boundary
            self.transformer_high = high
            self.transformer_low = low

        def get_primary_model(self):
            return high

    class _Pipeline:
        def __init__(self):
            self.config = {"resolutions": [480], "sample_num_frames": 5}
            self.device = torch.device("cpu")
            self.autocast_dtype = torch.bfloat16
            self.driver = _Driver()

        def encode_text(self, caps, dtype):
            return _cond_emb(neg_val)

    return Wan22Sampler(_Pipeline())


def test_wan22_denoise_no_cfg_is_single_conditional_forward():
    """WAN 2.2 override: guidance_scale<=1 keeps the single conditional forward."""
    log: list = []
    high = _EmbVelWan(log=log, name="high")
    low = _EmbVelWan(log=log, name="low")
    s = _wan22_sampler(high, low, boundary=0.5)
    noise = torch.zeros(1, 16, 2, 4, 4)

    out = s.denoise(noise, _cond_emb(1.0), num_steps=4, guidance_scale=1.0, seed=0)

    assert len(log) == 4  # one forward per step
    assert torch.allclose(out, torch.full_like(out, -1.0), atol=1e-4)


def test_wan22_denoise_cfg_combines_velocity():
    """WAN 2.2 override runs cond+uncond and combines v = u + s*(c-u)."""
    log: list = []
    high = _EmbVelWan(log=log, name="high")
    low = _EmbVelWan(log=log, name="low")
    s = _wan22_sampler(high, low, boundary=0.5, neg_val=0.25)
    noise = torch.zeros(1, 16, 2, 4, 4)

    out = s.denoise(noise, _cond_emb(1.0), num_steps=4, guidance_scale=3.0, seed=0)

    assert len(log) == 8  # two forwards per step across 4 steps
    # v = 0.25 + 3*(1.0-0.25) = 2.5 → endpoint all -2.5.
    assert torch.allclose(out, torch.full_like(out, -2.5), atol=1e-4)


def test_wan22_cfg_uncond_uses_same_expert_as_cond_per_step():
    """The uncond forward must go through the SAME expert as the cond forward
    at each step (dual-expert routing is sigma-based; both calls share the sigma).
    """
    log: list = []
    high = _EmbVelWan(log=log, name="high")
    low = _EmbVelWan(log=log, name="low")
    # boundary 0.5 → early (high-sigma) steps use high, later steps use low.
    s = _wan22_sampler(high, low, boundary=0.5, neg_val=0.25)
    noise = torch.zeros(1, 16, 2, 4, 4)

    s.denoise(noise, _cond_emb(1.0), num_steps=6, guidance_scale=3.0, seed=0)

    # 6 steps × (cond + uncond) = 12 calls, arriving as per-step pairs.
    assert len(log) == 12
    experts_used = set()
    for i in range(0, len(log), 2):
        cond_name, cond_mean = log[i]
        unc_name, unc_mean = log[i + 1]
        assert cond_name == unc_name  # same expert for cond & uncond in a step
        assert abs(cond_mean - 1.0) < 1e-4   # cond first
        assert abs(unc_mean - 0.25) < 1e-4   # uncond second
        experts_used.add(cond_name)
    # The schedule must actually cross the boundary (both experts exercised),
    # else the "same expert" assertion would be vacuous.
    assert experts_used == {"high", "low"}
