"""Qwen-Image precision-contract + D5 no-CFG tests (hardening W3-2).

Two things are pinned against the REAL ``QwenImageSampler.denoise`` path:

1. **Autocast sampler-collapse contract.** The flow-match Euler loop must run
   the transformer forward with NO ``torch.autocast`` — wrapping an N-step loop
   in ``autocast(bf16)`` can collapse multi-step sampling toward the conditional
   mean even when training is fine (project gotcha). A stub transformer records
   the precision regime of every real forward call; the REAL diffusers scheduler
   drives the loop on CPU with tiny latents (GPU-free).

2. **D5 — no-CFG previews are INTENTIONAL (decision 2026-07-11, hardening).**
   Qwen-Image in-training previews NEVER apply classifier-free guidance: this is
   faithful to the upstream ``QwenImagePipeline``, which passes ``guidance=None``
   (``guidance_embeds: false``) and runs a SINGLE conditional forward per step —
   there is no negative-prompt branch. ``denoise`` accepts a ``guidance_scale``
   argument for API uniformity but deliberately ignores it. The test below pins
   that: exactly ``num_steps`` forwards happen (never ``2 x num_steps``) and
   ``guidance`` is always ``None``, regardless of ``guidance_scale``. Do NOT
   "fix" this by adding a CFG branch — it would diverge from upstream.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from app.engine.models.families.qwen_image.sampler import QwenImageSampler


# ── Stubs ─────────────────────────────────────────────────────────────────


class _ProbeTransformer(torch.nn.Module):
    """DiT stand-in recording the precision regime + guidance of each forward."""

    def __init__(self, dtype: torch.dtype = torch.float32, in_channels: int = 64):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1, dtype=dtype), requires_grad=False)
        self.config = SimpleNamespace(in_channels=in_channels)
        self.observations: list[dict] = []

    def forward(self, *, hidden_states, timestep, guidance, **kwargs):  # noqa: D102
        self.observations.append(
            {
                "autocast": (
                    torch.is_autocast_enabled("cuda")
                    or torch.is_autocast_enabled("cpu")
                ),
                "hidden_dtype": hidden_states.dtype,
                "guidance": guidance,
            }
        )
        return (torch.zeros_like(hidden_states),)


class _FakeVAE:
    # No temperal_downsample attr -> sampler falls back to vae_sf = 8.
    dtype = torch.float32


def _make_sampler(dtype: torch.dtype = torch.float32) -> QwenImageSampler:
    transformer = _ProbeTransformer(dtype).eval()
    pipeline = SimpleNamespace(
        config={},
        device=torch.device("cpu"),
        transformer=transformer,
        vae=_FakeVAE(),
        definition=SimpleNamespace(architecture_params={}),
        _block_swap_managers=None,
    )
    return QwenImageSampler(pipeline)


def _run_denoise(sampler: QwenImageSampler, num_steps: int, guidance_scale: float):
    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = sampler._create_initial_noise(32, 32, gen)  # [1, 4, 64]
    emb = {
        "embeds": torch.randn(1, 5, 16),
        "mask": torch.ones(1, 5, dtype=torch.long),
    }
    return sampler.denoise(
        noise, emb, num_steps=num_steps, guidance_scale=guidance_scale, seed=0
    )


# ── Autocast sampler-collapse contract ────────────────────────────────────


@pytest.mark.parametrize("guidance_scale", [1.0, 4.0])
def test_denoise_forward_runs_without_autocast(guidance_scale):
    sampler = _make_sampler(torch.float32)
    _run_denoise(sampler, num_steps=2, guidance_scale=guidance_scale)

    obs = sampler.pipeline.transformer.observations
    assert obs, "transformer forward was never called"
    violations = [o for o in obs if o["autocast"]]
    assert not violations, (
        f"autocast enabled during {len(violations)}/{len(obs)} denoise forwards "
        "— the autocast sampler-collapse gotcha. QwenImage must run the forward "
        "with NO autocast (upstream QwenImagePipeline regime)."
    )


def test_probe_detects_autocast_bite():
    """Sanity: the autocast probe bites (production adds none)."""
    sampler = _make_sampler(torch.float32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _run_denoise(sampler, num_steps=2, guidance_scale=1.0)
    obs = sampler.pipeline.transformer.observations
    assert obs and all(o["autocast"] for o in obs)


# ── D5: no-CFG previews are INTENTIONAL (upstream-faithful) ────────────────


@pytest.mark.parametrize("guidance_scale", [1.0, 3.0, 7.5])
def test_previews_never_apply_cfg_d5(guidance_scale):
    """Exactly num_steps forwards + guidance=None for ANY guidance_scale.

    Decision D5 (2026-07-11, hardening program): Qwen-Image previews are
    upstream-faithful and do NOT run classifier-free guidance. A CFG preview
    would call the transformer twice per step (cond + uncond); the upstream
    pipeline calls it ONCE per step with ``guidance=None``. This pins the
    single-pass, no-CFG behavior so a future session does not "add CFG".
    """
    num_steps = 3
    sampler = _make_sampler(torch.float32)
    _run_denoise(sampler, num_steps=num_steps, guidance_scale=guidance_scale)

    obs = sampler.pipeline.transformer.observations
    assert len(obs) == num_steps, (
        f"expected exactly {num_steps} forwards (single conditional pass per "
        f"step, no CFG), got {len(obs)} — a doubled count means a negative/CFG "
        "branch crept in, diverging from the upstream QwenImagePipeline (D5)."
    )
    assert all(o["guidance"] is None for o in obs), (
        "QwenImage config has guidance_embeds=false; guidance must stay None (D5)"
    )
