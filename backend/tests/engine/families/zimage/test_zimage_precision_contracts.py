"""Z-Image precision-contract tests (hardening W3-2).

Pins the **autocast sampler-collapse** gotcha against the REAL
``ZImageSampler.denoise`` path, plus the two dtype invariants the Z-Image
sampler deliberately maintains (see ``sampler.py``):

* **No autocast** around the transformer forward in the denoise loop.
* **fp32 trajectory** — the sampler starts from ``noise.to(float32)``, does the
  CFG combine + velocity math in fp32, and feeds ``scheduler.step`` fp32
  tensors, so the multi-step accumulation never rounds in bf16 (which is what
  the autocast-collapse gotcha is about).
* **Velocity negation** — Z-Image negates the model velocity before the
  scheduler step (``noise_pred = -noise_pred``, upstream ``ZImagePipeline``).

A stub transformer records the precision regime at every real forward; the REAL
``FlowMatchEulerDiscreteScheduler`` drives the loop on CPU with tiny latents
(GPU-free).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from app.engine.models.families.zimage.sampler import ZImageSampler


# ── Stubs ─────────────────────────────────────────────────────────────────


class _ProbeModel(torch.nn.Module):
    """Z-Image DiT stand-in (called as ``model(x=..., t=..., cap_feats=...)``).

    Records the precision regime per forward and returns a per-sample velocity
    of the same shape as the input list elements ([C, 1, H, W] each).
    """

    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1, dtype=dtype), requires_grad=False)
        self.observations: list[dict] = []

    def forward(self, *, x, t, cap_feats, return_dict=False):  # noqa: D102
        self.observations.append(
            {
                "autocast": (
                    torch.is_autocast_enabled("cuda")
                    or torch.is_autocast_enabled("cpu")
                ),
                "x_dtype": x[0].dtype,
            }
        )
        # velocity per sample, same [C, 1, H, W] shape as each input element
        out = torch.stack([torch.ones_like(xi) for xi in x], dim=0)
        return (out,)


class _FakeVAE:
    config = SimpleNamespace(block_out_channels=[1, 2, 3, 4])  # vae_sf = 2^3 = 8


def _make_sampler(dtype: torch.dtype = torch.float32) -> ZImageSampler:
    model = _ProbeModel(dtype).eval()
    calls = {"encode_text": 0}

    def _encode_text(prompts, dtype):
        calls["encode_text"] += 1
        return [torch.randn(3, 16, dtype=dtype)]

    pipeline = SimpleNamespace(
        config={},
        device=torch.device("cpu"),
        model=model,
        vae=_FakeVAE(),
        definition=SimpleNamespace(architecture_params={"in_channels": 16}),
        encode_text=_encode_text,
        _block_swap_managers=None,
    )
    sampler = ZImageSampler(pipeline)
    sampler._encode_calls = calls  # type: ignore[attr-defined]
    return sampler


def _run_denoise(sampler: ZImageSampler, guidance_scale: float, num_steps: int = 2):
    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = sampler._create_initial_noise(32, 32, gen)  # [1, 16, 4, 4]
    emb = {"embeds": [torch.randn(3, 16)]}
    return sampler.denoise(
        noise, emb, num_steps=num_steps, guidance_scale=guidance_scale, seed=0
    )


# ── Autocast sampler-collapse contract ────────────────────────────────────


@pytest.mark.parametrize("guidance_scale", [1.0, 4.0])
def test_denoise_forward_runs_without_autocast(guidance_scale):
    sampler = _make_sampler(torch.float32)
    _run_denoise(sampler, guidance_scale)

    obs = sampler.pipeline.model.observations
    assert obs, "model forward was never called"
    violations = [o for o in obs if o["autocast"]]
    assert not violations, (
        f"autocast enabled during {len(violations)}/{len(obs)} denoise forwards "
        "— the autocast sampler-collapse gotcha. Z-Image runs the forward with "
        "NO autocast (upstream ZImagePipeline regime)."
    )


def test_denoise_trajectory_is_fp32():
    """The returned latent (scheduler-accumulated trajectory) stays fp32."""
    sampler = _make_sampler(torch.float32)
    out = _run_denoise(sampler, guidance_scale=4.0)
    assert out.dtype == torch.float32, (
        "Z-Image accumulates the Euler trajectory in fp32 (CFG combine + "
        "scheduler.step fed fp32); a non-fp32 result means bf16 crept into the "
        "multi-step accumulation (collapse risk)."
    )


def test_probe_detects_autocast_bite():
    """Sanity: the autocast probe bites (production adds none)."""
    sampler = _make_sampler(torch.float32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _run_denoise(sampler, guidance_scale=1.0)
    obs = sampler.pipeline.model.observations
    assert obs and all(o["autocast"] for o in obs)
