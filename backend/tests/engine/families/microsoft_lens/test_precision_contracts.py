"""Microsoft Lens precision-regime CHARACTERIZATION tests (hardening W3-2).

Unlike ideogram4 / krea2 / the video families — which FORBID autocast around
the denoise forward — the microsoft_lens sampler DELIBERATELY runs its DiT
forward under the SAME autocast regime as training. This is an intentional,
documented decision, not the autocast-collapse bug:

    commit 7c914c50 "fix(microsoft_lens): sample DiT under training autocast +
    fp32 Euler" — "The in-training sampler ran the DiT forward with no autocast
    and accumulated Euler steps in bf16. ... once the LoRA is trained the
    forward runs in a different precision regime than it was optimized under and
    the denoising trajectory diverges into noise ... Mirror the training forward
    exactly: run the DiT under torch.autocast(pipeline.autocast_dtype,
    enabled=use_amp), and do the CFG combine + scheduler.step in fp32 ...
    Matches the SDXL/Flux1/QwenImage samplers."

The sampler.py docstring restates this (lines ~157-169). These tests therefore
CHARACTERIZE (pin) the current behavior rather than align it to the fp32-only
contract:

* the DiT forward runs UNDER autocast when ``use_amp`` is True (mirrors the
  training regime — this is the deliberate decision), and NOT under autocast
  when ``use_amp`` is False;
* the CFG combine + Euler ``scheduler.step`` run in fp32 (``latents.float()``),
  so the multi-step trajectory itself never accumulates in bf16 — this is the
  half of the decision that protects against the collapse the other families
  guard against wholesale.

Do NOT "align" this family to a no-autocast contract without re-litigating the
decision above (the collapse it prevents is the OPPOSITE failure mode: a trained
adapter sampled outside its training precision regime).

Runs on CPU with a stub driver + fake VAE (GPU-free). Note ``use_amp`` gates a
``torch.autocast(device_type="cuda", ...)`` context; the flag state is what the
probe observes (cuda-autocast does not alter the CPU forward math).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from app.engine.models.families.microsoft_lens.sampler import MicrosoftLensSampler


# ── Stubs ─────────────────────────────────────────────────────────────────


class _ProbeDriver:
    """Records the autocast regime of each forward_pass call."""

    def __init__(self):
        self.observations: list[dict] = []

    def forward_pass(self, latents, ts, text, batch):
        self.observations.append(
            {
                "autocast_cuda": torch.is_autocast_enabled("cuda"),
                "autocast_cpu": torch.is_autocast_enabled("cpu"),
            }
        )
        return torch.zeros_like(latents)


class _FakeBN:
    running_mean = torch.zeros(128)
    running_var = torch.ones(128)
    eps = 1e-5


class _FakeVAE:
    bn = _FakeBN()
    dtype = torch.float32


class _Transformer(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1, dtype=dtype), requires_grad=False)


def _make_sampler(*, use_amp, autocast_dtype=torch.bfloat16, dtype=torch.float32):
    driver = _ProbeDriver()
    transformer = _Transformer(dtype).eval()
    pipeline = SimpleNamespace(
        config={},
        device=torch.device("cpu"),
        driver=driver,
        transformer=transformer,
        vae=_FakeVAE(),
        autocast_dtype=autocast_dtype,
        use_amp=use_amp,
        _block_swap_managers=None,
    )
    return MicrosoftLensSampler(pipeline)


def _prompt_embedding(dtype=torch.float32, s_txt=5):
    def _pair():
        return (
            torch.randn(1, 4, s_txt, 2880, dtype=dtype),
            torch.ones(1, s_txt, dtype=torch.bool),
        )

    return {"cond": _pair(), "uncond": _pair()}


def _run_denoise(sampler, guidance_scale=4.0, num_steps=2):
    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = sampler._create_initial_noise(32, 32, gen)  # grid 2x2, [1, 4, 128]
    emb = _prompt_embedding()
    return sampler.denoise(
        noise, emb, num_steps=num_steps, guidance_scale=guidance_scale, seed=0
    )


# ── Deliberate autocast regime (NOT the collapse bug) ─────────────────────


def test_forward_runs_without_autocast_when_amp_off():
    """use_amp=False -> forward NOT under autocast (CPU / AMP-off path)."""
    sampler = _make_sampler(use_amp=False)
    _run_denoise(sampler)
    obs = sampler.pipeline.driver.observations
    assert obs, "driver.forward_pass was never called"
    assert not any(o["autocast_cuda"] or o["autocast_cpu"] for o in obs)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="ms_lens samples under torch.autocast(device_type='cuda'); the "
    "enabled=True context needs a CUDA build to characterize on this host.",
)
def test_forward_runs_under_autocast_when_amp_on_deliberate():
    """use_amp=True -> forward runs UNDER autocast (deliberate; commit 7c914c50).

    This is the INTENTIONAL "mirror the training regime" decision, NOT the
    autocast-collapse bug. If a future change makes this fail (forward no longer
    under autocast), re-read the commit/docstring before "fixing" it.
    """
    sampler = _make_sampler(use_amp=True, autocast_dtype=torch.bfloat16)
    _run_denoise(sampler)
    obs = sampler.pipeline.driver.observations
    assert obs, "driver.forward_pass was never called"
    assert all(o["autocast_cuda"] for o in obs), (
        "ms_lens deliberately samples the DiT under training autocast "
        "(commit 7c914c50) — this characterization must hold"
    )


def test_euler_trajectory_stays_fp32_even_under_amp():
    """The CFG combine + scheduler.step run in fp32 (the collapse-guard half).

    Even though the forward may run under autocast, the multi-step Euler
    accumulation itself is fp32 (``latents.float()`` into ``scheduler.step``), so
    the trajectory never rounds in bf16. Pinning the fp32 output documents that
    protection. Uses the AMP-off path so it runs on any host.
    """
    sampler = _make_sampler(use_amp=False)
    out = _run_denoise(sampler)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
