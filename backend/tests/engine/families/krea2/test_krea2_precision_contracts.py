"""Krea-2 precision-contract tests (hardening W3-2).

First tests in a dedicated ``krea2`` family test dir. They pin the precision
invariants the Krea-2 sampler documents in its own docstring (invariant 4: "NO
autocast around the DiT forward ... fp32 trajectory") against the REAL
``Krea2Sampler.denoise`` path:

* **No autocast** around ``driver.forward_pass`` in the denoise loop — the
  autocast sampler-collapse gotcha (an autocast-wrapped N-step loop can collapse
  multi-step sampling toward the conditional mean even when training is fine).
* **fp32 trajectory** — latents start fp32, each velocity is cast ``.to(fp32)``,
  the CFG combine is fp32, and ``scheduler.step`` output is re-cast fp32, so the
  Euler accumulation never rounds in bf16.
* **Turbo vs Raw CFG branch** — ``guidance_scale == 0`` (distilled/Turbo) runs a
  SINGLE conditional pass; ``guidance_scale > 0`` (Raw) runs cond + uncond.

A stub driver records the precision regime of every ``forward_pass`` call; the
REAL ``FlowMatchEulerDiscreteScheduler`` drives the loop on CPU (GPU-free).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from app.engine.models.families.krea2.sampler import Krea2Sampler


# ── Stubs ─────────────────────────────────────────────────────────────────


class _ProbeDriver:
    """Records precision regime + input dtypes of each forward_pass call."""

    def __init__(self):
        self.observations: list[dict] = []

    def forward_pass(self, noisy_input, timesteps, text_embeddings, batch):
        self.observations.append(
            {
                "autocast": (
                    torch.is_autocast_enabled("cuda")
                    or torch.is_autocast_enabled("cpu")
                ),
                "noisy_dtype": noisy_input.dtype,
            }
        )
        return torch.zeros_like(noisy_input)


class _Transformer(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1, dtype=dtype), requires_grad=False)


def _make_sampler(dtype: torch.dtype = torch.float32, is_distilled: bool = False):
    driver = _ProbeDriver()
    transformer = _Transformer(dtype).eval()
    enc = {"n": 0}

    def _encode_text(prompts, dtype):
        enc["n"] += 1
        return torch.randn(1, 5, 4, 8, dtype=dtype), torch.ones(1, 5, dtype=torch.bool)

    definition = SimpleNamespace(
        architecture_params={
            "vae.vae_scale_factor": 8,
            "vae.latent_channels": 16,
            "transformer.patch_size": 2,
        },
        defaults={"is_distilled": is_distilled},
    )
    pipeline = SimpleNamespace(
        config={"sample_negative_prompt": ""},
        device=torch.device("cpu"),
        driver=driver,
        transformer=transformer,
        definition=definition,
        encode_text=_encode_text,
        _block_swap_managers=None,
    )
    sampler = Krea2Sampler(pipeline)
    sampler._encode = enc  # type: ignore[attr-defined]
    return sampler


def _run_denoise(sampler: Krea2Sampler, guidance_scale: float, num_steps: int = 2):
    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = sampler._create_initial_noise(32, 32, gen)  # [1, 16, 4, 4]
    emb = {
        "embeds": torch.randn(1, 5, 4, 8),
        "mask": torch.ones(1, 5, dtype=torch.bool),
    }
    return sampler.denoise(
        noise, emb, num_steps=num_steps, guidance_scale=guidance_scale, seed=0
    )


# ── Autocast sampler-collapse contract ────────────────────────────────────


@pytest.mark.parametrize("guidance_scale", [0.0, 5.0])
def test_denoise_forward_runs_without_autocast(guidance_scale):
    sampler = _make_sampler(torch.float32)
    _run_denoise(sampler, guidance_scale)

    obs = sampler.pipeline.driver.observations
    assert obs, "driver.forward_pass was never called"
    violations = [o for o in obs if o["autocast"]]
    assert not violations, (
        f"autocast enabled during {len(violations)}/{len(obs)} forwards — the "
        "autocast sampler-collapse gotcha. Krea-2 invariant 4 forbids autocast "
        "around the DiT forward."
    )


def test_denoise_trajectory_is_fp32():
    """Returned latents (fp32 Euler accumulation) stay fp32 (invariant 4)."""
    sampler = _make_sampler(torch.float32)
    out = _run_denoise(sampler, guidance_scale=5.0)
    assert out["latents"].dtype == torch.float32


def test_turbo_single_pass_raw_two_pass():
    """guidance_scale==0 (Turbo) -> 1 forward/step; >0 (Raw) -> 2 forwards/step."""
    num_steps = 2

    turbo = _make_sampler(torch.float32, is_distilled=True)
    _run_denoise(turbo, guidance_scale=0.0, num_steps=num_steps)
    assert len(turbo.pipeline.driver.observations) == num_steps, (
        "Turbo (guidance_scale=0) must run a single conditional pass per step"
    )

    raw = _make_sampler(torch.float32)
    _run_denoise(raw, guidance_scale=5.0, num_steps=num_steps)
    assert len(raw.pipeline.driver.observations) == 2 * num_steps, (
        "Raw (guidance_scale>0) runs cond + uncond per step"
    )


def test_probe_detects_autocast_bite():
    """Sanity: the autocast probe bites (production adds none)."""
    sampler = _make_sampler(torch.float32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _run_denoise(sampler, guidance_scale=0.0)
    obs = sampler.pipeline.driver.observations
    assert obs and all(o["autocast"] for o in obs)
