"""FLUX.2 Klein precision-contract tests (hardening W3-2).

First tests ever for the ``flux2`` family. They pin the **autocast
sampler-collapse** gotcha against the REAL ``Flux2Sampler.denoise`` code path:

* the Klein denoise loop must run the transformer forward **without**
  ``torch.autocast`` — wrapping an N-step flow-match loop in ``autocast(bf16)``
  can collapse multi-step sampling toward the conditional mean (a blurry
  average) even when training is fine (project gotcha; ideogram4 GPU ablation
  2026-06-10 measured cos(z_final, fp32 ref) = 0.32 → flat image under autocast
  vs ~1.0 without).

* the denoise trajectory runs in the loaded transformer's dtype with **no
  hidden re-promotion**: this mirrors the upstream ``Flux2KleinPipeline``, which
  runs the model in its native dtype with no autocast around the sampling loop.

The probe is the ideogram4 template style: a stub transformer records, at every
real forward call, whether an autocast region is active and the dtype of the
inputs it received. The REAL scheduler (``FlowMatchEulerDiscreteScheduler``)
drives the loop on CPU with tiny latents, so this is GPU-free.

NOTE: unlike ideogram4/krea2 (which forbid ANY non-fp32 trajectory), flux2
legitimately runs the trajectory in the model dtype — that is upstream-faithful
and is NOT a bug. The invariant pinned here is *no autocast* around the forward.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from app.engine.models.families.flux2.sampler import Flux2Sampler


# ── Stubs ─────────────────────────────────────────────────────────────────


class _ProbeTransformer(torch.nn.Module):
    """DiT stand-in that records the precision regime of every forward call.

    Returns ``zeros_like(hidden_states)`` so the real scheduler can step, and
    appends one observation per call to ``self.observations``.
    """

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        # A real parameter so ``next(.parameters()).dtype`` probes the dtype and
        # ``.to(device)`` behaves like a model.
        self.w = torch.nn.Parameter(torch.zeros(1, dtype=dtype), requires_grad=False)
        self.observations: list[dict] = []

    def forward(self, *, hidden_states, timestep, **kwargs):  # noqa: D102
        self.observations.append(
            {
                "autocast": (
                    torch.is_autocast_enabled("cuda")
                    or torch.is_autocast_enabled("cpu")
                ),
                "hidden_dtype": hidden_states.dtype,
                "timestep_dtype": timestep.dtype,
            }
        )
        return (torch.zeros_like(hidden_states),)


class _FakeBN:
    # 128 = 32 VAE channels * 2*2 patchify (matches _create_initial_noise).
    running_mean = torch.zeros(128)
    running_var = torch.ones(128)


class _FakeVAE:
    bn = _FakeBN()
    config = SimpleNamespace(batch_norm_eps=1e-5)
    dtype = torch.float32


def _make_sampler(dtype: torch.dtype = torch.float32) -> Flux2Sampler:
    transformer = _ProbeTransformer(dtype).eval()
    pipeline = SimpleNamespace(
        config={},
        device=torch.device("cpu"),
        transformer=transformer,
        vae=_FakeVAE(),
        _block_swap_managers=None,
    )
    return Flux2Sampler(pipeline)


def _prompt_embedding(sampler: Flux2Sampler, dtype: torch.dtype):
    cond = torch.randn(1, 4, 16, dtype=dtype)
    uncond = torch.randn(1, 4, 16, dtype=dtype)
    return {
        "cond": cond,
        "cond_ids": sampler._prepare_text_ids(cond),
        "uncond": uncond,
        "uncond_ids": sampler._prepare_text_ids(uncond),
    }


def _run_denoise(sampler: Flux2Sampler, dtype: torch.dtype, guidance_scale: float):
    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = sampler._create_initial_noise(32, 32, gen)  # 2x2 grid, [1, 4, 128]
    emb = _prompt_embedding(sampler, dtype)
    return sampler.denoise(noise, emb, num_steps=2, guidance_scale=guidance_scale, seed=0)


# ── Precision contract: no autocast around the denoise forward ────────────


@pytest.mark.parametrize("guidance_scale", [1.0, 3.5])
def test_denoise_forward_runs_without_autocast(guidance_scale):
    """The REAL Klein denoise must never enter an autocast region."""
    sampler = _make_sampler(torch.float32)
    _run_denoise(sampler, torch.float32, guidance_scale)

    obs = sampler.pipeline.transformer.observations
    assert obs, "transformer forward was never called"
    violations = [o for o in obs if o["autocast"]]
    assert not violations, (
        f"autocast was enabled during {len(violations)}/{len(obs)} denoise "
        "forwards — the autocast sampler-collapse gotcha. The Klein loop must "
        "run the forward with NO autocast (upstream Flux2KleinPipeline regime)."
    )


def test_denoise_trajectory_stays_in_model_dtype():
    """Trajectory dtype follows the loaded model (no hidden re-promotion).

    Upstream-faithful: the model runs in its native dtype; the forward receives
    inputs in exactly that dtype (here fp32), not silently upcast/downcast by an
    autocast wrapper.
    """
    sampler = _make_sampler(torch.float32)
    _run_denoise(sampler, torch.float32, guidance_scale=3.5)
    obs = sampler.pipeline.transformer.observations
    assert all(o["hidden_dtype"] == torch.float32 for o in obs)


def test_probe_detects_autocast_bite():
    """Sanity: the autocast probe is NOT vacuous.

    Wrapping the SAME real denoise call in an explicit autocast region makes the
    probe flag every forward — proving the assertion above bites and that the
    production loop (unwrapped) genuinely adds none.
    """
    sampler = _make_sampler(torch.float32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _run_denoise(sampler, torch.float32, guidance_scale=1.0)
    obs = sampler.pipeline.transformer.observations
    assert obs and all(o["autocast"] for o in obs), (
        "probe failed to detect an explicit autocast region — the no-autocast "
        "assertion would be vacuous"
    )
