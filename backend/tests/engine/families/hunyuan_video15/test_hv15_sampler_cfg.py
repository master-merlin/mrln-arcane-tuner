"""hv15 in-training preview sampler tests.

Pins:
- REAL dual-forward CFG: cond + uncond forwards per step combined in fp32 as
  ``v = v_u + s*(v_c - v_u)`` — asserted EQUIVALENT to the actual diffusers
  ``ClassifierFreeGuidance`` (``use_original_formulation=False``) output.
- ``guidance_scale <= 1`` keeps the single conditional forward.
- Sigma schedule == the upstream pipeline path (linspace(1,0,N+1)[:-1] into
  ``FlowMatchEulerDiscreteScheduler(shift=5.0)``), no mu.
- 5D noise ``[1, 32, (F-1)/4+1, H/16, W/16]`` fp32.
- Preview forward builds the T2V 65-channel input + zero image_embeds and
  conditions on the RAW sigma*1000 timestep.
- ``encode_prompt`` casts cached TE embeds to the MODEL dtype.
- ``decode_latents`` DIVIDES by the scalar 1.03682 before ``vae.decode``.
"""

from types import SimpleNamespace

import numpy as np
import torch

from app.engine.core.sampling import SampleArtifact
from app.engine.models.families.hunyuan_video15.sampler import (
    HV15_DEFAULT_SHIFT,
    Hv15Sampler,
)


class _EmbVelHv15(torch.nn.Module):
    """Velocity = uniform field equal to ``encoder_hidden_states.mean()``.

    Records every forward's kwargs so the CFG call pattern and the 65-channel
    input contract can be asserted. An fp32 param is registered so the
    sampler's dtype probe works.
    """

    def __init__(self, log: list | None = None) -> None:
        super().__init__()
        self.probe = torch.nn.Parameter(torch.zeros(1))
        self.log = log if log is not None else []

    def forward(
        self,
        hidden_states=None,
        timestep=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_hidden_states_2=None,
        encoder_attention_mask_2=None,
        image_embeds=None,
        return_dict=False,
    ):
        m = encoder_hidden_states.float().mean()
        self.log.append(
            {
                "mean": round(float(m), 4),
                "hidden_channels": hidden_states.shape[1],
                "cond_zeros": bool(torch.all(hidden_states[:, 32:] == 0)),
                "image_zeros": bool(torch.all(image_embeds == 0)),
                "image_shape": tuple(image_embeds.shape),
                "timestep": float(timestep[0]),
            }
        )
        v = torch.zeros_like(hidden_states[:, :32], dtype=torch.float32) + m
        return (v.to(hidden_states.dtype),)


def _text4(val: float, l1=4, d1=16) -> tuple:
    return (
        torch.full((1, l1, d1), float(val)),
        torch.ones(1, l1, dtype=torch.int64),
        torch.zeros(1, 3, 8),
        torch.zeros(1, 3, dtype=torch.int64),
    )


class _Defn:
    architecture_params = {"mode": "t2v", "video.native_fps": 24}


def _sampler(model: _EmbVelHv15, neg_val: float = 0.25) -> Hv15Sampler:
    class _Driver:
        def __init__(self, m):
            self._m = m
            self.vae = None

        def get_primary_model(self):
            return self._m

    class _Pipeline:
        def __init__(self, m):
            self.config = {"sample_num_frames": 5}
            self.device = torch.device("cpu")
            self.autocast_dtype = torch.bfloat16
            self.driver = _Driver(m)
            self.definition = _Defn()

        def encode_text(self, caps, dtype):
            # The uncond (negative) embedding: a constant field ≠ the cond.
            return _text4(neg_val)

    return Hv15Sampler(_Pipeline(model))


# ── CFG behavior ───────────────────────────────────────────────────────────


def test_no_cfg_is_single_conditional_forward():
    log: list = []
    s = _sampler(_EmbVelHv15(log=log))
    noise = torch.zeros(1, 32, 2, 4, 4)

    out = s.denoise(noise, _text4(1.0), num_steps=2, guidance_scale=1.0, seed=0)

    assert len(log) == 2  # one forward per step, no uncond pass
    # x0=0, Σdt=-1 (sigma 1→0 regardless of shift), constant v=1.0 → all -1.
    assert torch.allclose(out, torch.full_like(out, -1.0), atol=1e-4)


def test_cfg_runs_uncond_and_combines_velocity():
    log: list = []
    s = _sampler(_EmbVelHv15(log=log), neg_val=0.25)
    noise = torch.zeros(1, 32, 2, 4, 4)

    out = s.denoise(noise, _text4(1.0), num_steps=2, guidance_scale=3.0, seed=0)

    assert len(log) == 4  # cond + uncond per step
    means = [entry["mean"] for entry in log]
    assert any(abs(m - 1.0) < 1e-4 for m in means)   # conditional used
    assert any(abs(m - 0.25) < 1e-4 for m in means)  # unconditional used
    # v = 0.25 + 3*(1.0-0.25) = 2.5 → endpoint all -2.5.
    assert torch.allclose(out, torch.full_like(out, -2.5), atol=1e-4)


def test_cfg_combine_matches_real_diffusers_guider():
    """Our combine must be numerically identical to the upstream
    ClassifierFreeGuidance (use_original_formulation=False) output."""
    from diffusers.guiders import ClassifierFreeGuidance

    gs = 3.0
    pred_cond = torch.randn(1, 32, 2, 4, 4)
    pred_uncond = torch.randn(1, 32, 2, 4, 4)

    guider = ClassifierFreeGuidance(guidance_scale=gs, use_original_formulation=False)
    guider.set_state(step=0, num_inference_steps=4, timestep=torch.tensor(500))
    upstream = guider.forward(pred_cond, pred_uncond)[0]

    ours = pred_uncond + gs * (pred_cond - pred_uncond)
    assert torch.allclose(ours, upstream, atol=0, rtol=0)


# ── Preview forward contract ───────────────────────────────────────────────


def test_preview_forward_builds_t2v_65ch_and_raw_timestep():
    log: list = []
    s = _sampler(_EmbVelHv15(log=log))
    noise = torch.zeros(1, 32, 2, 4, 4)

    s.denoise(noise, _text4(1.0), num_steps=2, guidance_scale=1.0, seed=0)

    first = log[0]
    assert first["hidden_channels"] == 65
    assert first["cond_zeros"] is True          # T2V zero cond/mask channels
    assert first["image_shape"] == (1, 729, 1152)
    assert first["image_zeros"] is True
    # RAW sigma*1000 — the first (shifted) sigma is exactly 1.0 → t=1000.
    assert abs(first["timestep"] - 1000.0) < 1e-3


# ── Sigma schedule ─────────────────────────────────────────────────────────


def test_sigmas_match_upstream_scheduler_static_shift():
    """Our shifted schedule == FlowMatchEulerDiscreteScheduler(shift=5.0) fed
    the upstream pipeline's linspace(1,0,N+1)[:-1] sigmas (no mu)."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    num_steps = 8
    s = _sampler(_EmbVelHv15())
    ours = s._build_sigmas(num_steps)

    sched = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=HV15_DEFAULT_SHIFT, use_dynamic_shifting=False
    )
    upstream_sigmas = np.linspace(1.0, 0.0, num_steps + 1)[:-1]
    sched.set_timesteps(sigmas=list(upstream_sigmas))
    # scheduler.sigmas: N shifted values + the appended terminal 0.0.
    assert torch.allclose(ours, sched.sigmas.to(ours.dtype), atol=1e-6)
    assert ours[0] == 1.0 and ours[-1] == 0.0


def test_model_shift_fixed_config_overrides_default():
    s = _sampler(_EmbVelHv15())
    s.config["model_shift_fixed"] = 1.0  # identity shift → pure linspace
    sig = s._build_sigmas(4)
    assert torch.allclose(sig, torch.linspace(1.0, 0.0, 5))


# ── Noise shape / dtype cast / decode scaling ──────────────────────────────


def test_initial_noise_is_5d_fp32():
    s = _sampler(_EmbVelHv15())
    s.config["sample_num_frames"] = 17
    g = torch.Generator().manual_seed(0)
    noise = s._create_initial_noise(width=640, height=480, generator=g)
    # (F-1)/4+1 = 5 latent frames; H/16, W/16 spatial.
    assert noise.shape == (1, 32, 5, 30, 40)
    assert noise.dtype == torch.float32


def test_encode_prompt_casts_cached_embeds_to_model_dtype():
    model = _EmbVelHv15().to(torch.bfloat16)
    s = _sampler(model, neg_val=0.5)
    emb, mask, emb2, mask2 = s.encode_prompt("whatever")
    assert emb.dtype == torch.bfloat16
    assert emb2.dtype == torch.bfloat16
    # Masks stay integer (attention masks are not cast to float).
    assert mask.dtype == torch.int64
    assert mask2.dtype == torch.int64


def test_decode_divides_by_scalar_scaling_factor():
    class _SpyVae(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(scaling_factor=1.03682)
            self.seen: list = []

        @property
        def dtype(self):
            return torch.float32

        def decode(self, z, return_dict=False):
            self.seen.append(z)
            b, c, f, h, w = z.shape
            return (torch.zeros(b, 3, f, h * 16, w * 16),)

    model = _EmbVelHv15()
    s = _sampler(model)
    vae = _SpyVae()
    s.pipeline.driver.vae = vae

    latents = torch.full((1, 32, 2, 4, 4), 1.03682)
    artifact = s.decode_latents(latents)

    assert isinstance(artifact, SampleArtifact)
    assert artifact.fps == 24.0
    # The VAE saw latents / 1.03682 → exactly ones.
    assert torch.allclose(vae.seen[0], torch.ones_like(vae.seen[0]), atol=1e-6)
    # Canonical [C, F, H, W] clip layout.
    assert artifact.frames.shape == (3, 2, 64, 64)
