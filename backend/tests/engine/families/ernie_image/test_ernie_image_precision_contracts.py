"""ERNIE-Image sampler precision-regime contract (W5.T10).

House-wide contract (already pinned for zimage/qwen_image/boogu_image, see
MEMORY: "autocast sampler collapse gotcha"): the Euler denoising trajectory
must be STORED/ACCUMULATED in fp32 for the entire loop — only the per-step
transformer INPUT is cast to the model's native (possibly reduced-precision)
dtype, and the raw prediction is cast straight back to fp32 before the CFG
combine + scheduler.step. A trajectory that persists in bf16/fp16 across
steps can collapse sampling toward the conditional mean once the LoRA is
non-trivial (the model's velocities were learned in fp32/autocast, not a
bf16-accumulated recurrence).

Runs on CPU with a stub transformer (records the dtype it's called with) +
a real ``FlowMatchEulerDiscreteScheduler`` — no GPU, no real weights.
"""

from __future__ import annotations

import types

import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.ernie_image.sampler import ErnieImageSampler


class _CaptureTransformer(torch.nn.Module):
    """Records the dtype of every ``hidden_states`` it's forwarded; returns a
    same-shaped (native-dtype) tensor, mirroring a real velocity prediction."""

    def __init__(
        self, in_channels: int = 8, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(in_channels=in_channels)
        self._p = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self.seen_dtypes: list[torch.dtype] = []

    def forward(self, *, hidden_states, timestep, text_bth, text_lens, return_dict):
        self.seen_dtypes.append(hidden_states.dtype)
        return (torch.zeros_like(hidden_states),)


def _defn() -> ModelDefinition:
    return ModelDefinition(
        id="ernie-image-base",
        family="ernie_image",
        name="ERNIE-Image",
        defaults={},
        components={},
    )


def _make_sampler(in_channels: int = 8) -> ErnieImageSampler:
    pipeline = types.SimpleNamespace(
        config={},
        device=torch.device("cpu"),
        definition=_defn(),
        transformer=_CaptureTransformer(in_channels=in_channels),
    )
    return ErnieImageSampler(pipeline)


def _prompt_embedding(s_txt: int = 6, dim: int = 16) -> dict:
    def _pair():
        emb = torch.randn(1, s_txt, dim, dtype=torch.bfloat16)
        mask = torch.ones(1, s_txt, dtype=torch.long)
        return emb, mask

    cond_emb, cond_mask = _pair()
    uncond_emb, uncond_mask = _pair()
    return {
        "cond_emb": cond_emb,
        "cond_mask": cond_mask,
        "uncond_emb": uncond_emb,
        "uncond_mask": uncond_mask,
    }


def test_create_initial_noise_is_fp32():
    sampler = _make_sampler()
    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = sampler._create_initial_noise(width=128, height=128, generator=gen)
    assert noise.dtype == torch.float32


def test_denoise_trajectory_stays_fp32_while_transformer_sees_native_dtype():
    """The RETURNED (and internally accumulated) latents never leave fp32,
    even though the transformer itself is forwarded in its native bf16 —
    proving the cast is per-forward, not a persisted trajectory dtype."""
    sampler = _make_sampler()
    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = sampler._create_initial_noise(width=128, height=128, generator=gen)
    emb = _prompt_embedding()

    out = sampler.denoise(noise, emb, num_steps=3, guidance_scale=1.0, seed=0)

    assert out.dtype == torch.float32
    transformer = sampler.pipeline.transformer
    assert len(transformer.seen_dtypes) == 3
    assert all(d == torch.bfloat16 for d in transformer.seen_dtypes)


def test_denoise_with_cfg_trajectory_stays_fp32():
    """Same contract with the two-pass CFG combine engaged."""
    sampler = _make_sampler()
    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = sampler._create_initial_noise(width=128, height=128, generator=gen)
    emb = _prompt_embedding()

    out = sampler.denoise(noise, emb, num_steps=2, guidance_scale=4.0, seed=0)

    assert out.dtype == torch.float32
    transformer = sampler.pipeline.transformer
    assert len(transformer.seen_dtypes) == 2
    assert all(d == torch.bfloat16 for d in transformer.seen_dtypes)


def test_denoise_output_finite_and_shape_matches_noise():
    sampler = _make_sampler()
    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = sampler._create_initial_noise(width=128, height=128, generator=gen)
    emb = _prompt_embedding()

    out = sampler.denoise(noise, emb, num_steps=2, guidance_scale=1.0, seed=0)

    assert out.shape == noise.shape
    assert torch.isfinite(out).all()
