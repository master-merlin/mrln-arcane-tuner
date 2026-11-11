"""The MiniMax-H3-Image preview sampler (``minimax_h3_image/sampler.py``).

The model is distilled to a fixed 6-step grid with one output head per step and no CFG.
What this file holds the sampler to, on CPU, against the tiny models built from the
checkpoint's own config fixtures (never the real weights), following the reference
pipeline (diffusers PR #15230 merge commit 875373ac, ``modular_pipelines/minimax_h3_image``):

* the schedule: the 6 grid sigmas, each rounded through its float32 timestep ``1 - sigma``
  and back (``before_denoise.py`` ``MiniMaxH3ImageSetTimestepsStep``), then a terminal 0;
* step ``i`` runs head ``i`` at ``t = 1 - sigma_i`` (``denoise.py`` ``pdd_step_index=i``);
* the update is ``x + (sigma - sigma_next) * v`` in float32 (the scheduler's
  ``euler_update_form="velocity"``), and the last step lands on sigma 0;
* a canvas under 2048x2048 is generated larger at the same aspect, both sides rounded to
  the nearest multiple of 32 (``get_image_canvas_size``), and the decoded image is
  resized back to the requested size (``decoders.py``);
* no CFG and no negative prompt: a negative prompt is ignored with a warning;
* the trainer builds the sampler through ``_create_sampler`` when sampling is on.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
import torch.nn as nn
from PIL import Image

from tests.engine.families.test_minimax_h3_image_trainer import (
    H3_IMAGE_SIGMAS,
    TRANSFORMER_TINY,
    VAE_CONFIG,
    VAE_TINY,
    _HeadSpy,
    _Log,
    _tiny_config,
    _tiny_transformer,
    _vendor,
    _with_model,
    set_up,
)

PROMPT = "a red cube on a table"
MIN_AREA = 2048 * 2048  # before_denoise.py: min_generation_area default=2048 * 2048


def _sampler_module():
    from app.engine.models.families.minimax_h3_image import sampler

    return sampler


def _reference_sigmas() -> torch.Tensor:
    """before_denoise.py:679-680, verbatim: the grid through its float32 timestep and back."""
    timesteps = torch.tensor([1.0 - sigma for sigma in H3_IMAGE_SIGMAS], dtype=torch.float32)
    return torch.cat([1.0 - timesteps, torch.zeros(1)])


class _StubVAE(nn.Module):
    """A decoder of the right geometry and space: 16x nearest upsampling of the first 3
    latent channels into the VAE's ImageNet space (the shared pixel adapter's inner side)."""

    spatial_compression_ratio = 16

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.decoded: list[tuple[int, ...]] = []

        class _Cfg:
            latent_channels = 24
            latents_mean = None
            latents_std = None

        self.config = _Cfg()

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        self.decoded.append(tuple(z.shape))
        sample = torch.nn.functional.interpolate(torch.tanh(z[:, :3].float()), scale_factor=16, mode="nearest")
        return (sample,)


def _sampler(*, model: nn.Module | None = None, vae: nn.Module | None = None, **config: Any):
    trainer = set_up(**config)
    _with_model(trainer, model if model is not None else _tiny_transformer())
    if vae is not None:
        trainer.components["vae"] = vae
        trainer.driver.assign_components(trainer.components)
    trainer.vae = vae
    gen = torch.Generator().manual_seed(11)
    trainer.text_cache[PROMPT] = torch.randn(5, TRANSFORMER_TINY["text_dim"], generator=gen)
    sampler = _sampler_module().MiniMaxH3ImageSampler(trainer)
    sampler.logger = _Log()
    return sampler


def _noise(hw: tuple[int, int] = (4, 6), seed: int = 5) -> torch.Tensor:
    return torch.randn(1, 24, *hw, generator=torch.Generator().manual_seed(seed))


# ── The canvas ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "height, width, canvas",
    [
        (1024, 1024, (2048, 2048)),  # 1 MP -> exactly 4 MP
        (2048, 2048, (2048, 2048)),  # already 4 MP: unchanged
        (768, 1344, (1536, 2720)),  # nearest multiple of 32 on each side, as the reference rounds
        (512, 512, (2048, 2048)),
        (2048, 2080, (2048, 2080)),  # above the area: only rounded
    ],
)
def test_the_canvas_follows_the_reference_rule(height, width, canvas):
    sampler = _sampler_module()
    assert sampler.image_canvas_size(height, width, MIN_AREA, 32) == canvas
    scale = max(1.0, math.sqrt(MIN_AREA / (width * height)))
    expect = (max(32, round(height * scale / 32) * 32), max(32, round(width * scale / 32) * 32))
    assert canvas == expect


def test_the_minimum_area_is_the_reference_default():
    assert _sampler_module().MIN_GENERATION_AREA == MIN_AREA


# ── The schedule ──────────────────────────────────────────────────────────


def test_the_schedule_is_the_grid_rounded_through_fp32_then_zero():
    sampler = _sampler()
    sigmas = sampler.sampling_sigmas()
    assert sigmas.dtype == torch.float32
    assert sigmas.numel() == 7
    assert torch.equal(sigmas, _reference_sigmas())
    assert float(sigmas[-1]) == 0.0
    assert torch.allclose(sigmas[:-1].double(), torch.tensor(H3_IMAGE_SIGMAS, dtype=torch.float64), atol=1e-7)


def test_the_six_updates_use_the_six_heads_in_order():
    model = _tiny_transformer()
    spy = _HeadSpy(model)
    sampler = _sampler(model=model)
    emb = sampler.encode_prompt(PROMPT)
    sampler.denoise(_noise(), emb, 6, 1.0, 0)
    assert [c["head"] for c in spy.calls] == [0, 1, 2, 3, 4, 5]
    sigmas = _reference_sigmas()
    for i, call in enumerate(spy.calls):
        assert call["timestep"].dtype == torch.float32
        assert torch.equal(call["timestep"].reshape(-1), (1.0 - sigmas[i]).reshape(1))
        assert call["rows"] == (4 // 2) * (6 // 2)


def test_the_update_is_euler_on_the_velocity_in_fp32(monkeypatch):
    sampler = _sampler()
    velocities = [torch.full((1, 24, 4, 6), float(i + 1), dtype=torch.bfloat16) for i in range(6)]
    seen: list[torch.dtype] = []

    def predict(noisy, text_rows, t, head):
        seen.append(noisy.dtype)
        return velocities[head]

    monkeypatch.setattr(sampler.pipeline.driver, "predict_velocity", predict)
    noise = _noise()
    out = sampler.denoise(noise, sampler.encode_prompt(PROMPT), 6, 1.0, 0)
    sigmas = _reference_sigmas()
    expect = noise.clone().float()
    for i in range(6):
        expect = expect + (sigmas[i] - sigmas[i + 1]) * velocities[i].float()
    assert out.dtype == torch.float32
    assert torch.equal(out, expect)
    assert seen == [torch.float32] * 6  # the trajectory stays fp32 between steps


def test_the_final_sigma_is_zero(monkeypatch):
    """With a unit velocity the steps sum to sigma_0 - sigma_last = 1 - 0: the loop ends at 0."""
    sampler = _sampler()
    monkeypatch.setattr(sampler.pipeline.driver, "predict_velocity", lambda x, e, t, head: torch.ones_like(x))
    noise = _noise()
    out = sampler.denoise(noise, sampler.encode_prompt(PROMPT), 6, 1.0, 0)
    assert torch.allclose(out, noise + 1.0, atol=1e-6)


def test_the_dit_sees_autocast_off():
    model = _tiny_transformer()
    states: list[bool] = []
    original = model.forward

    def forward(*args, **kwargs):
        states.append(torch.is_autocast_enabled("cpu"))
        return original(*args, **kwargs)

    model.forward = forward
    sampler = _sampler(model=model)
    emb = sampler.encode_prompt(PROMPT)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        sampler.denoise(_noise(), emb, 6, 1.0, 0)
    assert states == [False] * 6


# ── The 4 MP rule, end to end ─────────────────────────────────────────────


def test_a_1mp_request_runs_at_4mp_and_comes_back_at_its_size():
    model = _tiny_transformer()
    spy = _HeadSpy(model)
    vae = _StubVAE()
    sampler = _sampler(model=model, vae=vae)
    image = sampler._sample_single(
        {"prompt": PROMPT, "seed": 3, "width": 1024, "height": 1024, "num_inference_steps": 6, "guidance_scale": 1.0},
        step=0,
    )
    assert isinstance(image, Image.Image)
    assert image.size == (1024, 1024)
    # 2048x2048 canvas -> 128x128 latents -> 64x64 = 4096 rows of 2x2 patches
    assert vae.decoded == [(1, 24, 128, 128)]
    assert [c["rows"] for c in spy.calls] == [4096] * 6
    assert 2048 * 2048 >= MIN_AREA


def test_a_non_square_request_keeps_its_aspect_and_size(monkeypatch):
    vae = _StubVAE()
    sampler = _sampler(vae=vae)
    monkeypatch.setattr(sampler.pipeline.driver, "predict_velocity", lambda x, e, t, head: torch.zeros_like(x))
    image = sampler._sample_single(
        {"prompt": PROMPT, "seed": 3, "width": 1344, "height": 768, "num_inference_steps": 6, "guidance_scale": 1.0},
        step=0,
    )
    assert image.size == (1344, 768)
    assert vae.decoded == [(1, 24, 1536 // 16, 2720 // 16)]


def test_the_noise_is_drawn_at_the_canvas_from_the_seed():
    sampler = _sampler(vae=_StubVAE())
    a = sampler._create_initial_noise(1024, 1024, torch.Generator().manual_seed(9))
    b = sampler._create_initial_noise(1024, 1024, torch.Generator().manual_seed(9))
    assert a.shape == (1, 24, 128, 128)
    assert a.dtype == torch.float32
    assert torch.equal(a, b)
    # the reference draws (1, C, 1, H, W) fp32 noise; the same values, one frame
    ref = torch.randn((1, 24, 1, 128, 128), generator=torch.Generator().manual_seed(9), dtype=torch.float32)
    assert torch.equal(a, ref[:, :, 0])


def test_decode_goes_through_the_vae_in_imagenet_space_and_resizes_back():
    from app.engine.components.latents import LatentManager
    from app.engine.models.families.minimax_h3_shared.pixel_adapter import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        H3PixelAdaptedVAE,
    )

    torch.manual_seed(0)
    inner = _vendor("AutoencoderKLMiniMaxH3Image").from_config(_tiny_config(VAE_CONFIG, VAE_TINY)).eval()
    vae = H3PixelAdaptedVAE(inner)
    sampler = _sampler(vae=vae)
    sampler._output_size = (48, 40)  # (height, width) of the request
    latents = torch.randn(1, 24, 4, 4, generator=torch.Generator().manual_seed(2))
    image = sampler.decode_latents(latents)

    with torch.no_grad():
        raw = LatentManager.denormalize_latents(latents, vae)
        decoded = inner.decode(raw, return_dict=False)[0].float()
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    pixels = (decoded * std + mean).clamp(0, 1)
    pixels = (pixels * 255).clamp(0, 255).to(torch.uint8)  # truncation, as the reference quantizes
    ref = Image.fromarray(pixels[0].permute(1, 2, 0).numpy()).resize((40, 48), resample=Image.LANCZOS)
    assert image.size == (40, 48)
    assert image.mode == ref.mode
    assert image.tobytes() == ref.tobytes()


# ── No CFG, no negative prompt, fixed grid ────────────────────────────────


def test_a_negative_prompt_is_ignored_with_a_warning(monkeypatch):
    sampler = _sampler(sample_negative_prompt="blurry, low quality")
    asked: list[list[str]] = []
    original = sampler.pipeline.encode_text

    def encode_text(captions, dtype, batch=None):
        asked.append(list(captions))
        return original(captions, dtype)

    monkeypatch.setattr(sampler.pipeline, "encode_text", encode_text)
    sampler.encode_prompt(PROMPT)
    sampler.encode_prompt(PROMPT)
    assert asked == [[PROMPT], [PROMPT]]
    warnings = sampler.logger.named("minimax_h3_image_negative_prompt_ignored")
    assert len(warnings) == 1


def test_no_negative_prompt_no_warning():
    sampler = _sampler()
    sampler.encode_prompt(PROMPT)
    assert sampler.logger.named("minimax_h3_image_negative_prompt_ignored") == []


def test_guidance_and_another_step_count_are_ignored_with_a_warning(monkeypatch):
    sampler = _sampler()
    calls: list[int] = []
    monkeypatch.setattr(
        sampler.pipeline.driver, "predict_velocity", lambda x, e, t, head: calls.append(head) or torch.zeros_like(x)
    )
    sampler.denoise(_noise(), sampler.encode_prompt(PROMPT), 20, 3.5, 0)
    assert calls == [0, 1, 2, 3, 4, 5]
    assert len(sampler.logger.named("minimax_h3_image_guidance_ignored")) == 1
    assert len(sampler.logger.named("minimax_h3_image_steps_fixed")) == 1


def test_the_fallbacks_are_the_models_own():
    sampler = _sampler()
    fallbacks = sampler._sample_prompt_fallbacks()
    assert fallbacks["num_inference_steps"] == 6
    assert fallbacks["guidance_scale"] == 1.0
    assert fallbacks["width"] == fallbacks["height"] == 1024


def test_the_default_round_raises_no_warning(monkeypatch):
    sampler = _sampler()
    monkeypatch.setattr(sampler.pipeline.driver, "predict_velocity", lambda x, e, t, head: torch.zeros_like(x))
    fallbacks = sampler._sample_prompt_fallbacks()
    sampler.denoise(
        _noise(), sampler.encode_prompt(PROMPT), fallbacks["num_inference_steps"], fallbacks["guidance_scale"], 0
    )
    assert sampler.logger.events == []


# ── The trainer builds it ─────────────────────────────────────────────────


def test_the_trainer_creates_the_sampler_when_sampling_is_on():
    trainer = set_up(sample_every_n_steps=50)
    sampler = trainer._create_sampler()
    assert isinstance(sampler, _sampler_module().MiniMaxH3ImageSampler)
    assert sampler.needs_live_te is False
    assert set_up(sample_every_n_steps=0)._create_sampler() is None
