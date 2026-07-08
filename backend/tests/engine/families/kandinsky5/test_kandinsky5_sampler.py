"""Kandinsky 5.0 sampler tests — fp32 trajectory, CFG, channels-last slice.

Pins the four sampler contracts:

1. fp32 Euler trajectory, no autocast collapse (precision-contract harness
   driving the REAL ``euler_integrate``),
2. inline dual-forward CFG math ``v_u + gs * (v_c - v_u)`` + the pipeline's
   DEFAULT negative prompt injection when none is configured,
3. channels-last step slice: only ``[..., :C]`` of the (cond-concatenated)
   state advances,
4. I2V frame-0 skip: an image-conditioned denoise never updates frame 0.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from app.engine.core.sampling import SampleArtifact
from app.engine.core.text_encoding import TextEncoderOutput
from app.engine.models.families.kandinsky5.driver import (
    KANDINSKY5_DEFAULT_NEGATIVE_PROMPT,
    Kandinsky5Driver,
    build_cu_seqlens,
    tiny_transformer_config,
)
from app.engine.models.families.kandinsky5.sampler import Kandinsky5Sampler
from tests.engine.precision_contracts import (
    LinearVelocityFakeTransformer,
    assert_no_autocast_collapse,
)


def _definition(**arch_extra) -> MagicMock:
    d = MagicMock()
    d.family = "kandinsky5"
    d.id = "k5-test"
    d.lora_targetable_modules = []
    d.architecture_params = {
        "mode": "t2v",
        "transformer.in_visual_dim": 4,
        "transformer.num_visual_blocks": 1,
        "video.vae_spatial": 8,
        "scheduler.shift": 5.0,
        **arch_extra,
    }
    return d


def _teo(length: int = 6, marker: float | None = None) -> TextEncoderOutput:
    emb = torch.randn(1, length, 16)
    if marker is not None:
        emb = torch.full((1, length, 16), marker)
    return TextEncoderOutput(
        embeddings=emb,
        attention_mask=build_cu_seqlens([length]),
        pooled=torch.randn(1, 8),
    )


def _sampler(config: dict | None = None, transformer=None) -> Kandinsky5Sampler:
    definition = _definition()
    driver = Kandinsky5Driver(definition, torch.device("cpu"))
    driver.transformer = transformer
    driver.visual_cond = True
    pipeline = SimpleNamespace(
        config=config or {},
        device=torch.device("cpu"),
        driver=driver,
        definition=definition,
        _block_swap_managers=None,
    )
    s = Kandinsky5Sampler(pipeline)
    return s


# ── 1. fp32 trajectory / no autocast collapse ──────────────────────────────


def test_no_autocast_collapse_through_real_integrator():
    s = _sampler()
    fake = LinearVelocityFakeTransformer(a=-1.0, b=0.5)
    assert_no_autocast_collapse(
        s.build_denoise(fake),
        fake.analytic_endpoint_euler,
        steps=8,
        atol=1e-3,
    )


def test_euler_integrate_accumulates_fp32_even_for_bf16_velocity():
    s = _sampler()

    def bf16_velocity(x, sigma):
        return torch.full_like(x, 0.125).to(torch.bfloat16)

    x0 = torch.zeros(1, 2, 4, 4, 4)
    sigmas = torch.linspace(1.0, 0.0, 9)
    out = s.euler_integrate(x0, sigmas, bf16_velocity)
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.full_like(out, -0.125), atol=1e-6)


# ── 2. CFG math + default negative injection ───────────────────────────────


def test_cfg_dual_forward_combines_velocities():
    """v = v_u + gs * (v_c - v_u), read off a transformer whose velocity is
    the mean of its text embedding (cond=2.0, uncond=0.5)."""
    s = _sampler(config={"sample_num_frames": 5})

    def fake_transformer(**kwargs):
        val = kwargs["encoder_hidden_states"].float().mean()
        return (torch.full(
            (*kwargs["hidden_states"].shape[:-1], 4), float(val)
        ),)

    s.pipeline.driver.transformer = fake_transformer
    s.encode_prompt = lambda prompt: _teo(marker=0.5)  # the uncond branch

    noise = torch.zeros(1, 2, 4, 4, 4)
    sigmas_one_step = torch.tensor([1.0, 0.0])
    s._build_sigmas = lambda n: sigmas_one_step

    out = s.denoise(noise, _teo(marker=2.0), num_steps=1, guidance_scale=3.0, seed=0)
    # v = 0.5 + 3.0 * (2.0 - 0.5) = 5.0; dt = -1 → x = -5.0 everywhere.
    assert torch.allclose(out, torch.full_like(out, -5.0), atol=1e-5)


def test_cfg_off_single_forward():
    s = _sampler()
    calls = []

    def fake_transformer(**kwargs):
        calls.append(1)
        return (torch.zeros(*kwargs["hidden_states"].shape[:-1], 4),)

    s.pipeline.driver.transformer = fake_transformer
    s._build_sigmas = lambda n: torch.tensor([1.0, 0.0])
    s.denoise(torch.zeros(1, 2, 4, 4, 4), _teo(), 1, guidance_scale=1.0, seed=0)
    assert len(calls) == 1  # no uncond forward at gs <= 1


def test_default_negative_prompt_injected_when_unset():
    s = _sampler(config={})  # no sample_negative_prompt
    requested: list[str] = []

    def spy_encode(prompt: str) -> TextEncoderOutput:
        requested.append(prompt)
        return _teo(marker=0.0)

    s.encode_prompt = spy_encode
    s.pipeline.driver.transformer = lambda **kw: (
        torch.zeros(*kw["hidden_states"].shape[:-1], 4),
    )
    s._build_sigmas = lambda n: torch.tensor([1.0, 0.0])
    s.denoise(torch.zeros(1, 2, 4, 4, 4), _teo(), 1, guidance_scale=5.0, seed=0)
    assert requested == [KANDINSKY5_DEFAULT_NEGATIVE_PROMPT]


def test_configured_negative_prompt_wins():
    s = _sampler(config={"sample_negative_prompt": "blurry"})
    requested: list[str] = []
    s.encode_prompt = lambda p: (requested.append(p), _teo(marker=0.0))[1]
    s.pipeline.driver.transformer = lambda **kw: (
        torch.zeros(*kw["hidden_states"].shape[:-1], 4),
    )
    s._build_sigmas = lambda n: torch.tensor([1.0, 0.0])
    s.denoise(torch.zeros(1, 2, 4, 4, 4), _teo(), 1, guidance_scale=5.0, seed=0)
    assert requested == ["blurry"]


# ── 3. Channels-last step slice ────────────────────────────────────────────


def test_step_updates_only_latent_channels_of_cond_concat_state():
    s = _sampler()
    x = torch.zeros(1, 2, 4, 4, 9)  # C=4 latents + 4 cond + 1 mask
    x[..., 4:8] = 7.0  # cond content
    x[..., 8] = 1.0  # mask content
    v = torch.ones(1, 2, 4, 4, 4)

    out = s.euler_integrate(
        x, torch.tensor([1.0, 0.0]), lambda xf, sg: v, num_channels=4
    )
    assert torch.allclose(out[..., :4], torch.full_like(out[..., :4], -1.0))
    assert torch.all(out[..., 4:8] == 7.0)  # cond untouched
    assert torch.all(out[..., 8] == 1.0)  # mask untouched


def test_denoise_state_carries_cond_concat_but_returns_c_channels():
    s = _sampler()
    seen_shapes = []

    def fake_transformer(**kwargs):
        seen_shapes.append(tuple(kwargs["hidden_states"].shape))
        return (torch.zeros(*kwargs["hidden_states"].shape[:-1], 4),)

    s.pipeline.driver.transformer = fake_transformer
    s._build_sigmas = lambda n: torch.tensor([1.0, 0.0])
    out = s.denoise(torch.zeros(1, 2, 4, 4, 4), _teo(), 1, 1.0, 0)
    assert seen_shapes == [(1, 2, 4, 4, 9)]  # visual_cond concat fed forward
    assert out.shape == (1, 2, 4, 4, 4)  # concat stripped for decode


# ── 4. I2V frame-0 skip ────────────────────────────────────────────────────


def test_i2v_image_latent_pins_frame0_through_denoise():
    s = _sampler()
    image_latent = torch.full((1, 1, 4, 4, 4), 3.0)
    s._conditioning_image_latent = lambda: image_latent

    def fake_transformer(**kwargs):
        hs = kwargs["hidden_states"]
        # cond stream carries the image in frame 0, mask=1 there.
        assert torch.all(hs[:, :1, :, :, 4:8] == 3.0)
        assert torch.all(hs[:, :1, :, :, 8] == 1.0)
        assert torch.all(hs[:, 1:, :, :, 8] == 0.0)
        return (torch.ones(*hs.shape[:-1], 4),)

    s.pipeline.driver.transformer = fake_transformer
    s._build_sigmas = lambda n: torch.tensor([1.0, 0.0])
    out = s.denoise(torch.zeros(1, 3, 4, 4, 4), _teo(), 1, 1.0, 0)
    assert torch.all(out[:, :1] == 3.0)  # frame 0 == image latent, never stepped
    assert torch.allclose(out[:, 1:], torch.full_like(out[:, 1:], -1.0))


# ── Noise / sigmas / decode ────────────────────────────────────────────────


def test_initial_noise_is_channels_last_5d():
    s = _sampler(config={"sample_num_frames": 17})
    gen = torch.Generator().manual_seed(0)
    noise = s._create_initial_noise(768, 512, gen)
    # 17 frames → (17-1)/4+1 = 5 latent frames; 512x768 px → 64x96 latents.
    assert noise.shape == (1, 5, 64, 96, 4)
    assert noise.dtype == torch.float32


def test_sigmas_use_repo_shift_5():
    s = _sampler()
    sigmas = s._build_sigmas(8)
    assert len(sigmas) == 9
    assert sigmas[0].item() == 1.0 and sigmas[-1].item() == 0.0
    # shift 5: sigma'(0.5) = 5*0.5 / (1 + 4*0.5) = 0.8333…
    mid = sigmas[4].item()
    assert abs(mid - 5 * 0.5 / (1 + 4 * 0.5)) < 1e-6


def test_model_shift_fixed_config_overrides_shift():
    s = _sampler(config={"model_shift_fixed": 1.0})
    sigmas = s._build_sigmas(8)
    assert abs(sigmas[4].item() - 0.5) < 1e-6  # shift 1 → linear


def test_decode_divides_by_scaling_factor_and_returns_artifact():
    s = _sampler()
    captured = {}

    class _FakeVAE:
        dtype = torch.float32
        config = SimpleNamespace(scaling_factor=0.476986)

        def decode(self, z, return_dict=False):
            captured["z"] = z
            b, c, f, h, w = z.shape
            return (torch.zeros(b, 3, (f - 1) * 4 + 1, h * 8, w * 8),)

    s.pipeline.driver.vae = _FakeVAE()
    latents = torch.full((1, 2, 4, 4, 4), 0.476986)
    artifact = s.decode_latents(latents)

    z = captured["z"]
    assert z.shape == (1, 4, 2, 4, 4)  # channels-first for the VAE
    assert torch.allclose(z, torch.ones_like(z))  # ÷ 0.476986 applied
    assert isinstance(artifact, SampleArtifact)
    assert artifact.fps == 24.0
    assert artifact.frames.shape == (3, 5, 32, 32)  # [C, F, H, W]


# ── End-to-end on the REAL tiny transformer ────────────────────────────────


def test_denoise_end_to_end_real_tiny_transformer():
    from diffusers import Kandinsky5Transformer3DModel

    model = Kandinsky5Transformer3DModel(**tiny_transformer_config(visual_cond=True))
    model.eval()
    s = _sampler()
    s.pipeline.driver.transformer = model
    s.encode_prompt = lambda p: _teo()  # uncond branch for CFG

    noise = torch.randn(1, 2, 8, 8, 4)
    out = s.denoise(noise, _teo(), num_steps=2, guidance_scale=5.0, seed=0)
    assert out.shape == (1, 2, 8, 8, 4)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
