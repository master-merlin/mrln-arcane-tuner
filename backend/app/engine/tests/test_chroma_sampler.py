"""Tests for ChromaSampler — precision contract + CFG + scheduler math.

Correctness invariants (mirroring test_ovis_image_sampler.py):
  1. encode_prompt returns dict with 3-D embeds + mask via trainer.encode_text
  2. denoise: fp32 trajectory, raw [0,1000] timesteps into driver.forward_pass
  3. CFG is REAL per ChromaPipeline: gate ``guidance_scale > 1``; combine
     ``neg + g * (pos - neg)``
  4. scheduler is built from architecture_params (real diffusers class),
     correctly differentiating chroma1-hd's static-shift config from
     chroma1-base's beta-sigma config
  5. NO torch.autocast around the DiT forward (autocast-collapse gotcha)
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest
import torch


_TINY_CFG = dict(
    patch_size=1,
    in_channels=64,
    out_channels=64,
    num_layers=1,
    num_single_layers=1,
    attention_head_dim=8,
    num_attention_heads=2,
    joint_attention_dim=16,
    axes_dims_rope=(2, 4, 2),
    approximator_num_channels=8,
    approximator_hidden_dim=16,
    approximator_layers=1,
)

_W, _H = 32, 32       # pixel dims -> lat 4x4 (vae_sf=8)
_VAE_SF = 8
_LAT = 2 * (_H // (_VAE_SF * 2))   # = 4
_TXT_SEQ = 7
_TEXT_DIM = 16        # == joint_attention_dim of the tiny model

_ARCH_HD = {
    "scheduler.num_train_timesteps": 1000,
    "scheduler.shift": 3.0,
    "scheduler.use_dynamic_shifting": False,
    "scheduler.base_shift": 0.5,
    "scheduler.max_shift": 1.15,
    "scheduler.base_image_seq_len": 256,
    "scheduler.max_image_seq_len": 4096,
    "scheduler.use_beta_sigmas": False,
    "vae.latent_channels": 16,
    "vae.vae_scale_factor": 8,
}

_ARCH_BASE = {
    "scheduler.num_train_timesteps": 1000,
    "scheduler.shift": 1.0,
    "scheduler.use_dynamic_shifting": False,
    "scheduler.base_shift": 0.5,
    "scheduler.max_shift": 1.15,
    "scheduler.base_image_seq_len": 256,
    "scheduler.max_image_seq_len": 4096,
    "scheduler.use_beta_sigmas": True,
    "vae.latent_channels": 16,
    "vae.vae_scale_factor": 8,
}


def _build_tiny_model():
    from diffusers.models.transformers.transformer_chroma import (
        ChromaTransformer2DModel,
    )

    torch.manual_seed(0)
    return ChromaTransformer2DModel(**_TINY_CFG).eval()


def _build_mock_vae():
    vae = MagicMock()
    vae.dtype = torch.float32
    vae.parameters = lambda: iter([torch.zeros(1)])
    vae.config = MagicMock()
    vae.config.scaling_factor = 0.3611
    vae.config.shift_factor = 0.1159

    def _decode(latents, return_dict=False):
        b = latents.shape[0]
        return (torch.zeros(b, 3, _H, _W),)

    vae.decode = _decode
    vae.to = lambda *a, **k: vae
    return vae


def _build_mock_pipeline(model, arch=None):
    from app.engine.models.families.chroma.driver import ChromaDriver

    drv_defn = MagicMock()
    drv_defn.family = "chroma"
    drv_defn.id = "chroma-test"
    drv_defn.lora_targetable_modules = []
    drv_defn.architecture_params = dict(arch or _ARCH_HD)

    driver = ChromaDriver(drv_defn, torch.device("cpu"))
    driver.assign_components({
        "unet": model, "vae": None, "text_encoder": None, "tokenizer": None,
    })

    pipeline = MagicMock()
    pipeline.device = torch.device("cpu")
    pipeline.transformer = model
    pipeline.vae = _build_mock_vae()
    pipeline.driver = driver

    def _encode_text(prompts, dtype=None):
        b = len(prompts)
        torch.manual_seed(len(prompts[0]) if prompts[0] else 1)
        emb = torch.randn(b, _TXT_SEQ, _TEXT_DIM)
        mask = torch.ones(b, _TXT_SEQ)
        return emb, mask

    pipeline.encode_text = _encode_text

    defn = MagicMock()
    defn.architecture_params = dict(arch or _ARCH_HD)
    defn.defaults = {}
    pipeline.definition = defn

    pipeline.config = {"sample_every_n_steps": 50, "sample_negative_prompt": ""}
    pipeline._block_swap_managers = None
    return pipeline


def _build_sampler(arch=None):
    from app.engine.models.families.chroma.sampler import ChromaSampler

    model = _build_tiny_model()
    pipeline = _build_mock_pipeline(model, arch)
    return ChromaSampler(pipeline), model


# ── Step 1: encode_prompt ────────────────────────────────────────────────────


class TestEncodePrompt:
    def test_encode_prompt_returns_3d_embeds_and_mask(self):
        sampler, _ = _build_sampler()
        result = sampler.encode_prompt("a test prompt")

        assert isinstance(result, dict)
        assert "embeds" in result and "mask" in result
        assert result["embeds"].ndim == 3
        assert result["embeds"].shape[0] == 1
        assert result["mask"].ndim == 2

    def test_encode_prompt_delegates_to_pipeline(self):
        sampler, _ = _build_sampler()
        calls = []

        def _spy(prompts, dtype=None):
            calls.append(prompts)
            return (
                torch.randn(len(prompts), _TXT_SEQ, _TEXT_DIM),
                torch.ones(len(prompts), _TXT_SEQ),
            )

        sampler.pipeline.encode_text = _spy
        sampler.encode_prompt("hello")
        assert calls == [["hello"]]


# ── Step 2: scheduler construction (per-definition) ──────────────────────────


class TestScheduler:
    def test_hd_scheduler_is_static_shift_not_dynamic(self):
        from diffusers import FlowMatchEulerDiscreteScheduler

        sampler, _ = _build_sampler(_ARCH_HD)
        sched = sampler._get_scheduler()
        assert isinstance(sched, FlowMatchEulerDiscreteScheduler)
        assert sched.config.use_dynamic_shifting is False
        assert sched.config.shift == 3.0
        assert sched.config.use_beta_sigmas is False

    def test_base_scheduler_is_beta_sigma(self):
        from diffusers import FlowMatchEulerDiscreteScheduler

        sampler, _ = _build_sampler(_ARCH_BASE)
        sched = sampler._get_scheduler()
        assert isinstance(sched, FlowMatchEulerDiscreteScheduler)
        assert sched.config.use_dynamic_shifting is False
        assert sched.config.shift == 1.0
        assert sched.config.use_beta_sigmas is True

    def test_mu_matches_pipeline_calculate_shift(self):
        from diffusers.pipelines.chroma.pipeline_chroma import calculate_shift

        sampler, _ = _build_sampler()
        for seq_len in (4, 256, 1024, 4096):
            expected = calculate_shift(seq_len, 256, 4096, 0.5, 1.15)
            assert sampler._compute_mu(seq_len) == pytest.approx(expected)


# ── Step 3: denoise — shapes, CFG gating, fp32 trajectory ────────────────────


class TestDenoise:
    @pytest.mark.parametrize("arch", [_ARCH_HD, _ARCH_BASE])
    def test_denoise_shape_fp32_and_finite_cfg(self, arch):
        sampler, _ = _build_sampler(arch)
        gen = torch.Generator().manual_seed(42)
        noise = sampler._create_initial_noise(_W, _H, gen)
        assert noise.shape == (1, 16, _LAT, _LAT)

        prompt_emb = sampler.encode_prompt("a test image")
        latents = sampler.denoise(
            noise=noise, prompt_embedding=prompt_emb,
            num_steps=2, guidance_scale=5.0, seed=42,
        )

        assert latents.shape == (1, 16, _LAT, _LAT)
        assert latents.dtype == torch.float32, "trajectory must stay fp32"
        assert latents.isfinite().all()
        assert latents.float().std() > 0

    def test_cfg_double_forward_when_gs_above_1(self):
        sampler, model = _build_sampler()
        forward_calls = []
        original = model.forward

        def _spy(*args, **kwargs):
            forward_calls.append(1)
            return original(*args, **kwargs)

        model.forward = _spy
        gen = torch.Generator().manual_seed(1)
        noise = sampler._create_initial_noise(_W, _H, gen)
        sampler.denoise(
            noise=noise, prompt_embedding=sampler.encode_prompt("test cfg"),
            num_steps=2, guidance_scale=5.0, seed=1,
        )
        assert len(forward_calls) == 4, (
            f"CFG must run cond+uncond per step (4 total), got {len(forward_calls)}"
        )

    @pytest.mark.parametrize("gs", [1.0, 0.0])
    def test_no_cfg_single_forward_when_gs_at_or_below_1(self, gs):
        """ChromaPipeline gates CFG at guidance_scale > 1."""
        sampler, model = _build_sampler()
        forward_calls = []
        original = model.forward

        def _spy(*args, **kwargs):
            forward_calls.append(1)
            return original(*args, **kwargs)

        model.forward = _spy
        gen = torch.Generator().manual_seed(2)
        noise = sampler._create_initial_noise(_W, _H, gen)
        latents = sampler.denoise(
            noise=noise, prompt_embedding=sampler.encode_prompt("test"),
            num_steps=2, guidance_scale=gs, seed=2,
        )
        assert latents.isfinite().all()
        assert len(forward_calls) == 2, (
            f"gs={gs} must run a single cond pass per step, got {len(forward_calls)}"
        )

    def test_cfg_combine_matches_pipeline_convention(self):
        """velocity = neg + g * (pos - neg) — ChromaPipeline's exact formula."""
        from app.engine.models.families.chroma.sampler import _combine_cfg

        pos = torch.ones(1, 64, 2, 2)
        neg = torch.zeros(1, 64, 2, 2)
        out = _combine_cfg(pos, neg, 5.0)
        assert torch.allclose(out, torch.full_like(pos, 5.0))

        pos2 = torch.full((1, 4), 2.0)
        neg2 = torch.full((1, 4), 1.0)
        assert torch.allclose(_combine_cfg(pos2, neg2, 3.0), torch.full_like(pos2, 4.0))


# ── Step 4: decode + trainer wiring ──────────────────────────────────────────


class TestDecodeAndTrainerWiring:
    def test_decode_latents_returns_pil(self):
        from PIL import Image

        sampler, _ = _build_sampler()
        latents = torch.randn(1, 16, _LAT, _LAT)
        result = sampler.decode_latents(latents)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (_W, _H)

    def test_create_sampler_returns_chroma_sampler_when_enabled(self):
        from app.engine.models.families.chroma.sampler import ChromaSampler
        from app.engine.models.families.chroma.trainer import ChromaTrainer

        trainer = MagicMock(spec=ChromaTrainer)
        trainer.config = {"sample_every_n_steps": 50}
        trainer.device = torch.device("cpu")
        trainer.definition = MagicMock()

        sampler = ChromaTrainer._create_sampler(trainer)
        assert isinstance(sampler, ChromaSampler)

    def test_create_sampler_returns_none_when_disabled(self):
        from app.engine.models.families.chroma.trainer import ChromaTrainer

        trainer = MagicMock(spec=ChromaTrainer)
        trainer.config = {"sample_every_n_steps": 0}
        trainer.device = torch.device("cpu")
        trainer.definition = MagicMock()

        assert ChromaTrainer._create_sampler(trainer) is None


# ── Step 5: precision contract ───────────────────────────────────────────────


class TestPrecisionContract:
    def test_no_autocast_in_denoise_source(self):
        from app.engine.models.families.chroma.sampler import ChromaSampler

        source = inspect.getsource(ChromaSampler.denoise)
        non_comment = "\n".join(
            line for line in source.splitlines()
            if not line.strip().startswith("#")
        )
        assert "torch.autocast" not in non_comment, (
            "denoise must NOT use torch.autocast (autocast-collapse gotcha)"
        )

    def test_multistep_run_stays_non_degenerate(self):
        sampler, _ = _build_sampler()
        gen = torch.Generator().manual_seed(7)
        noise = sampler._create_initial_noise(_W, _H, gen)
        latents = sampler.denoise(
            noise=noise, prompt_embedding=sampler.encode_prompt("a precision test"),
            num_steps=4, guidance_scale=5.0, seed=7,
        )
        assert latents.isfinite().all()
        assert latents.float().std() > 0
        assert latents.dtype == torch.float32

    def test_driver_receives_raw_thousand_scale_timesteps(self):
        """The sampler hands RAW [0,1000] timesteps to driver.forward_pass
        (the driver divides by 1000 — never divide twice)."""
        sampler, _ = _build_sampler()
        seen_ts = []
        real_forward = sampler.pipeline.driver.forward_pass

        def _spy(noisy_input, timesteps, text_embeddings, batch):
            seen_ts.append(float(timesteps.flatten()[0]))
            return real_forward(
                noisy_input=noisy_input, timesteps=timesteps,
                text_embeddings=text_embeddings, batch=batch,
            )

        sampler.pipeline.driver = MagicMock(wraps=sampler.pipeline.driver)
        sampler.pipeline.driver.forward_pass = _spy

        gen = torch.Generator().manual_seed(3)
        noise = sampler._create_initial_noise(_W, _H, gen)
        sampler.denoise(
            noise=noise, prompt_embedding=sampler.encode_prompt("ts scale"),
            num_steps=2, guidance_scale=0.0, seed=3,
        )
        assert seen_ts, "driver.forward_pass never called"
        assert seen_ts[0] > 500.0, (
            f"first timestep should be near 1000 (raw scale), got {seen_ts[0]}"
        )


# -- Native sample defaults (pipeline __call__: 35 steps, guidance 5.0) -------


class TestNativeSampleDefaults:
    def test_sample_single_fills_chroma_native_defaults(self, monkeypatch):
        from app.engine.core.sampling import GenericSamplingPipeline

        sampler, _ = _build_sampler()
        captured: dict = {}

        def _fake_base(self, cfg, step):
            captured.update(cfg)
            return MagicMock()

        monkeypatch.setattr(GenericSamplingPipeline, "_sample_single", _fake_base)

        sampler._sample_single({"prompt": "defaults"}, 0)
        assert captured["num_inference_steps"] == 35
        assert captured["guidance_scale"] == 5.0
        assert captured["width"] == 1024
        assert captured["height"] == 1024

    def test_sample_single_respects_explicit_values(self, monkeypatch):
        from app.engine.core.sampling import GenericSamplingPipeline

        sampler, _ = _build_sampler()
        captured: dict = {}

        def _fake_base(self, cfg, step):
            captured.update(cfg)
            return MagicMock()

        monkeypatch.setattr(GenericSamplingPipeline, "_sample_single", _fake_base)

        sampler._sample_single(
            {"prompt": "explicit", "width": 768, "height": 768,
             "num_inference_steps": 12, "guidance_scale": 1.0},
            0,
        )
        assert captured["num_inference_steps"] == 12
        assert captured["guidance_scale"] == 1.0
        assert captured["width"] == 768

    def test_sample_single_sources_from_definition_not_constant(self, monkeypatch):
        from app.engine.core.sampling import GenericSamplingPipeline

        sampler, _ = _build_sampler()
        sampler.pipeline.definition.defaults = {
            "num_inference_steps": 33, "guidance_scale": 2.2, "resolution": 512,
        }
        captured: dict = {}

        def _fake_base(self, cfg, step):
            captured.update(cfg)
            return MagicMock()

        monkeypatch.setattr(GenericSamplingPipeline, "_sample_single", _fake_base)

        sampler._sample_single({"prompt": "sourced"}, 0)
        assert captured["num_inference_steps"] == 33
        assert captured["guidance_scale"] == 2.2
        assert captured["width"] == 512
        assert captured["height"] == 512

    @pytest.mark.parametrize("fname", ["chroma1_base.yaml", "chroma1_hd.yaml"])
    def test_shipped_yaml_carries_native_sample_defaults(self, fname):
        """Both shipped YAMLs are the source of truth for the native preview
        defaults (35 steps / 5.0 guidance)."""
        import pathlib

        import yaml

        base = (
            pathlib.Path(__file__).resolve().parents[1]
            / "models" / "families" / "chroma" / "definitions" / fname
        )
        defaults = yaml.safe_load(base.read_text(encoding="utf-8"))["defaults"]
        assert defaults["num_inference_steps"] == 35
        assert defaults["guidance_scale"] == 5.0
        assert defaults["resolution"] == 1024
