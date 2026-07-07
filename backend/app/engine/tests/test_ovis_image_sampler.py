"""Tests for OvisImageSampler — precision contract + CFG + mu/shift math.

Correctness invariants (mirroring test_krea2_sampler.py):
  1. encode_prompt returns dict with 3-D embeds + mask via trainer.encode_text
  2. denoise: fp32 trajectory, raw [0,1000] timesteps into driver.forward_pass
  3. CFG per OvisImagePipeline: gate ``guidance_scale > 1``; combine
     ``neg + g * (pos - neg)``
  4. mu/shift replicate the pipeline: sigmas = linspace(1, 1/n, n),
     mu = calculate_shift(image_seq_len, 256, 4096, 0.5, 1.15)
  5. NO torch.autocast around the DiT forward (autocast-collapse gotcha)
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest
import torch


# ── Shared tiny transformer config (1+1 blocks, minimal dims) ────────────────

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
)

_W, _H = 32, 32       # pixel dims → lat 4×4 (vae_sf=8), img_seq = 2*2 = 4
_VAE_SF = 8
_LAT = 2 * (_H // (_VAE_SF * 2))   # = 4 (pipeline prepare_latents formula)
_TXT_SEQ = 7
_TEXT_DIM = 16        # == joint_attention_dim of the tiny model

# Ovis scheduler facts (checkpoint scheduler_config.json)
_ARCH = {
    "scheduler.num_train_timesteps": 1000,
    "scheduler.shift": 3.0,
    "scheduler.use_dynamic_shifting": True,
    "scheduler.base_shift": 0.5,
    "scheduler.max_shift": 1.15,
    "scheduler.base_image_seq_len": 256,
    "scheduler.max_image_seq_len": 4096,
    "vae.latent_channels": 16,
    "vae.vae_scale_factor": 8,
}


def _build_tiny_model():
    from diffusers.models.transformers.transformer_ovis_image import (
        OvisImageTransformer2DModel,
    )

    torch.manual_seed(0)
    return OvisImageTransformer2DModel(**_TINY_CFG).eval()


def _build_mock_vae():
    vae = MagicMock()
    vae.dtype = torch.float32

    def _params():
        return iter([torch.zeros(1)])

    vae.parameters = _params
    vae.config = MagicMock()
    vae.config.scaling_factor = 0.3611
    vae.config.shift_factor = 0.1159

    def _decode(latents, return_dict=False):
        b = latents.shape[0]
        out = torch.zeros(b, 3, _H, _W)
        return (out,)

    vae.decode = _decode
    vae.to = lambda *a, **k: vae
    return vae


def _build_mock_pipeline(model):
    """Mock OvisImageTrainer with a REAL OvisImageDriver + tiny model."""
    from app.engine.models.families.ovis_image.driver import OvisImageDriver

    drv_defn = MagicMock()
    drv_defn.family = "ovis_image"
    drv_defn.id = "ovis-image-test"
    drv_defn.lora_targetable_modules = []
    drv_defn.architecture_params = dict(_ARCH)

    driver = OvisImageDriver(drv_defn, torch.device("cpu"))
    driver.assign_components({
        "unet": model,
        "vae": None,
        "text_encoder": None,
        "tokenizer": None,
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
        mask = torch.ones(b, _TXT_SEQ, dtype=torch.long)
        return emb, mask

    pipeline.encode_text = _encode_text

    defn = MagicMock()
    defn.architecture_params = dict(_ARCH)
    defn.defaults = {"is_distilled": False}
    pipeline.definition = defn

    pipeline.config = {
        "sample_every_n_steps": 50,
        "sample_negative_prompt": "",
    }
    pipeline._block_swap_managers = None
    return pipeline


def _build_sampler():
    from app.engine.models.families.ovis_image.sampler import OvisImageSampler

    model = _build_tiny_model()
    pipeline = _build_mock_pipeline(model)
    return OvisImageSampler(pipeline), model


# ── Step 1: encode_prompt ────────────────────────────────────────────────────


class TestEncodePrompt:
    def test_encode_prompt_returns_3d_embeds_and_mask(self):
        sampler, _ = _build_sampler()
        result = sampler.encode_prompt("a test prompt")

        assert isinstance(result, dict)
        assert "embeds" in result and "mask" in result
        assert result["embeds"].ndim == 3, "embeds must be [B, L, D]"
        assert result["embeds"].shape[0] == 1
        assert result["mask"].ndim == 2

    def test_encode_prompt_delegates_to_pipeline(self):
        sampler, _ = _build_sampler()
        calls = []

        def _spy(prompts, dtype=None):
            calls.append(prompts)
            return (
                torch.randn(len(prompts), _TXT_SEQ, _TEXT_DIM),
                torch.ones(len(prompts), _TXT_SEQ, dtype=torch.long),
            )

        sampler.pipeline.encode_text = _spy
        sampler.encode_prompt("hello")
        assert calls == [["hello"]]


# ── Step 2: scheduler + mu math ──────────────────────────────────────────────


class TestSchedulerAndMu:
    def test_scheduler_matches_checkpoint_config(self):
        from diffusers import FlowMatchEulerDiscreteScheduler

        sampler, _ = _build_sampler()
        sched = sampler._get_scheduler()
        assert isinstance(sched, FlowMatchEulerDiscreteScheduler)
        assert sched.config.num_train_timesteps == 1000
        assert sched.config.shift == 3.0
        assert sched.config.use_dynamic_shifting is True
        assert sched.config.base_shift == 0.5
        assert sched.config.max_shift == 1.15
        assert sched.config.base_image_seq_len == 256
        assert sched.config.max_image_seq_len == 4096

    def test_mu_matches_pipeline_calculate_shift(self):
        """_compute_mu replicates diffusers calculate_shift for Ovis."""
        from diffusers.pipelines.ovis_image.pipeline_ovis_image import (
            calculate_shift,
        )

        sampler, _ = _build_sampler()
        for seq_len in (4, 256, 1024, 4096):
            expected = calculate_shift(seq_len, 256, 4096, 0.5, 1.15)
            assert sampler._compute_mu(seq_len) == pytest.approx(expected), (
                f"mu mismatch at seq_len={seq_len}"
            )


# ── Step 3: denoise — shapes, CFG gating, fp32 trajectory ────────────────────


class TestDenoise:
    def test_denoise_shape_fp32_and_finite_cfg(self):
        sampler, _ = _build_sampler()
        gen = torch.Generator().manual_seed(42)
        noise = sampler._create_initial_noise(_W, _H, gen)
        assert noise.shape == (1, 16, _LAT, _LAT)

        prompt_emb = sampler.encode_prompt("a test image")
        latents = sampler.denoise(
            noise=noise,
            prompt_embedding=prompt_emb,
            num_steps=2,
            guidance_scale=5.0,
            seed=42,
        )

        assert latents.shape == (1, 16, _LAT, _LAT)
        assert latents.dtype == torch.float32, "trajectory must stay fp32"
        assert latents.isfinite().all(), "denoise output contains NaN or inf"
        assert latents.float().std() > 0, "denoise output is degenerate"

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
            noise=noise,
            prompt_embedding=sampler.encode_prompt("test cfg"),
            num_steps=2,
            guidance_scale=5.0,
            seed=1,
        )
        assert len(forward_calls) == 4, (
            f"CFG must run cond+uncond per step (4 total), got {len(forward_calls)}"
        )

    @pytest.mark.parametrize("gs", [1.0, 0.0])
    def test_no_cfg_single_forward_when_gs_at_or_below_1(self, gs):
        """OvisImagePipeline gates CFG at guidance_scale > 1."""
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
            noise=noise,
            prompt_embedding=sampler.encode_prompt("test"),
            num_steps=2,
            guidance_scale=gs,
            seed=2,
        )
        assert latents.isfinite().all()
        assert len(forward_calls) == 2, (
            f"gs={gs} must run a single cond pass per step, got {len(forward_calls)}"
        )

    def test_cfg_combine_matches_pipeline_convention(self):
        """velocity = neg + g * (pos - neg) — the OvisImagePipeline formula."""
        from app.engine.models.families.ovis_image.sampler import _combine_cfg

        pos = torch.ones(1, 16, 2, 2)
        neg = torch.zeros(1, 16, 2, 2)
        out = _combine_cfg(pos, neg, 5.0)
        assert torch.allclose(out, torch.full_like(pos, 5.0))

        pos2 = torch.full((1, 4), 2.0)
        neg2 = torch.full((1, 4), 1.0)
        # 1 + 3*(2-1) = 4
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

    def test_create_sampler_returns_ovis_sampler_when_enabled(self):
        from app.engine.models.families.ovis_image.sampler import OvisImageSampler
        from app.engine.models.families.ovis_image.trainer import OvisImageTrainer

        trainer = MagicMock(spec=OvisImageTrainer)
        trainer.config = {"sample_every_n_steps": 50}
        trainer.device = torch.device("cpu")
        trainer.definition = MagicMock()

        sampler = OvisImageTrainer._create_sampler(trainer)
        assert isinstance(sampler, OvisImageSampler)

    def test_create_sampler_returns_none_when_disabled(self):
        from app.engine.models.families.ovis_image.trainer import OvisImageTrainer

        trainer = MagicMock(spec=OvisImageTrainer)
        trainer.config = {"sample_every_n_steps": 0}
        trainer.device = torch.device("cpu")
        trainer.definition = MagicMock()

        assert OvisImageTrainer._create_sampler(trainer) is None


# ── Step 5: precision contract ───────────────────────────────────────────────


class TestPrecisionContract:
    def test_no_autocast_in_denoise_source(self):
        from app.engine.models.families.ovis_image.sampler import OvisImageSampler

        source = inspect.getsource(OvisImageSampler.denoise)
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
            noise=noise,
            prompt_embedding=sampler.encode_prompt("a precision test"),
            num_steps=4,
            guidance_scale=5.0,
            seed=7,
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
                noisy_input=noisy_input,
                timesteps=timesteps,
                text_embeddings=text_embeddings,
                batch=batch,
            )

        sampler.pipeline.driver = MagicMock(wraps=sampler.pipeline.driver)
        sampler.pipeline.driver.forward_pass = _spy

        gen = torch.Generator().manual_seed(3)
        noise = sampler._create_initial_noise(_W, _H, gen)
        sampler.denoise(
            noise=noise,
            prompt_embedding=sampler.encode_prompt("ts scale"),
            num_steps=2,
            guidance_scale=0.0,
            seed=3,
        )
        assert seen_ts, "driver.forward_pass never called"
        # First sigma is 1.0 → timestep 1000 on the [0,1000] scale.
        assert seen_ts[0] > 500.0, (
            f"first timestep should be near 1000 (raw scale), got {seen_ts[0]}"
        )
