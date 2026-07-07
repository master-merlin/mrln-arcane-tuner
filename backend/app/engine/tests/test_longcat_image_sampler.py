"""Tests for LongCatImageSampler — precision contract + CFG + mu/shift.

Mirrors test_krea2_sampler.py with LongCat-Image semantics:
  1. encode_prompt returns dict with [B, L, D] embeds + mask
  2. denoise: shape/finiteness, CFG double-forward, no-CFG single-forward
  3. mu/shift math matching LongCatImagePipeline.calculate_shift exactly
  4. CFG renorm (enable_cfg_renorm default) clamps the combined prediction
  5. Precision contract: fp32 latent trajectory, NO torch.autocast
  6. decode_latents → PIL; trainer._create_sampler gating
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import torch


# ── Shared tiny transformer config (matches test_longcat_image_family.py) ───

_TINY_CFG = dict(
    patch_size=1,
    in_channels=4,   # → 1 latent channel × 2×2 packing
    num_layers=1,
    num_single_layers=1,
    attention_head_dim=8,
    num_attention_heads=2,
    joint_attention_dim=16,
    pooled_projection_dim=16,
    axes_dims_rope=[4, 2, 2],
)

_W, _H = 16, 16       # pixel dims → lat 2×2 → packed seq len 1
_VAE_SF = 8
_TXT_SEQ = 7
_TEXT_DIM = 16


def _build_tiny_model():
    from diffusers.models.transformers.transformer_longcat_image import (
        LongCatImageTransformer2DModel,
    )

    return LongCatImageTransformer2DModel(**_TINY_CFG).eval()


def _build_mock_vae():
    """Mock 16→3 VAE with standard AutoencoderKL config surface."""
    vae = MagicMock()
    vae.dtype = torch.float32
    vae.config = MagicMock()
    vae.config.block_out_channels = [1, 2, 4, 4]  # 2^(4-1) = 8× scale
    vae.config.scaling_factor = 0.3611
    vae.config.shift_factor = 0.1159
    # No latents_mean/latents_std on a standard AutoencoderKL
    vae.config.latents_mean = None
    vae.config.latents_std = None

    def _decode(latents, return_dict=False):
        B = latents.shape[0]
        out = torch.zeros(B, 3, _H, _W)
        if return_dict:
            result = MagicMock()
            result.sample = out
            return result
        return (out,)

    vae.decode = _decode
    return vae


def _build_mock_pipeline(model):
    """Mock LongCatImageTrainer-like pipeline with the tiny model wired up."""
    pipeline = MagicMock()
    pipeline.device = torch.device("cpu")
    pipeline.transformer = model
    pipeline.model = model
    pipeline.vae = _build_mock_vae()
    pipeline._block_swap_managers = None

    def _encode_text(prompts, dtype=None):
        B = len(prompts)
        emb = torch.randn(B, _TXT_SEQ, _TEXT_DIM)
        mask = torch.ones(B, _TXT_SEQ, dtype=torch.long)
        return emb, mask

    pipeline.encode_text = _encode_text

    defn = MagicMock()
    defn.architecture_params = {"te.max_length": 512}
    defn.defaults = {}
    pipeline.definition = defn

    pipeline.config = {
        "sample_every_n_steps": 50,
        "sample_negative_prompt": "",
    }
    return pipeline


def _build_sampler():
    from app.engine.models.families.longcat_image.sampler import LongCatImageSampler

    model = _build_tiny_model()
    pipeline = _build_mock_pipeline(model)
    return LongCatImageSampler(pipeline), model


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — encode_prompt
# ─────────────────────────────────────────────────────────────────────────────


class TestEncodePrompt:

    def test_encode_prompt_returns_embeds_and_mask(self):
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


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — mu/shift math (pipeline calculate_shift, verified constants)
# ─────────────────────────────────────────────────────────────────────────────


class TestMuShift:

    def test_mu_matches_pipeline_calculate_shift(self):
        """mu is linear in image_seq_len: 256 → 0.5 and 4096 → 1.15
        (LongCatImagePipeline.calculate_shift defaults)."""
        sampler, _ = _build_sampler()
        assert abs(sampler._compute_mu(256) - 0.5) < 1e-9
        assert abs(sampler._compute_mu(4096) - 1.15) < 1e-9
        # midpoint check: linear interpolation
        m = (1.15 - 0.5) / (4096 - 256)
        expected = 1024 * m + (0.5 - m * 256)
        assert abs(sampler._compute_mu(1024) - expected) < 1e-9

    def test_scheduler_uses_dynamic_shifting(self):
        sampler, _ = _build_sampler()
        scheduler = sampler._get_scheduler()
        assert scheduler.config.use_dynamic_shifting is True
        assert scheduler.config.num_train_timesteps == 1000


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — denoise: shape + CFG branches
# ─────────────────────────────────────────────────────────────────────────────


class TestDenoise:

    def test_denoise_shape_and_finite_cfg(self):
        sampler, _ = _build_sampler()
        gen = torch.Generator().manual_seed(42)
        noise = sampler._create_initial_noise(_W, _H, gen)

        prompt_emb = sampler.encode_prompt("a test image")
        result = sampler.denoise(
            noise=noise,
            prompt_embedding=prompt_emb,
            num_steps=2,
            guidance_scale=4.5,
            seed=42,
        )

        assert isinstance(result, dict) and "latents" in result
        latents = result["latents"]
        # Unpacked 4D latents ready for VAE decode
        assert latents.shape == (1, 1, 2, 2), f"unexpected shape {latents.shape}"
        assert latents.isfinite().all()
        assert latents.float().std() > 0

    def test_cfg_double_forward_per_step(self):
        """guidance_scale > 1 → cond + uncond forward per step (pipeline
        do_classifier_free_guidance = guidance_scale > 1)."""
        sampler, model = _build_sampler()
        forward_calls = []
        original_forward = model.forward

        def _spy(*args, **kwargs):
            forward_calls.append(1)
            return original_forward(*args, **kwargs)

        model.forward = _spy
        gen = torch.Generator().manual_seed(1)
        noise = sampler._create_initial_noise(_W, _H, gen)
        sampler.denoise(
            noise=noise,
            prompt_embedding=sampler.encode_prompt("test cfg"),
            num_steps=2,
            guidance_scale=4.5,
            seed=1,
        )
        assert len(forward_calls) == 4, (
            f"CFG must run 2 forwards/step (4 total for 2 steps), got {len(forward_calls)}"
        )

    def test_no_cfg_single_forward_per_step(self):
        """guidance_scale <= 1 disables CFG → one forward per step."""
        sampler, model = _build_sampler()
        forward_calls = []
        original_forward = model.forward

        def _spy(*args, **kwargs):
            forward_calls.append(1)
            return original_forward(*args, **kwargs)

        model.forward = _spy
        gen = torch.Generator().manual_seed(0)
        noise = sampler._create_initial_noise(_W, _H, gen)
        result = sampler.denoise(
            noise=noise,
            prompt_embedding=sampler.encode_prompt("test"),
            num_steps=2,
            guidance_scale=1.0,
            seed=0,
        )
        assert result["latents"].isfinite().all()
        assert len(forward_calls) == 2, (
            f"no-CFG must run 1 forward/step, got {len(forward_calls)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — CFG renorm (pipeline enable_cfg_renorm=True default)
# ─────────────────────────────────────────────────────────────────────────────


class TestCfgRenorm:

    def test_renorm_clamps_to_cond_norm(self):
        """When the CFG-combined norm exceeds the cond norm, renorm scales it
        back down to exactly the cond norm (scale clamped to max 1.0)."""
        from app.engine.models.families.longcat_image.sampler import _cfg_renorm

        cond = torch.ones(1, 4, 8)
        pred = cond * 3.0  # combined norm 3× cond norm
        out = _cfg_renorm(pred, cond, cfg_renorm_min=0.0)
        assert torch.allclose(
            torch.norm(out, dim=-1), torch.norm(cond, dim=-1), atol=1e-4
        )

    def test_renorm_no_upscale_when_pred_smaller(self):
        """scale = cond_norm/noise_norm > 1 is clamped to 1.0 → unchanged."""
        from app.engine.models.families.longcat_image.sampler import _cfg_renorm

        cond = torch.ones(1, 4, 8) * 3.0
        pred = torch.ones(1, 4, 8)  # smaller than cond
        out = _cfg_renorm(pred, cond, cfg_renorm_min=0.0)
        assert torch.allclose(out, pred)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Precision contract
# ─────────────────────────────────────────────────────────────────────────────


class TestPrecisionContract:

    def test_no_autocast_in_denoise_source(self):
        """Denoise must NOT wrap the DiT forward in torch.autocast
        (autocast-collapse gotcha: N-step sampling degenerates to the
        conditional mean even though training works)."""
        from app.engine.models.families.longcat_image.sampler import (
            LongCatImageSampler,
        )

        source = inspect.getsource(LongCatImageSampler.denoise)
        non_comment = "\n".join(
            line for line in source.splitlines()
            if not line.strip().startswith("#")
        )
        assert "torch.autocast" not in non_comment, (
            "denoise must NOT use torch.autocast"
        )

    def test_latent_trajectory_stays_fp32(self):
        """The Euler trajectory accumulates in fp32 even for a bf16 model."""
        sampler, model = _build_sampler()
        model.to(torch.bfloat16)

        gen = torch.Generator().manual_seed(3)
        noise = sampler._create_initial_noise(_W, _H, gen)
        assert noise.dtype == torch.float32, "initial noise must be fp32"

        result = sampler.denoise(
            noise=noise,
            prompt_embedding=sampler.encode_prompt("precision"),
            num_steps=2,
            guidance_scale=4.5,
            seed=3,
        )
        assert result["latents"].dtype == torch.float32, (
            "latent trajectory must stay fp32"
        )

    def test_multistep_run_stays_non_degenerate(self):
        sampler, _ = _build_sampler()
        gen = torch.Generator().manual_seed(7)
        noise = sampler._create_initial_noise(_W, _H, gen)
        result = sampler.denoise(
            noise=noise,
            prompt_embedding=sampler.encode_prompt("a precision test"),
            num_steps=4,
            guidance_scale=4.5,
            seed=7,
        )
        latents = result["latents"]
        assert latents.isfinite().all()
        assert latents.float().std() > 0


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — decode_latents + trainer wiring
# ─────────────────────────────────────────────────────────────────────────────


class TestDecodeAndTrainerWiring:

    def test_decode_latents_returns_pil(self):
        from PIL import Image

        sampler, _ = _build_sampler()
        bundle = {
            "latents": torch.randn(1, 1, 2, 2),
            "height": _H,
            "width": _W,
        }
        result = sampler.decode_latents(bundle)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (_W, _H)

    def test_create_sampler_returns_sampler_when_enabled(self):
        from app.engine.models.families.longcat_image.trainer import (
            LongCatImageTrainer,
        )
        from app.engine.models.families.longcat_image.sampler import (
            LongCatImageSampler,
        )

        trainer = MagicMock(spec=LongCatImageTrainer)
        trainer.config = {"sample_every_n_steps": 50}
        trainer.device = torch.device("cpu")
        trainer.definition = MagicMock()

        sampler = LongCatImageTrainer._create_sampler(trainer)
        assert isinstance(sampler, LongCatImageSampler)

    def test_create_sampler_returns_none_when_disabled(self):
        from app.engine.models.families.longcat_image.trainer import (
            LongCatImageTrainer,
        )

        trainer = MagicMock(spec=LongCatImageTrainer)
        trainer.config = {"sample_every_n_steps": 0}
        trainer.device = torch.device("cpu")
        trainer.definition = MagicMock()

        assert LongCatImageTrainer._create_sampler(trainer) is None
