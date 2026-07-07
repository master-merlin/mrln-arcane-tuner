"""Tests for DreamLiteSampler — precision contract + CFG batching + mu math.

Correctness invariants (mirroring test_krea2_sampler.py / ovis):
  1. encode_prompt returns dict with 3-D embeds + mask via trainer.encode_text
     (positive → "[Generate]: " prefix inside the trainer); the CFG negative
     goes through trainer.encode_uncond_text (RAW).
  2. denoise: fp32 trajectory, RAW [0,1000] timesteps into
     driver.forward_pass (the UNet consumes raw timesteps — NO /1000).
  3. CFG per DreamLitePipeline (base): ONE batched forward per step with
     [uncond, cond] stacked (batch 2), combine
     ``uncond + guidance_scale * (cond - uncond)``. Mobile (distilled /
     guidance 0): single un-batched forward, no negative encode.
  4. mu/shift replicate the pipeline AT RUNTIME: the pipeline reads
     ``scheduler.config`` — the CHECKPOINT ships base_shift 0.5 /
     max_shift **1.15** / 256 / 4096 (NOT the calculate_shift signature
     default 1.16), with image_seq_len = lat_h * lat_w // 4.
  5. NO torch.autocast around the UNet forward; cached TE embeddings are
     cast to the MODEL dtype before the forward (fp32-cache vs bf16-model
     crash — Wave 1 lesson).
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest
import torch


# ── Shared tiny UNet config (structurally faithful — see family tests) ──────

_TINY_UNET_CFG = dict(
    in_channels=4,
    out_channels=4,
    block_out_channels=(8, 16, 32),
    layers_per_block=1,
    transformer_layers_per_block=(1, 2, 4),
    attention_head_dim=4,
    cross_attention_dim=16,
    norm_num_groups=8,
    use_linear_projection=True,
    encoder_hid_dim=12,
    encoder_hid_dim_type="text_proj_rms",
    addition_embed_type="time",
    addition_time_embed_dim=8,
    projection_class_embeddings_input_dim=16,
    num_kv_heads=1,
    qk_norm="rms_norm",
    ff_mult=3,
    use_sep_conv=True,
)

_W, _H = 64, 64        # pixel dims → lat 8×8 (vae_sf=8), img_seq = 64//4 = 16
_VAE_SF = 8
_LAT = _H // _VAE_SF   # = 8
_TXT_SEQ = 7
_TEXT_DIM = 12         # == encoder_hid_dim of the tiny model

# DreamLite scheduler facts (checkpoint scheduler_config.json)
_ARCH = {
    "scheduler.num_train_timesteps": 1000,
    "scheduler.shift": 3.0,
    "scheduler.use_dynamic_shifting": True,
    "scheduler.base_shift": 0.5,
    "scheduler.max_shift": 1.15,
    "scheduler.base_image_seq_len": 256,
    "scheduler.max_image_seq_len": 4096,
    "vae.latent_channels": 4,
    "vae.vae_scale_factor": 8,
    "te.max_sequence_length": _TXT_SEQ,
    "te.drop_idx": 34,
}


def _build_tiny_unet():
    from diffusers.models.unets.unet_dreamlite import DreamLiteUNetModel

    torch.manual_seed(0)
    return DreamLiteUNetModel(**_TINY_UNET_CFG).eval()


def _build_mock_vae():
    vae = MagicMock()
    vae.dtype = torch.float32

    def _params():
        return iter([torch.zeros(1)])

    vae.parameters = _params
    vae.config = MagicMock()
    vae.config.scaling_factor = 1.0
    vae.config.shift_factor = 0.0

    def _decode(latents, return_dict=False):
        b = latents.shape[0]
        return (torch.zeros(b, 3, _H, _W),)

    vae.decode = _decode
    vae.to = lambda *a, **k: vae
    return vae


def _build_mock_pipeline(model, *, is_distilled=False):
    """Mock DreamLiteTrainer with a REAL DreamLiteDriver + tiny UNet."""
    from app.engine.models.families.dreamlite.driver import DreamLiteDriver

    drv_defn = MagicMock()
    drv_defn.family = "dreamlite"
    drv_defn.id = "dreamlite-test"
    drv_defn.lora_targetable_modules = []
    drv_defn.architecture_params = dict(_ARCH)

    driver = DreamLiteDriver(drv_defn, torch.device("cpu"))
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
    pipeline.encode_uncond_text = lambda texts, dtype=None: _encode_text(
        texts, dtype,
    )

    defn = MagicMock()
    defn.architecture_params = dict(_ARCH)
    defn.defaults = {"is_distilled": is_distilled}
    pipeline.definition = defn

    pipeline.config = {
        "sample_every_n_steps": 50,
        "sample_negative_prompt": "",
    }
    pipeline._block_swap_managers = None
    return pipeline


def _build_sampler(*, is_distilled=False):
    from app.engine.models.families.dreamlite.sampler import DreamLiteSampler

    model = _build_tiny_unet()
    pipeline = _build_mock_pipeline(model, is_distilled=is_distilled)
    return DreamLiteSampler(pipeline), model


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

    def test_encode_prompt_delegates_to_pipeline_positive_path(self):
        """Positive prompts go through trainer.encode_text (which applies
        the [Generate] prefix internally)."""
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

    def test_mu_matches_pipeline_calculate_shift_with_checkpoint_config(self):
        """_compute_mu == diffusers calculate_shift fed the CHECKPOINT
        scheduler config (the pipeline reads scheduler.config at runtime —
        max_shift 1.15, NOT the function-signature default 1.16)."""
        from diffusers.pipelines.dreamlite.pipeline_dreamlite import (
            calculate_shift,
        )

        sampler, _ = _build_sampler()
        for seq_len in (16, 256, 1024, 4096):
            expected = calculate_shift(seq_len, 256, 4096, 0.5, 1.15)
            assert sampler._compute_mu(seq_len) == pytest.approx(expected), (
                f"mu mismatch at seq_len={seq_len}"
            )
        # 1024×1024 → latent 128×128 → seq (128*128)//4 = 4096 → mu = max_shift
        assert sampler._compute_mu(4096) == pytest.approx(1.15)

    def test_image_seq_len_is_quarter_of_latent_area(self):
        """The pipeline computes image_seq_len = lat_h * lat_w // 4."""
        sampler, _ = _build_sampler()
        mus = []
        real_compute = sampler._compute_mu

        def _spy(seq_len):
            mus.append(seq_len)
            return real_compute(seq_len)

        sampler._compute_mu = _spy
        gen = torch.Generator().manual_seed(0)
        noise = sampler._create_initial_noise(_W, _H, gen)
        sampler.denoise(
            noise=noise,
            prompt_embedding=sampler.encode_prompt("seq len test"),
            num_steps=1,
            guidance_scale=0.0,
            seed=0,
        )
        assert mus == [(_LAT * _LAT) // 4], f"wrong image_seq_len: {mus}"


# ── Step 3: denoise — shapes, CFG batching, fp32 trajectory ──────────────────


class TestDenoise:
    def test_denoise_shape_fp32_and_finite_cfg(self):
        sampler, _ = _build_sampler()
        gen = torch.Generator().manual_seed(42)
        noise = sampler._create_initial_noise(_W, _H, gen)
        assert noise.shape == (1, 4, _LAT, _LAT)

        prompt_emb = sampler.encode_prompt("a test image")
        latents = sampler.denoise(
            noise=noise,
            prompt_embedding=prompt_emb,
            num_steps=2,
            guidance_scale=3.5,
            seed=42,
        )

        assert latents.shape == (1, 4, _LAT, _LAT)
        assert latents.dtype == torch.float32, "trajectory must stay fp32"
        assert latents.isfinite().all(), "denoise output contains NaN or inf"
        assert latents.float().std() > 0, "denoise output is degenerate"

    def test_cfg_is_batched_single_forward_per_step(self):
        """Base pipeline convention: cond+uncond run in ONE batched UNet
        forward (batch 2) per step — NOT two sequential forwards."""
        sampler, model = _build_sampler()
        batch_sizes = []
        original = model.forward

        def _spy(sample, *args, **kwargs):
            batch_sizes.append(sample.shape[0])
            return original(sample, *args, **kwargs)

        model.forward = _spy
        gen = torch.Generator().manual_seed(1)
        noise = sampler._create_initial_noise(_W, _H, gen)
        sampler.denoise(
            noise=noise,
            prompt_embedding=sampler.encode_prompt("test cfg"),
            num_steps=2,
            guidance_scale=3.5,
            seed=1,
        )
        assert batch_sizes == [2, 2], (
            f"CFG must be ONE batched (B=2) forward per step, got {batch_sizes}"
        )

    def test_no_cfg_single_unbatched_forward_when_gs_zero(self):
        """guidance_scale 0 (mobile convention) → single B=1 pass per step,
        and the negative prompt is never encoded."""
        sampler, model = _build_sampler()
        uncond_calls = []
        sampler.pipeline.encode_uncond_text = lambda texts, dtype=None: (
            uncond_calls.append(texts)
            or (torch.randn(1, _TXT_SEQ, _TEXT_DIM),
                torch.ones(1, _TXT_SEQ, dtype=torch.long))
        )
        batch_sizes = []
        original = model.forward

        def _spy(sample, *args, **kwargs):
            batch_sizes.append(sample.shape[0])
            return original(sample, *args, **kwargs)

        model.forward = _spy
        gen = torch.Generator().manual_seed(2)
        noise = sampler._create_initial_noise(_W, _H, gen)
        latents = sampler.denoise(
            noise=noise,
            prompt_embedding=sampler.encode_prompt("test"),
            num_steps=2,
            guidance_scale=0.0,
            seed=2,
        )
        assert latents.isfinite().all()
        assert batch_sizes == [1, 1], (
            f"gs=0 must run a single un-batched pass per step, got {batch_sizes}"
        )
        assert uncond_calls == [], "negative must NOT be encoded without CFG"

    def test_distilled_definition_forces_no_cfg(self):
        """dreamlite-mobile (is_distilled: true) ignores guidance — CFG was
        distilled away (DreamLiteMobilePipeline warns + ignores)."""
        sampler, model = _build_sampler(is_distilled=True)
        batch_sizes = []
        original = model.forward

        def _spy(sample, *args, **kwargs):
            batch_sizes.append(sample.shape[0])
            return original(sample, *args, **kwargs)

        model.forward = _spy
        gen = torch.Generator().manual_seed(3)
        noise = sampler._create_initial_noise(_W, _H, gen)
        sampler.denoise(
            noise=noise,
            prompt_embedding=sampler.encode_prompt("distilled"),
            num_steps=2,
            guidance_scale=3.5,  # deliberately non-zero — must be ignored
            seed=3,
        )
        assert batch_sizes == [1, 1], (
            f"distilled checkpoint must never CFG-batch, got {batch_sizes}"
        )

    def test_cfg_combine_matches_pipeline_convention(self):
        """velocity = uncond + g * (cond - uncond) — the DreamLitePipeline
        formula (standard CFG; NOT krea2's cond + g*(cond - uncond))."""
        from app.engine.models.families.dreamlite.sampler import _combine_cfg

        cond = torch.ones(1, 4, 2, 2)
        uncond = torch.zeros(1, 4, 2, 2)
        out = _combine_cfg(cond, uncond, 3.5)
        assert torch.allclose(out, torch.full_like(cond, 3.5))

        cond2 = torch.full((1, 4), 2.0)
        uncond2 = torch.full((1, 4), 1.0)
        # 1 + 3*(2-1) = 4
        assert torch.allclose(
            _combine_cfg(cond2, uncond2, 3.0), torch.full_like(cond2, 4.0),
        )


# ── Step 4: decode + trainer wiring ──────────────────────────────────────────


class TestDecodeAndTrainerWiring:
    def test_decode_latents_returns_pil(self):
        from PIL import Image

        sampler, _ = _build_sampler()
        latents = torch.randn(1, 4, _LAT, _LAT)
        result = sampler.decode_latents(latents)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (_W, _H)

    def test_decode_applies_pipeline_scaling_formula(self):
        """decode input = latents / scaling_factor + shift_factor."""
        sampler, _ = _build_sampler()
        vae = sampler.pipeline.vae
        vae.config.scaling_factor = 0.5
        vae.config.shift_factor = 0.25
        seen = {}

        def _decode(latents, return_dict=False):
            seen["latents"] = latents
            return (torch.zeros(latents.shape[0], 3, _H, _W),)

        vae.decode = _decode
        latents = torch.randn(1, 4, _LAT, _LAT)
        sampler.decode_latents(latents)
        assert torch.allclose(seen["latents"], latents / 0.5 + 0.25)

    def test_create_sampler_returns_dreamlite_sampler_when_enabled(self):
        from app.engine.models.families.dreamlite.sampler import (
            DreamLiteSampler,
        )
        from app.engine.models.families.dreamlite.trainer import (
            DreamLiteTrainer,
        )

        trainer = MagicMock(spec=DreamLiteTrainer)
        trainer.config = {"sample_every_n_steps": 50}
        trainer.device = torch.device("cpu")
        trainer.definition = MagicMock()

        sampler = DreamLiteTrainer._create_sampler(trainer)
        assert isinstance(sampler, DreamLiteSampler)

    def test_create_sampler_returns_none_when_disabled(self):
        from app.engine.models.families.dreamlite.trainer import (
            DreamLiteTrainer,
        )

        trainer = MagicMock(spec=DreamLiteTrainer)
        trainer.config = {"sample_every_n_steps": 0}
        trainer.device = torch.device("cpu")
        trainer.definition = MagicMock()

        assert DreamLiteTrainer._create_sampler(trainer) is None


# ── Step 5: precision contract ───────────────────────────────────────────────


class TestPrecisionContract:
    def test_no_autocast_in_denoise_source(self):
        from app.engine.models.families.dreamlite.sampler import (
            DreamLiteSampler,
        )

        source = inspect.getsource(DreamLiteSampler.denoise)
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
            guidance_scale=3.5,
            seed=7,
        )
        assert latents.isfinite().all()
        assert latents.float().std() > 0
        assert latents.dtype == torch.float32

    def test_driver_receives_raw_thousand_scale_timesteps(self):
        """The sampler hands RAW [0,1000] timesteps to driver.forward_pass
        (the DreamLite UNet consumes raw timesteps — never rescale)."""
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

    def test_cached_fp32_embeds_are_cast_to_model_dtype(self):
        """Wave 1 lesson: fp32-cached TE embeddings vs a bf16 model is a
        real crash — the sampler must cast embeds to the MODEL dtype before
        the forward."""
        sampler, model = _build_sampler()
        model.to(torch.bfloat16)

        seen_dtypes = []
        real_forward = sampler.pipeline.driver.forward_pass

        def _spy(noisy_input, timesteps, text_embeddings, batch):
            emb, _mask = text_embeddings
            seen_dtypes.append(emb.dtype)
            return real_forward(
                noisy_input=noisy_input,
                timesteps=timesteps,
                text_embeddings=text_embeddings,
                batch=batch,
            )

        sampler.pipeline.driver = MagicMock(wraps=sampler.pipeline.driver)
        sampler.pipeline.driver.forward_pass = _spy

        gen = torch.Generator().manual_seed(5)
        noise = sampler._create_initial_noise(_W, _H, gen)
        latents = sampler.denoise(
            noise=noise,
            prompt_embedding=sampler.encode_prompt("dtype cast"),
            num_steps=1,
            guidance_scale=3.5,
            seed=5,
        )
        assert seen_dtypes and all(d == torch.bfloat16 for d in seen_dtypes), (
            f"embeds must be cast to the model dtype (bf16), got {seen_dtypes}"
        )
        assert latents.dtype == torch.float32
        assert latents.isfinite().all()
