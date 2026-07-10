"""Tests for PRXSampler — precision contract + CFG + no-mu scheduler.

Correctness invariants (mirroring test_ovis_image_sampler.py):
  1. encode_prompt returns dict with 3-D embeds + bool mask via
     trainer.encode_text
  2. denoise: fp32 trajectory, raw [0,1000] timesteps into
     driver.forward_pass (the shared adapter divides by 1000)
  3. CFG per PRXPipeline: gate ``guidance_scale > 1.0``; combine
     ``uncond + g * (cond - uncond)``
  4. Scheduler: checkpoint FlowMatchEulerDiscreteScheduler with STATIC
     shift 3.0 — plain set_timesteps, NO mu / sigmas / dynamic shifting
  5. NO torch.autocast around the DiT forward (autocast-collapse gotcha);
     cached TE embeddings are cast to the model dtype at the boundary
  6. Native 512 default resolution / 28 steps / guidance 4.0 fill-in
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest
import torch


# ── Shared tiny transformer config (1 block, minimal dims) ───────────────────

_TINY_CFG = dict(
    in_channels=4,
    patch_size=2,
    context_in_dim=8,
    hidden_size=32,
    num_heads=2,
    depth=1,
    axes_dim=[8, 8],
)

_W, _H = 32, 32       # pixel dims → lat 4×4 (vae_sf=8; divisible by patch 2)
_VAE_SF = 8
_LAT = _H // _VAE_SF  # = 4 (pipeline prepare_latents formula — no packing)
_TXT_SEQ = 7
_TEXT_DIM = 8         # == context_in_dim of the tiny model

# PRX scheduler facts (checkpoint scheduler_config.json)
_ARCH = {
    "scheduler.num_train_timesteps": 1000,
    "scheduler.shift": 3.0,
    "scheduler.use_dynamic_shifting": False,
    "vae.latent_channels": 4,
    "vae.vae_scale_factor": 8,
}


def _build_tiny_model():
    from diffusers.models.transformers.transformer_prx import (
        PRXTransformer2DModel,
    )

    torch.manual_seed(0)
    return PRXTransformer2DModel(**_TINY_CFG).eval()


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
    """Mock PRXTrainer with a REAL PRXDriver + tiny model."""
    from app.engine.models.families.prx.driver import PRXDriver

    drv_defn = MagicMock()
    drv_defn.family = "prx"
    drv_defn.id = "prx-test"
    drv_defn.lora_targetable_modules = []
    drv_defn.architecture_params = dict(_ARCH)

    driver = PRXDriver(drv_defn, torch.device("cpu"))
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
        mask = torch.ones(b, _TXT_SEQ, dtype=torch.bool)
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
    from app.engine.models.families.prx.sampler import PRXSampler

    model = _build_tiny_model()
    pipeline = _build_mock_pipeline(model)
    return PRXSampler(pipeline), model


# ── Step 1: encode_prompt ────────────────────────────────────────────────────


class TestEncodePrompt:
    def test_encode_prompt_returns_3d_embeds_and_bool_mask(self):
        sampler, _ = _build_sampler()
        result = sampler.encode_prompt("a test prompt")

        assert isinstance(result, dict)
        assert "embeds" in result and "mask" in result
        assert result["embeds"].ndim == 3, "embeds must be [B, L, D]"
        assert result["embeds"].shape[0] == 1
        assert result["mask"].ndim == 2
        assert result["mask"].dtype == torch.bool

    def test_encode_prompt_delegates_to_pipeline(self):
        sampler, _ = _build_sampler()
        calls = []

        def _spy(prompts, dtype=None):
            calls.append(prompts)
            return (
                torch.randn(len(prompts), _TXT_SEQ, _TEXT_DIM),
                torch.ones(len(prompts), _TXT_SEQ, dtype=torch.bool),
            )

        sampler.pipeline.encode_text = _spy
        sampler.encode_prompt("hello")
        assert calls == [["hello"]]


# ── Step 2: scheduler — static shift, NO mu ──────────────────────────────────


class TestScheduler:
    def test_scheduler_matches_checkpoint_config(self):
        from diffusers import FlowMatchEulerDiscreteScheduler

        sampler, _ = _build_sampler()
        sched = sampler._get_scheduler()
        assert isinstance(sched, FlowMatchEulerDiscreteScheduler)
        assert sched.config.num_train_timesteps == 1000
        assert sched.config.shift == 3.0
        assert sched.config.use_dynamic_shifting is False

    def test_set_timesteps_called_plain_no_mu_no_sigmas(self):
        """PRXPipeline prepares timesteps with plain set_timesteps(n) —
        passing mu or custom sigmas would silently change the schedule."""
        sampler, _ = _build_sampler()
        sched = sampler._get_scheduler()
        seen: list[dict] = []
        original = sched.set_timesteps

        def _spy(*args, **kwargs):
            seen.append({"args": args, "kwargs": dict(kwargs)})
            return original(*args, **kwargs)

        sched.set_timesteps = _spy

        gen = torch.Generator().manual_seed(11)
        noise = sampler._create_initial_noise(_W, _H, gen)
        sampler.denoise(
            noise=noise,
            prompt_embedding=sampler.encode_prompt("no mu"),
            num_steps=2,
            guidance_scale=0.0,
            seed=11,
        )

        assert seen, "set_timesteps never called"
        for call in seen:
            assert "mu" not in call["kwargs"], "PRX must NOT pass mu"
            assert "sigmas" not in call["kwargs"], "PRX must NOT pass sigmas"

    def test_first_timestep_is_raw_1000(self):
        """shift-3.0 static schedule starts at raw t=1000 (sigma 1.0)."""
        sampler, _ = _build_sampler()
        sched = sampler._get_scheduler()
        sched.set_timesteps(4)
        assert float(sched.timesteps[0]) == pytest.approx(1000.0)


# ── Step 3: denoise — shapes, CFG gating, fp32 trajectory ────────────────────


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
            guidance_scale=4.0,
            seed=42,
        )

        assert latents.shape == (1, 4, _LAT, _LAT)
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
            guidance_scale=4.0,
            seed=1,
        )
        assert len(forward_calls) == 4, (
            f"CFG must run cond+uncond per step (4 total), got {len(forward_calls)}"
        )

    @pytest.mark.parametrize("gs", [1.0, 0.0])
    def test_no_cfg_single_forward_when_gs_at_or_below_1(self, gs):
        """PRXPipeline gates CFG at guidance_scale > 1.0."""
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
        """velocity = uncond + g * (cond - uncond) — the PRXPipeline formula."""
        from app.engine.models.families.prx.sampler import _combine_cfg

        pos = torch.ones(1, 4, 2, 2)
        neg = torch.zeros(1, 4, 2, 2)
        out = _combine_cfg(pos, neg, 4.0)
        assert torch.allclose(out, torch.full_like(pos, 4.0))

        pos2 = torch.full((1, 4), 2.0)
        neg2 = torch.full((1, 4), 1.0)
        # 1 + 3*(2-1) = 4
        assert torch.allclose(_combine_cfg(pos2, neg2, 3.0), torch.full_like(pos2, 4.0))


# ── Step 4: native-512 defaults + decode + trainer wiring ────────────────────


class TestDefaultsDecodeAndWiring:
    def test_sample_single_fills_prx_native_defaults(self, monkeypatch):
        """Unset width/height/steps/guidance default to 512/512/28/4.0
        (definition-driven) instead of the generic base's 1024/20/3.5."""
        from app.engine.core.sampling import GenericSamplingPipeline

        sampler, _ = _build_sampler()
        captured: dict = {}

        def _fake_base(self, cfg, step):
            captured.update(cfg)
            return MagicMock()

        monkeypatch.setattr(GenericSamplingPipeline, "_sample_single", _fake_base)

        sampler._sample_single({"prompt": "defaults"}, 0)
        assert captured["width"] == 512
        assert captured["height"] == 512
        assert captured["num_inference_steps"] == 28
        assert captured["guidance_scale"] == 4.0

    def test_sample_single_respects_explicit_values(self, monkeypatch):
        from app.engine.core.sampling import GenericSamplingPipeline

        sampler, _ = _build_sampler()
        captured: dict = {}

        def _fake_base(self, cfg, step):
            captured.update(cfg)
            return MagicMock()

        monkeypatch.setattr(GenericSamplingPipeline, "_sample_single", _fake_base)

        # Native-compatible explicit values (both <= 512) pass through.
        sampler._sample_single(
            {"prompt": "explicit", "width": 384, "height": 256,
             "num_inference_steps": 12, "guidance_scale": 1.0},
            0,
        )
        assert captured["width"] == 384
        assert captured["height"] == 256
        assert captured["num_inference_steps"] == 12
        assert captured["guidance_scale"] == 1.0

    def test_sample_single_clamps_over_native_resolution(self, monkeypatch):
        """A 512-native checkpoint must NOT be previewed above native.

        The UI stamps sample prompts with the global-default 1024; rendering
        the 512-native model at 2x native produces doubled/mirrored
        composition. _sample_single clamps the longest side down to 512,
        preserving aspect ratio and the 16px dimension multiple.
        """
        from app.engine.core.sampling import GenericSamplingPipeline

        sampler, _ = _build_sampler()
        captured: dict = {}

        def _fake_base(self, cfg, step):
            captured.update(cfg)
            return MagicMock()

        monkeypatch.setattr(GenericSamplingPipeline, "_sample_single", _fake_base)

        # Square 2x-native → clamp to native square.
        sampler._sample_single(
            {"prompt": "big", "width": 1024, "height": 1024}, 0
        )
        assert captured["width"] == 512
        assert captured["height"] == 512

        # Non-square over-native → longest side to 512, aspect preserved,
        # each dim a multiple of 16.
        captured.clear()
        sampler._sample_single(
            {"prompt": "wide", "width": 1024, "height": 512}, 0
        )
        assert captured["width"] == 512
        assert captured["height"] == 256
        assert captured["width"] % 16 == 0 and captured["height"] % 16 == 0

    def test_decode_latents_returns_pil(self):
        from PIL import Image

        sampler, _ = _build_sampler()
        latents = torch.randn(1, 4, _LAT, _LAT)
        result = sampler.decode_latents(latents)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (_W, _H)

    def test_create_sampler_returns_prx_sampler_when_enabled(self):
        from app.engine.models.families.prx.sampler import PRXSampler
        from app.engine.models.families.prx.trainer import PRXTrainer

        trainer = MagicMock(spec=PRXTrainer)
        trainer.config = {"sample_every_n_steps": 50}
        trainer.device = torch.device("cpu")
        trainer.definition = MagicMock()

        sampler = PRXTrainer._create_sampler(trainer)
        assert isinstance(sampler, PRXSampler)

    def test_create_sampler_returns_none_when_disabled(self):
        from app.engine.models.families.prx.trainer import PRXTrainer

        trainer = MagicMock(spec=PRXTrainer)
        trainer.config = {"sample_every_n_steps": 0}
        trainer.device = torch.device("cpu")
        trainer.definition = MagicMock()

        assert PRXTrainer._create_sampler(trainer) is None


# ── Step 5: precision contract ───────────────────────────────────────────────


class TestPrecisionContract:
    def test_no_autocast_in_denoise_source(self):
        from app.engine.models.families.prx.sampler import PRXSampler

        source = inspect.getsource(PRXSampler.denoise)
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
            guidance_scale=4.0,
            seed=7,
        )
        assert latents.isfinite().all()
        assert latents.float().std() > 0
        assert latents.dtype == torch.float32

    def test_driver_receives_raw_thousand_scale_timesteps(self):
        """The sampler hands RAW [0,1000] timesteps to driver.forward_pass
        (the shared adapter divides by 1000 — never divide twice)."""
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
        assert seen_ts[0] == pytest.approx(1000.0), (
            f"first timestep must be raw 1000, got {seen_ts[0]}"
        )

    def test_cached_fp32_embeds_cast_to_model_dtype_at_boundary(self):
        """Wave-1 lesson: cached TE embeddings (fp32 on disk) must be cast
        to the transformer dtype before the forward — pin by running a bf16
        transformer against the fp32-producing encode_text stub."""
        sampler, model = _build_sampler()
        model.to(torch.bfloat16)

        captured: dict = {}
        original = model.forward

        def _spy(*args, **kwargs):
            captured.update(kwargs)
            return original(*args, **kwargs)

        model.forward = _spy

        gen = torch.Generator().manual_seed(9)
        noise = sampler._create_initial_noise(_W, _H, gen)
        latents = sampler.denoise(
            noise=noise,
            prompt_embedding=sampler.encode_prompt("dtype boundary"),
            num_steps=2,
            guidance_scale=0.0,
            seed=9,
        )
        assert captured["encoder_hidden_states"].dtype == torch.bfloat16, (
            "embeds must be cast to the model dtype at the forward boundary"
        )
        # Bool mask passes through uncast
        assert captured["attention_mask"].dtype == torch.bool
        # Trajectory still fp32 outside the model
        assert latents.dtype == torch.float32
        assert latents.isfinite().all()
