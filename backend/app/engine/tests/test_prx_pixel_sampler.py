"""Tests for PRXPixelSampler — pixel-space x0 sampling contract.

Correctness invariants (mirroring test_prx_sampler.py plus the pixel/x0
specifics from PRXPixelPipeline.__call__):
  1. encode_prompt returns dict with 3-D embeds + bool mask via
     trainer.encode_text
  2. INITIAL NOISE: pixel-space [1, 3, H, W] (NO vae downscale) scaled by
     noise_scale 2.0 — prepare_latents does randn × noise_scale
  3. X0→VELOCITY per step: the scheduler receives
     (latents - x0_cfg) / clamp(t/1000, min=0.05), with CFG applied to the
     x0 PREDICTION before conversion
  4. CFG per the pipeline: gate ``guidance_scale > 1.0``; combine
     ``uncond + g * (cond - uncond)``
  5. Scheduler: checkpoint FlowMatchEulerDiscreteScheduler with STATIC
     shift 3.0 — plain set_timesteps, NO mu / sigmas / dynamic shifting
  6. NO torch.autocast around the DiT forward (autocast-collapse gotcha);
     cached TE embeddings are cast to the model dtype at the boundary;
     fp32 trajectory
  7. NO VAE: decode_latents is a pure [-1,1]→PIL postprocess
  8. Native 1024 default resolution / 28 steps / guidance 4.0 fill-in
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest
import torch


# ── Shared tiny PIXEL-variant transformer config (1 block) ───────────────────

_TINY_CFG = dict(
    in_channels=3,
    patch_size=2,
    context_in_dim=8,
    hidden_size=32,
    num_heads=2,
    depth=1,
    axes_dim=[8, 8],
    bottleneck_size=16,
    resolution_embeds=True,
)

_W, _H = 16, 16       # pixel dims — pixel space, no VAE downscale
_TXT_SEQ = 7
_TEXT_DIM = 8         # == context_in_dim of the tiny model

# PRXPixel facts (checkpoint scheduler_config.json + model_index.json)
_ARCH = {
    "scheduler.num_train_timesteps": 1000,
    "scheduler.shift": 3.0,
    "scheduler.use_dynamic_shifting": False,
    "transformer.in_channels": 3,
    "pipeline.noise_scale": 2.0,
    "pipeline.velocity_t_floor": 0.05,
}

_DEFAULTS = {
    "is_distilled": False,
    "resolution": 1024,
    "num_inference_steps": 28,
    "guidance_scale": 4.0,
}


def _build_tiny_model():
    from diffusers.models.transformers.transformer_prx import (
        PRXTransformer2DModel,
    )

    torch.manual_seed(0)
    return PRXTransformer2DModel(**_TINY_CFG).eval()


def _build_mock_pipeline(model):
    """Mock PRXPixelTrainer with a REAL PRXPixelDriver + tiny model."""
    from app.engine.models.families.prx_pixel.driver import PRXPixelDriver

    drv_defn = MagicMock()
    drv_defn.family = "prx_pixel"
    drv_defn.id = "prx-pixel-test"
    drv_defn.lora_targetable_modules = []
    drv_defn.architecture_params = dict(_ARCH)

    driver = PRXPixelDriver(drv_defn, torch.device("cpu"))
    driver.assign_components({
        "unet": model,
        "text_encoder": None,
        "tokenizer": None,
    })

    pipeline = MagicMock()
    pipeline.device = torch.device("cpu")
    pipeline.transformer = model
    pipeline.vae = None  # pixel space — the trainer never loads a VAE
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
    defn.defaults = dict(_DEFAULTS)
    pipeline.definition = defn

    pipeline.config = {
        "sample_every_n_steps": 50,
        "sample_negative_prompt": "",
    }
    pipeline._block_swap_managers = None
    return pipeline


def _build_sampler():
    from app.engine.models.families.prx_pixel.sampler import PRXPixelSampler

    model = _build_tiny_model()
    pipeline = _build_mock_pipeline(model)
    return PRXPixelSampler(pipeline), model


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


# ── Step 2: initial noise — pixel space × noise_scale ────────────────────────


class TestInitialNoise:
    def test_noise_is_pixel_shape_fp32_and_scaled_by_two(self):
        """prepare_latents contract: randn at FULL pixel resolution (no VAE
        downscale), in_channels=3, multiplied by noise_scale 2.0."""
        sampler, _ = _build_sampler()

        gen = torch.Generator().manual_seed(42)
        noise = sampler._create_initial_noise(_W, _H, gen)

        assert noise.shape == (1, 3, _H, _W), (
            f"pixel-space noise must be [1,3,{_H},{_W}], got {tuple(noise.shape)}"
        )
        assert noise.dtype == torch.float32

        # Byte-exact scale check: same seed reproduces raw randn; the
        # sampler's output must be exactly 2× that draw.
        gen2 = torch.Generator().manual_seed(42)
        raw = torch.randn(
            (1, 3, _H, _W), generator=gen2, dtype=torch.float32,
        )
        assert torch.allclose(noise, raw * 2.0), (
            "initial noise must be randn × noise_scale (2.0)"
        )

    def test_noise_scale_is_definition_driven(self):
        sampler, _ = _build_sampler()
        sampler.pipeline.definition.architecture_params = {
            **_ARCH, "pipeline.noise_scale": 1.5,
        }

        gen = torch.Generator().manual_seed(7)
        noise = sampler._create_initial_noise(_W, _H, gen)
        gen2 = torch.Generator().manual_seed(7)
        raw = torch.randn((1, 3, _H, _W), generator=gen2, dtype=torch.float32)
        assert torch.allclose(noise, raw * 1.5)


# ── Step 3: scheduler — static shift, NO mu ──────────────────────────────────


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
        """PRXPixelPipeline prepares timesteps with plain set_timesteps(n) —
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
            assert "mu" not in call["kwargs"], "PRXPixel must NOT pass mu"
            assert "sigmas" not in call["kwargs"], "PRXPixel must NOT pass sigmas"


# ── Step 4: denoise — x0→velocity conversion, CFG on x0, shapes ──────────────


class TestDenoise:
    def test_denoise_shape_fp32_and_finite_cfg(self):
        sampler, _ = _build_sampler()
        gen = torch.Generator().manual_seed(42)
        noise = sampler._create_initial_noise(_W, _H, gen)

        prompt_emb = sampler.encode_prompt("a test image")
        latents = sampler.denoise(
            noise=noise,
            prompt_embedding=prompt_emb,
            num_steps=2,
            guidance_scale=4.0,
            seed=42,
        )

        assert latents.shape == (1, 3, _H, _W)
        assert latents.dtype == torch.float32, "trajectory must stay fp32"
        assert latents.isfinite().all(), "denoise output contains NaN or inf"
        assert latents.float().std() > 0, "denoise output is degenerate"

    def test_scheduler_receives_x0_converted_velocity(self):
        """THE x0 contract: what reaches scheduler.step must be
        (latents - x0_pred) / clamp(t/1000, 0.05) — NOT the raw model
        output (feeding x0 straight to the flow scheduler silently produces
        garbage trajectories)."""
        sampler, model = _build_sampler()
        sched = sampler._get_scheduler()

        x0_outputs: list[torch.Tensor] = []
        original_forward = model.forward

        def _forward_spy(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            x0_outputs.append(
                (out[0] if isinstance(out, tuple) else out).detach().float()
            )
            return out

        model.forward = _forward_spy

        step_calls: list[dict] = []
        original_step = sched.step

        def _step_spy(model_output, timestep, sample, *args, **kwargs):
            step_calls.append({
                "velocity": model_output.detach().clone(),
                "t": float(timestep),
                "sample": sample.detach().clone(),
            })
            return original_step(model_output, timestep, sample, *args, **kwargs)

        sched.step = _step_spy

        gen = torch.Generator().manual_seed(5)
        noise = sampler._create_initial_noise(_W, _H, gen)
        sampler.denoise(
            noise=noise,
            prompt_embedding=sampler.encode_prompt("x0 conversion"),
            num_steps=2,
            guidance_scale=0.0,  # single forward → x0_outputs[i] is the pred
            seed=5,
        )

        assert step_calls and x0_outputs
        for i, call in enumerate(step_calls):
            t_x = max(call["t"] / 1000.0, 0.05)
            expected = (call["sample"] - x0_outputs[i]) / t_x
            assert torch.allclose(call["velocity"], expected, atol=1e-4), (
                f"step {i}: scheduler must receive (latents - x0)/t_x"
            )

    def test_cfg_applied_to_x0_before_conversion(self):
        """CFG combines the x0 PREDICTIONS first, then converts once:
        v = (latents - (uncond + g·(cond - uncond))) / t_x. Applying CFG
        after conversion is algebraically identical here, but the pinned
        formula keeps the pipeline-verbatim order."""
        sampler, model = _build_sampler()
        sched = sampler._get_scheduler()
        g = 4.0

        x0_outputs: list[torch.Tensor] = []
        original_forward = model.forward

        def _forward_spy(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            x0_outputs.append(
                (out[0] if isinstance(out, tuple) else out).detach().float()
            )
            return out

        model.forward = _forward_spy

        step_calls: list[dict] = []
        original_step = sched.step

        def _step_spy(model_output, timestep, sample, *args, **kwargs):
            step_calls.append({
                "velocity": model_output.detach().clone(),
                "t": float(timestep),
                "sample": sample.detach().clone(),
            })
            return original_step(model_output, timestep, sample, *args, **kwargs)

        sched.step = _step_spy

        gen = torch.Generator().manual_seed(6)
        noise = sampler._create_initial_noise(_W, _H, gen)
        sampler.denoise(
            noise=noise,
            prompt_embedding=sampler.encode_prompt("cfg order"),
            num_steps=2,
            guidance_scale=g,
            seed=6,
        )

        # Two forwards per step: [cond, uncond] per our sampler convention.
        assert len(x0_outputs) == 2 * len(step_calls)
        for i, call in enumerate(step_calls):
            cond, uncond = x0_outputs[2 * i], x0_outputs[2 * i + 1]
            x0_cfg = uncond + g * (cond - uncond)
            t_x = max(call["t"] / 1000.0, 0.05)
            expected = (call["sample"] - x0_cfg) / t_x
            assert torch.allclose(call["velocity"], expected, atol=1e-4), (
                f"step {i}: CFG must combine x0 preds before the conversion"
            )

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
        """PRXPixelPipeline gates CFG at guidance_scale > 1.0."""
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


# ── Step 5: native-1024 defaults + pixel decode + trainer wiring ─────────────


class TestDefaultsDecodeAndWiring:
    def test_sample_single_fills_pixel_native_defaults(self, monkeypatch):
        """Unset width/height/steps/guidance default to 1024/1024/28/4.0
        (definition-driven — PRXPixel is a 1024px model)."""
        from app.engine.core.sampling import GenericSamplingPipeline

        sampler, _ = _build_sampler()
        captured: dict = {}

        def _fake_base(self, cfg, step):
            captured.update(cfg)
            return MagicMock()

        monkeypatch.setattr(GenericSamplingPipeline, "_sample_single", _fake_base)

        sampler._sample_single({"prompt": "defaults"}, 0)
        assert captured["width"] == 1024
        assert captured["height"] == 1024
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

        sampler._sample_single(
            {"prompt": "explicit", "width": 704, "height": 352,
             "num_inference_steps": 12, "guidance_scale": 1.0},
            0,
        )
        assert captured["width"] == 704
        assert captured["height"] == 352
        assert captured["num_inference_steps"] == 12
        assert captured["guidance_scale"] == 1.0

    def test_decode_latents_is_pure_pixel_postprocess(self):
        """NO VAE: the denoised output IS the image in [-1,1]. decode maps
        -1→0, 0→127/128, +1→255 with clamping — and never touches a VAE."""
        from PIL import Image

        sampler, _ = _build_sampler()
        assert sampler.pipeline.vae is None  # would explode if decode used it

        latents = torch.zeros(1, 3, _H, _W)
        latents[0, :, 0, 0] = -5.0   # clamps to -1 → 0
        latents[0, :, 0, 1] = 1.0    # → 255
        result = sampler.decode_latents(latents)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (_W, _H)
        px = result.load()
        assert px[0, 0] == (0, 0, 0), "clamped -1 must map to black"
        assert px[1, 0] == (255, 255, 255), "+1 must map to white"
        # 0 → 127.5 → 127 (uint8 floor)
        assert px[2, 0][0] in (127, 128)

    def test_create_sampler_returns_pixel_sampler_when_enabled(self):
        from app.engine.models.families.prx_pixel.sampler import PRXPixelSampler
        from app.engine.models.families.prx_pixel.trainer import PRXPixelTrainer

        trainer = MagicMock(spec=PRXPixelTrainer)
        trainer.config = {"sample_every_n_steps": 50}
        trainer.device = torch.device("cpu")
        trainer.definition = MagicMock()

        sampler = PRXPixelTrainer._create_sampler(trainer)
        assert isinstance(sampler, PRXPixelSampler)

    def test_create_sampler_returns_none_when_disabled(self):
        from app.engine.models.families.prx_pixel.trainer import PRXPixelTrainer

        trainer = MagicMock(spec=PRXPixelTrainer)
        trainer.config = {"sample_every_n_steps": 0}
        trainer.device = torch.device("cpu")
        trainer.definition = MagicMock()

        assert PRXPixelTrainer._create_sampler(trainer) is None


# ── Step 6: precision contract ───────────────────────────────────────────────


class TestPrecisionContract:
    def test_no_autocast_in_denoise_source(self):
        from app.engine.models.families.prx_pixel.sampler import PRXPixelSampler

        source = inspect.getsource(PRXPixelSampler.denoise)
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
