"""Tests for BooguImageSampler — Base (vendored-scheduler CFG loop) + Turbo
(DMD few-step loop) in-training preview sampling (Task 6).

Mirrors the krea2/ovis_image sampler-test style: tiny fake/real components,
no GPU, no network, no downloads. Upstream evidence trail (cite file:line
for every formula pinned here — see task-6-report.md for the full trail):

- Base denoise loop: ``pipeline_boogu.py:3243`` (``processing()``), guidance
  branch for pure-T2I (``control_inputs: 0`` -> ``task_type == "t2i"``
  always, ``_get_task_type_by_ref_latents`` at :3001-3011) is the FINAL
  ``elif text_guidance_scale > 1.0:`` branch at :3615-3649 (Lumina-style
  scale-1 guidance: ``model_pred + (text_guidance_scale - 1) * (model_pred -
  model_pred_drop_all)``, gated ``> 1.0``); scheduler.step at :3651-3653;
  VAE decode order (divide by scaling_factor THEN add shift_factor) at
  :3681-3686.
- Turbo DMD loop: ``pipeline_boogu_turbo.py`` -- sigma ladder
  ``linspace(conditioning_sigma, 1.0, steps+1)[:-1]`` (default
  ``conditioning_sigma=0.001``) at :43-72/:128; predict step
  ``latents + (1 - sigma) * model_pred`` at :74-98; renoise
  ``(1 - sigma_next) * FRESH_noise + sigma_next * x0_hat`` at :100-118; hard
  assert ``text_gs == image_gs == 1.0 and empty_gs == 0`` at :163-171; same
  VAE decode tail at :211-218.

Correctness invariants pinned:
1. Base scheduler: the LOADER-provided vendored ``FlowMatchEulerDiscreteScheduler``
   (``driver.scheduler``), never a fresh/stock instance.
2. Guidance gate: ``text_gs > 1.0`` -> ON (two forwards, negative encode via
   ``driver.encode_text("")`` -- the DROP prompt); ``== 1.0`` -> OFF (one
   forward, no negative encode).
3. Native defaults (ovis F-lesson): unset steps/guidance/resolution fill
   definition-sourced 50/4.0/1024 (base) or 4/1.0/1024 (turbo).
4. Turbo ladder + hard assert + DMD renoise math (derived from upstream, not
   guessed).
5. Precision contract: fp32 trajectory, no ``torch.autocast``, cached-embed
   dtype cast to model dtype before the forward.
6. VAE decode: ``latents / scaling_factor + shift_factor`` (matches ovis's
   convention -- same FLUX-style AutoencoderKL family).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from app.engine.models.families.boogu_image.driver import BooguImageDriver
from app.engine.models.families.boogu_image.vendor.schedulers.scheduling_flow_match_euler_discrete_time_shifting import (
    FlowMatchEulerDiscreteScheduler,
)

TINY_AXES_DIM_ROPE = (2, 2, 4)
TINY_AXES_LENS = (64, 64, 64)
TINY_IN_CHANNELS = 4
TINY_TEXT_DIM = 8


# ── Shared fake-model helpers ────────────────────────────────────────────


class _ConstantModel(nn.Module):
    """Fake transformer: output == a constant derived from the mean of
    ``instruction_hidden_states``. Lets tests fabricate exactly-known
    cond/uncond velocities by controlling the mean of the fed embeddings.
    Records call count and captures the last call's kwargs for inspection.
    """

    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = SimpleNamespace(
            axes_dim_rope=TINY_AXES_DIM_ROPE, axes_lens=TINY_AXES_LENS,
        )
        self.dummy = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.calls = 0
        self.last_kwargs: dict = {}

    def forward(self, hidden_states, instruction_hidden_states, **kwargs):
        self.calls += 1
        self.last_kwargs = dict(
            hidden_states=hidden_states,
            instruction_hidden_states=instruction_hidden_states,
            **kwargs,
        )
        scale = instruction_hidden_states.float().mean()
        return [torch.full_like(h, float(scale)) for h in hidden_states]


class _PerfectVelocityModel(nn.Module):
    """Oracle: returns the TRUE, t-independent velocity ``x0 - noise`` for
    the Base loop's constant-velocity round-trip proof (mirrors
    test_boogu_image_driver.py's ``TestTimeConventionRoundTrip``)."""

    def __init__(self, velocity: torch.Tensor):
        super().__init__()
        self.config = SimpleNamespace(
            axes_dim_rope=TINY_AXES_DIM_ROPE, axes_lens=TINY_AXES_LENS,
        )
        self.dummy = nn.Parameter(torch.zeros(1))
        self.velocity = velocity

    def forward(self, hidden_states, **kwargs):
        return [self.velocity[i] for i in range(len(hidden_states))]


class _PerfectDmdGenerator(nn.Module):
    """Oracle for the DMD/Turbo loop: given ANY current latent ``z`` at
    sigma ``s`` (assuming ``z = (1-s)*noise + s*x0`` for SOME noise), returns
    the velocity that makes ``_predict_dmd_student_step`` land EXACTLY on
    ``x0_true`` regardless of ``z``/``s``.

    Derivation: ``_predict_dmd_student_step`` computes
    ``x0_hat = z + (1 - s) * v`` (pipeline_boogu_turbo.py:74-98). Solve for
    the ``v`` that forces ``x0_hat == x0_true``:
        x0_true = z + (1 - s) * v
        v = (x0_true - z) / (1 - s)
    This holds for ANY z (not just ones actually reachable from x0_true via
    the lerp), which is exactly what makes a "perfect one-step generator"
    self-correcting at every DMD step: whatever the previous step's fresh
    renoise produced, THIS step's predict always recovers x0_true exactly.
    Valid for sigma < 1 (all DMD ladder sigmas satisfy this: the ladder is
    ``linspace(conditioning_sigma, 1.0, steps+1)[:-1]``, which excludes 1.0).
    """

    def __init__(self, x0_true: torch.Tensor):
        super().__init__()
        self.config = SimpleNamespace(
            axes_dim_rope=TINY_AXES_DIM_ROPE, axes_lens=TINY_AXES_LENS,
        )
        self.dummy = nn.Parameter(torch.zeros(1))
        self.x0_true = x0_true

    def forward(self, hidden_states, timestep, **kwargs):
        out = []
        for i, h in enumerate(hidden_states):
            sigma = float(timestep[i])
            x0_i = self.x0_true[i]
            out.append((x0_i - h) / (1.0 - sigma))
        return out


def _definition(is_distilled: bool, **defaults_overrides) -> MagicMock:
    d = MagicMock()
    d.family = "boogu_image"
    d.id = "boogu-image-test"
    d.lora_targetable_modules = ["single_stream_layers.0.attn.to_q"]
    d.architecture_params = {
        "vae.vae_scale_factor": 8,
        "vae.latent_channels": TINY_IN_CHANNELS,
    }
    defaults = {
        "resolution": 1024,
        "is_distilled": is_distilled,
        "guidance_scale": 1.0 if is_distilled else 4.0,
        "num_inference_steps": 4 if is_distilled else 50,
    }
    defaults.update(defaults_overrides)
    d.defaults = defaults
    return d


def _make_scheduler(seq_len: int = 64) -> FlowMatchEulerDiscreteScheduler:
    """Same static-shift config as the shipped definitions (scheduler.*)."""
    return FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        do_shift=True,
        dynamic_time_shift=False,
        time_shift_version="v1",
        seq_len=seq_len,
    )


def _build_mock_vae(h: int = 32, w: int = 32) -> MagicMock:
    vae = MagicMock()
    vae.dtype = torch.float32
    vae.parameters = lambda: iter([torch.zeros(1)])
    vae.config = SimpleNamespace(scaling_factor=0.3611, shift_factor=0.1159)

    captured = {}

    def _decode(latents, return_dict=False):
        captured["latents"] = latents
        b = latents.shape[0]
        return (torch.zeros(b, 3, h, w),)

    vae.decode = _decode
    vae.to = lambda *a, **k: vae
    vae._captured = captured
    return vae


def _build_pipeline(
    drv: BooguImageDriver,
    definition: MagicMock,
    encode_text_fn=None,
) -> MagicMock:
    pipeline = MagicMock()
    pipeline.device = torch.device("cpu")
    pipeline.transformer = drv.model
    pipeline.driver = drv
    pipeline.vae = _build_mock_vae()
    pipeline.definition = definition
    pipeline.config = {"sample_every_n_steps": 10}
    pipeline._block_swap_managers = None

    if encode_text_fn is None:

        def encode_text_fn(captions, dtype=None, batch=None):
            b = len(captions)
            vals = [0.0 if c == "" else 1.0 for c in captions]
            emb = torch.stack(
                [torch.full((3, TINY_TEXT_DIM), v) for v in vals], dim=0,
            )
            mask = torch.ones(b, 3, dtype=torch.long)
            return emb.to(dtype or torch.float32), mask

    pipeline.encode_text = encode_text_fn
    return pipeline


def _driver_with_model(
    model: nn.Module,
    definition: MagicMock,
    scheduler: FlowMatchEulerDiscreteScheduler | None = None,
) -> BooguImageDriver:
    """Build a driver with model/scheduler set DIRECTLY (not via
    ``assign_components``, which would overwrite ``.model`` back to None
    since no ``"unet"`` key is supplied here)."""
    drv = BooguImageDriver(definition, torch.device("cpu"))
    drv.model = model
    drv.scheduler = scheduler
    return drv


# ── encode_prompt ────────────────────────────────────────────────────────


class TestEncodePrompt:
    def test_encode_prompt_delegates_to_pipeline_encode_text(self):
        from app.engine.models.families.boogu_image.sampler import BooguImageSampler

        model = _ConstantModel()
        drv = _driver_with_model(model, _definition(False))
        calls = []

        def spy(captions, dtype=None, batch=None):
            calls.append(list(captions))
            return torch.randn(1, 3, TINY_TEXT_DIM), torch.ones(1, 3, dtype=torch.long)

        pipeline = _build_pipeline(drv, _definition(False), encode_text_fn=spy)
        sampler = BooguImageSampler(pipeline)

        result = sampler.encode_prompt("a cat")
        assert calls == [["a cat"]]
        assert "embeds" in result and "mask" in result


# ── Base loop: guidance gate + formula ──────────────────────────────────


class TestBaseGuidanceGateAndFormula:
    def test_gs_1_single_forward_no_negative_encode(self):
        from app.engine.models.families.boogu_image.sampler import BooguImageSampler

        model = _ConstantModel()
        definition = _definition(False)
        drv = _driver_with_model(model, definition)
        drv.scheduler = _make_scheduler()

        encode_calls = []

        def spy(captions, dtype=None, batch=None):
            encode_calls.append(list(captions))
            emb = torch.full((1, 3, TINY_TEXT_DIM), 1.0)
            return emb, torch.ones(1, 3, dtype=torch.long)

        pipeline = _build_pipeline(drv, definition, encode_text_fn=spy)
        sampler = BooguImageSampler(pipeline)

        prompt_emb = sampler.encode_prompt("cat")
        encode_calls.clear()

        noise = torch.randn(1, TINY_IN_CHANNELS, 4, 4)
        sampler.denoise(noise, prompt_emb, num_steps=3, guidance_scale=1.0, seed=0)

        assert model.calls == 3, "gs==1.0 must be exactly one forward per step"
        assert encode_calls == [], "gs==1.0 must NOT encode a negative prompt"

    def test_gs_above_1_two_forwards_and_negative_encode_via_drop_prompt(self):
        """text_gs=4.0: two forwards/step; negative encode goes through
        driver.encode_text("") -- spy on the driver's OWN encode_text, not a
        re-implementation (Task-5 fix: DROP system prompt for the CFG
        negative, see driver.py's _select_system_prompt)."""
        from app.engine.models.families.boogu_image.sampler import BooguImageSampler

        model = _ConstantModel()
        definition = _definition(False)
        drv = _driver_with_model(model, definition)
        drv.scheduler = _make_scheduler()
        drv.processor = MagicMock()
        drv.text_encoder = MagicMock()

        driver_encode_calls = []

        def spy_driver_encode_text(captions, dtype):
            driver_encode_calls.append(list(captions))
            emb = torch.full((1, 3, TINY_TEXT_DIM), 0.0)
            from app.engine.core.text_encoding import TextEncoderOutput

            return TextEncoderOutput(embeddings=emb, attention_mask=torch.ones(1, 3, dtype=torch.long))

        drv.encode_text = spy_driver_encode_text

        def pipeline_encode_text(captions, dtype=None, batch=None):
            # Mirrors BooguImageTrainer._encode_text_direct: unwraps the
            # driver's TextEncoderOutput -> (embeddings, mask) tuple.
            out = drv.encode_text(captions, dtype or torch.float32)
            return out.embeddings, out.attention_mask

        pipeline = _build_pipeline(drv, definition, encode_text_fn=pipeline_encode_text)
        sampler = BooguImageSampler(pipeline)

        # Positive prompt embeds mean == 1.0 (cond), fed in externally.
        prompt_emb = {
            "embeds": torch.full((1, 3, TINY_TEXT_DIM), 1.0),
            "mask": torch.ones(1, 3, dtype=torch.long),
        }

        noise = torch.randn(1, TINY_IN_CHANNELS, 4, 4)
        sampler.denoise(noise, prompt_emb, num_steps=2, guidance_scale=4.0, seed=0)

        assert model.calls == 4, "gs>1.0 must be two forwards per step"
        assert driver_encode_calls == [[""]], (
            "negative must be encoded via driver.encode_text('') exactly once "
            "(embeds are cached/reused across steps, not re-encoded per step)"
        )

    def test_combined_pred_matches_hand_computed_lumina_formula(self):
        """model_pred + (text_gs - 1) * (model_pred - model_pred_drop_all)
        (pipeline_boogu.py:3649, the pure-T2I branch) -- checked against a
        real scheduler.step() using the exact hand-computed combined pred."""
        from app.engine.models.families.boogu_image.sampler import BooguImageSampler

        model = _ConstantModel()
        definition = _definition(False)
        drv = _driver_with_model(model, definition)
        drv.scheduler = _make_scheduler()

        def encode_text_fn(captions, dtype=None, batch=None):
            val = 0.5 if captions[0] == "" else 2.0
            emb = torch.full((1, 3, TINY_TEXT_DIM), val)
            return emb, torch.ones(1, 3, dtype=torch.long)

        pipeline = _build_pipeline(drv, definition, encode_text_fn=encode_text_fn)
        sampler = BooguImageSampler(pipeline)
        prompt_emb = sampler.encode_prompt("cat")  # mean == 2.0 -> pred_cond == 2.0

        noise = torch.zeros(1, TINY_IN_CHANNELS, 4, 4)
        latents = sampler.denoise(noise, prompt_emb, num_steps=1, guidance_scale=4.0, seed=0)

        # pred_cond=2.0 (const tensor), pred_uncond=0.5 -> combined =
        # 2.0 + 3*(2.0-0.5) = 6.5 (hand-computed, matches the Lumina formula).
        expected_pred = 2.0 + 3.0 * (2.0 - 0.5)

        ref_sched = _make_scheduler()
        ref_sched.set_timesteps(
            num_inference_steps=1, device="cpu", num_tokens=4 * 4,
        )
        t = ref_sched.timesteps[0]
        t_next = ref_sched._timesteps[1]
        expected_latents = 0.0 + (t_next - t) * expected_pred

        assert torch.allclose(latents, torch.full_like(latents, float(expected_latents)), atol=1e-4)


class TestBaseLoopUsesLoaderScheduler:
    def test_denoise_raises_without_loader_scheduler(self):
        """driver.scheduler must be the LOADER-provided instance -- a driver
        with no scheduler assigned must fail loudly, not silently construct
        a fresh default (which would drop the checkpoint's shift config)."""
        from app.engine.models.families.boogu_image.sampler import BooguImageSampler

        model = _ConstantModel()
        definition = _definition(False)
        drv = _driver_with_model(model, definition)
        # No assign_components() call -- drv.scheduler stays None.

        pipeline = _build_pipeline(drv, definition)
        sampler = BooguImageSampler(pipeline)
        prompt_emb = sampler.encode_prompt("cat")
        noise = torch.randn(1, TINY_IN_CHANNELS, 4, 4)

        with pytest.raises(RuntimeError, match="scheduler"):
            sampler.denoise(noise, prompt_emb, num_steps=2, guidance_scale=1.0, seed=0)

    def test_denoise_uses_the_exact_driver_scheduler_instance(self):
        from app.engine.models.families.boogu_image.sampler import BooguImageSampler

        model = _ConstantModel()
        definition = _definition(False)
        drv = _driver_with_model(model, definition)
        sentinel = _make_scheduler(seq_len=1234)
        drv.scheduler = sentinel

        original_set_timesteps = sentinel.set_timesteps
        calls = []

        def spy_set_timesteps(*a, **k):
            calls.append((a, k))
            return original_set_timesteps(*a, **k)

        sentinel.set_timesteps = spy_set_timesteps

        pipeline = _build_pipeline(drv, definition)
        sampler = BooguImageSampler(pipeline)
        prompt_emb = sampler.encode_prompt("cat")
        noise = torch.randn(1, TINY_IN_CHANNELS, 4, 4)
        sampler.denoise(noise, prompt_emb, num_steps=2, guidance_scale=1.0, seed=0)

        assert len(calls) == 1, "the LOADER scheduler's set_timesteps must be called (not a fresh one)"


# ── Base loop: perfect-velocity oracle round trip ────────────────────────


class TestBaseLoopOracleRoundTrip:
    def test_perfect_velocity_round_trip_lands_on_x0_guidance_off(self):
        """Full loop from seeded noise through the SAMPLER's Base loop
        (guidance OFF, gs=1.0) reproduces x0 within tolerance -- pins loop
        direction, scheduler wiring, and no double-scaling in one shot
        (mirrors test_boogu_image_driver.py's TestTimeConventionRoundTrip)."""
        from app.engine.models.families.boogu_image.sampler import BooguImageSampler

        torch.manual_seed(0)
        x0 = torch.randn(1, TINY_IN_CHANNELS, 4, 4)
        noise = torch.randn(1, TINY_IN_CHANNELS, 4, 4)
        oracle_velocity = x0 - noise  # driver's own compute_target convention

        model = _PerfectVelocityModel(oracle_velocity)
        definition = _definition(False)
        drv = _driver_with_model(model, definition)
        drv.scheduler = _make_scheduler()

        pipeline = _build_pipeline(drv, definition)
        sampler = BooguImageSampler(pipeline)
        prompt_emb = sampler.encode_prompt("cat")

        latents = sampler.denoise(
            noise.clone(), prompt_emb, num_steps=7, guidance_scale=1.0, seed=0,
        )

        assert torch.allclose(latents, x0, atol=1e-4)

    def test_perfect_velocity_round_trip_is_step_count_invariant(self):
        from app.engine.models.families.boogu_image.sampler import BooguImageSampler

        torch.manual_seed(1)
        x0 = torch.randn(1, TINY_IN_CHANNELS, 4, 4)
        noise = torch.randn(1, TINY_IN_CHANNELS, 4, 4)
        oracle_velocity = x0 - noise

        for n_steps in (1, 3, 12):
            model = _PerfectVelocityModel(oracle_velocity)
            definition = _definition(False)
            drv = _driver_with_model(model, definition)
            drv.scheduler = _make_scheduler()
            pipeline = _build_pipeline(drv, definition)
            sampler = BooguImageSampler(pipeline)
            prompt_emb = sampler.encode_prompt("cat")

            latents = sampler.denoise(
                noise.clone(), prompt_emb, num_steps=n_steps, guidance_scale=1.0, seed=0,
            )
            assert torch.allclose(latents, x0, atol=1e-4), f"failed at n_steps={n_steps}"


# ── Native sample defaults (ovis F-lesson) ───────────────────────────────


class TestNativeSampleDefaults:
    def test_base_fills_50_steps_4_guidance_1024(self, monkeypatch):
        from app.engine.core.sampling import GenericSamplingPipeline
        from app.engine.models.families.boogu_image.sampler import BooguImageSampler

        model = _ConstantModel()
        definition = _definition(False)
        drv = _driver_with_model(model, definition)
        pipeline = _build_pipeline(drv, definition)
        sampler = BooguImageSampler(pipeline)

        captured: dict = {}

        def fake_base(self, cfg, step):
            captured.update(cfg)
            return MagicMock()

        monkeypatch.setattr(GenericSamplingPipeline, "_sample_single", fake_base)
        sampler._sample_single({"prompt": "x"}, 0)

        assert captured["num_inference_steps"] == 50
        assert captured["guidance_scale"] == 4.0
        assert captured["width"] == 1024
        assert captured["height"] == 1024

    def test_turbo_fills_4_steps_1_guidance_1024(self, monkeypatch):
        from app.engine.core.sampling import GenericSamplingPipeline
        from app.engine.models.families.boogu_image.sampler import BooguImageSampler

        model = _ConstantModel()
        definition = _definition(True)
        drv = _driver_with_model(model, definition)
        pipeline = _build_pipeline(drv, definition)
        sampler = BooguImageSampler(pipeline)

        captured: dict = {}

        def fake_base(self, cfg, step):
            captured.update(cfg)
            return MagicMock()

        monkeypatch.setattr(GenericSamplingPipeline, "_sample_single", fake_base)
        sampler._sample_single({"prompt": "x"}, 0)

        assert captured["num_inference_steps"] == 4
        assert captured["guidance_scale"] == 1.0
        assert captured["width"] == 1024
        assert captured["height"] == 1024

    def test_explicit_values_respected(self, monkeypatch):
        from app.engine.core.sampling import GenericSamplingPipeline
        from app.engine.models.families.boogu_image.sampler import BooguImageSampler

        model = _ConstantModel()
        definition = _definition(False)
        drv = _driver_with_model(model, definition)
        pipeline = _build_pipeline(drv, definition)
        sampler = BooguImageSampler(pipeline)

        captured: dict = {}

        def fake_base(self, cfg, step):
            captured.update(cfg)
            return MagicMock()

        monkeypatch.setattr(GenericSamplingPipeline, "_sample_single", fake_base)
        sampler._sample_single(
            {"prompt": "x", "width": 512, "height": 512,
             "num_inference_steps": 9, "guidance_scale": 2.0},
            0,
        )
        assert captured["num_inference_steps"] == 9
        assert captured["guidance_scale"] == 2.0
        assert captured["width"] == 512
        assert captured["height"] == 512


# ── Turbo loop: ladder + hard assert + renoise math ──────────────────────


class TestTurboLadder:
    def test_ladder_matches_linspace_conditioning_sigma_to_1_exclusive(self):
        """linspace(0.001, 1.0, steps+1)[:-1] -- pipeline_boogu_turbo.py:66-72,
        default conditioning_sigma=0.001 (__call__ default at :128)."""
        from app.engine.models.families.boogu_image.sampler import _build_dmd_sigmas

        for n_steps in (1, 4, 8):
            sigmas = _build_dmd_sigmas(n_steps, torch.device("cpu"), torch.float32, 0.001)
            expected = torch.linspace(0.001, 1.0, n_steps + 1)[:-1]
            assert sigmas.shape == (n_steps,)
            assert torch.allclose(sigmas, expected, atol=1e-6)
            assert torch.all(sigmas < 1.0)


class TestTurboHardAssert:
    def test_guidance_not_1_raises(self):
        from app.engine.models.families.boogu_image.sampler import BooguImageSampler

        model = _ConstantModel()
        definition = _definition(True)
        drv = _driver_with_model(model, definition)
        pipeline = _build_pipeline(drv, definition)
        sampler = BooguImageSampler(pipeline)
        prompt_emb = sampler.encode_prompt("cat")
        noise = torch.randn(1, TINY_IN_CHANNELS, 4, 4)

        with pytest.raises(ValueError, match="1.0"):
            sampler.denoise(noise, prompt_emb, num_steps=4, guidance_scale=2.0, seed=0)


class TestTurboRenoiseMath:
    def test_renoise_matches_hand_computed_step(self):
        """x0_hat = latents + (1-sigma)*model_pred (pipeline_boogu_turbo.py
        :74-98); next latents = (1-sigma_next)*FRESH_noise + sigma_next*x0_hat
        (:100-118). A controlled constant-output fake model + a
        seed-replayed generator lets us hand-reproduce both steps exactly."""
        from app.engine.models.families.boogu_image.sampler import (
            BooguImageSampler,
            _build_dmd_sigmas,
        )

        model = _ConstantModel()
        definition = _definition(True)
        drv = _driver_with_model(model, definition)
        pipeline = _build_pipeline(drv, definition)
        sampler = BooguImageSampler(pipeline)

        # cond embeds mean == 3.0 -> model always predicts velocity == 3.0.
        prompt_emb = {
            "embeds": torch.full((1, 3, TINY_TEXT_DIM), 3.0),
            "mask": torch.ones(1, 3, dtype=torch.long),
        }
        noise0 = torch.zeros(1, TINY_IN_CHANNELS, 2, 2)
        seed = 123

        latents = sampler.denoise(
            noise0.clone(), prompt_emb, num_steps=2, guidance_scale=1.0, seed=seed,
        )

        # Hand-replay: same sigma ladder, same model output (constant 3.0),
        # same generator seed/sequence.
        sigmas = _build_dmd_sigmas(2, torch.device("cpu"), torch.float32, 0.001).tolist()
        gen = torch.Generator(device="cpu").manual_seed(seed)

        z = noise0.clone()
        v = 3.0
        # Step 0: predict -> x0_hat; renoise (not last step).
        x0_hat_0 = z + (1.0 - sigmas[0]) * v
        fresh_noise_0 = torch.randn(z.shape, generator=gen, device="cpu", dtype=torch.float32)
        z = (1.0 - sigmas[1]) * fresh_noise_0 + sigmas[1] * x0_hat_0
        # Step 1 (last): predict -> x0_hat, NO renoise.
        x0_hat_1 = z + (1.0 - sigmas[1]) * v

        assert torch.allclose(latents, x0_hat_1, atol=1e-5)


class TestTurboOracleConsistency:
    def test_perfect_one_step_generator_yields_x0_after_walk(self):
        """A 'perfect' DMD generator (see _PerfectDmdGenerator's derivation)
        must reproduce x0 EXACTLY after the full N-step walk, regardless of
        the fresh noise drawn at each intermediate renoise -- this is the
        DMD analogue of the Base loop's constant-velocity round trip."""
        from app.engine.models.families.boogu_image.sampler import BooguImageSampler

        torch.manual_seed(3)
        x0_true = torch.randn(1, TINY_IN_CHANNELS, 3, 3)
        model = _PerfectDmdGenerator(x0_true)
        definition = _definition(True)
        drv = _driver_with_model(model, definition)
        pipeline = _build_pipeline(drv, definition)
        sampler = BooguImageSampler(pipeline)
        prompt_emb = sampler.encode_prompt("cat")

        noise0 = torch.randn(1, TINY_IN_CHANNELS, 3, 3)
        latents = sampler.denoise(
            noise0, prompt_emb, num_steps=4, guidance_scale=1.0, seed=7,
        )
        assert torch.allclose(latents, x0_true, atol=1e-4)


# ── Precision contract ────────────────────────────────────────────────────


class TestPrecisionContract:
    def test_no_autocast_in_denoise_source(self):
        """Check the actual loop bodies (not the module docstring, which
        legitimately discusses the autocast-collapse gotcha in prose)."""
        from app.engine.models.families.boogu_image.sampler import BooguImageSampler

        for method in (
            BooguImageSampler.denoise,
            BooguImageSampler._denoise_base,
            BooguImageSampler._denoise_turbo,
        ):
            source = inspect.getsource(method)
            non_comment = "\n".join(
                line for line in source.splitlines()
                if not line.strip().startswith("#")
            )
            assert "torch.autocast" not in non_comment, method.__name__

    def test_base_trajectory_stays_fp32_with_bf16_ish_model(self):
        """Model 'weights' dtype != fp32 (bfloat16); trajectory tensor
        returned by denoise() must still be fp32 (no drift to model dtype)."""
        from app.engine.models.families.boogu_image.sampler import BooguImageSampler

        model = _ConstantModel(dtype=torch.bfloat16)
        definition = _definition(False)
        drv = _driver_with_model(model, definition)
        drv.scheduler = _make_scheduler()
        pipeline = _build_pipeline(drv, definition)
        sampler = BooguImageSampler(pipeline)
        prompt_emb = sampler.encode_prompt("cat")

        noise = torch.randn(1, TINY_IN_CHANNELS, 4, 4)
        latents = sampler.denoise(noise, prompt_emb, num_steps=2, guidance_scale=1.0, seed=0)
        assert latents.dtype == torch.float32

    def test_cached_fp32_embeds_cast_to_model_dtype_before_forward(self):
        """Cached text embeddings are stored fp32 on disk/memory (T5 cache
        contract); the sampler must cast them to the model's own dtype
        before the forward, mirroring the driver's cached-embed dtype cast
        (fp32 cache vs bf16 model crash otherwise)."""
        from app.engine.models.families.boogu_image.sampler import BooguImageSampler

        model = _ConstantModel(dtype=torch.bfloat16)
        definition = _definition(False)
        drv = _driver_with_model(model, definition)
        drv.scheduler = _make_scheduler()
        pipeline = _build_pipeline(drv, definition)
        sampler = BooguImageSampler(pipeline)

        fp32_embeds = torch.full((1, 3, TINY_TEXT_DIM), 1.0, dtype=torch.float32)
        prompt_emb = {"embeds": fp32_embeds, "mask": torch.ones(1, 3, dtype=torch.long)}

        noise = torch.randn(1, TINY_IN_CHANNELS, 4, 4)
        sampler.denoise(noise, prompt_emb, num_steps=1, guidance_scale=1.0, seed=0)

        assert model.last_kwargs["instruction_hidden_states"].dtype == torch.bfloat16


# ── VAE decode ─────────────────────────────────────────────────────────────


class TestVaeDecode:
    def test_decode_unscales_then_unshifts_matching_hand_math(self):
        from app.engine.models.families.boogu_image.sampler import BooguImageSampler

        model = _ConstantModel()
        definition = _definition(False)
        drv = _driver_with_model(model, definition)
        pipeline = _build_pipeline(drv, definition)
        sampler = BooguImageSampler(pipeline)

        latents = torch.full((1, TINY_IN_CHANNELS, 4, 4), 0.7222)
        sampler.decode_latents(latents)

        scaling_factor = 0.3611
        shift_factor = 0.1159
        expected = latents / scaling_factor + shift_factor
        actual = pipeline.vae._captured["latents"]
        assert torch.allclose(actual, expected, atol=1e-5)

    def test_decode_latents_returns_pil_image(self):
        from PIL import Image

        from app.engine.models.families.boogu_image.sampler import BooguImageSampler

        model = _ConstantModel()
        definition = _definition(False)
        drv = _driver_with_model(model, definition)
        pipeline = _build_pipeline(drv, definition)
        sampler = BooguImageSampler(pipeline)

        latents = torch.randn(1, TINY_IN_CHANNELS, 4, 4)
        result = sampler.decode_latents(latents)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"


# ── _create_initial_noise ────────────────────────────────────────────────


class TestCreateInitialNoise:
    def test_shape_matches_vae_scale_factor_division(self):
        from app.engine.models.families.boogu_image.sampler import BooguImageSampler

        model = _ConstantModel()
        definition = _definition(False)
        drv = _driver_with_model(model, definition)
        pipeline = _build_pipeline(drv, definition)
        sampler = BooguImageSampler(pipeline)

        gen = torch.Generator().manual_seed(1)
        noise = sampler._create_initial_noise(64, 32, gen)
        assert noise.shape == (1, TINY_IN_CHANNELS, 4, 8)  # h//8, w//8
        assert noise.dtype == torch.float32


# ── trainer wiring (_create_sampler) ─────────────────────────────────────


class TestCreateSamplerWiring:
    def test_create_sampler_returns_boogu_sampler_when_enabled(self):
        from app.engine.models.families.boogu_image.sampler import BooguImageSampler
        from app.engine.models.families.boogu_image.trainer import BooguImageTrainer

        trainer = MagicMock(spec=BooguImageTrainer)
        trainer.config = {"sample_every_n_steps": 50}
        trainer.device = torch.device("cpu")
        trainer.definition = MagicMock()

        sampler = BooguImageTrainer._create_sampler(trainer)
        assert isinstance(sampler, BooguImageSampler)

    def test_create_sampler_returns_none_when_disabled(self):
        from app.engine.models.families.boogu_image.trainer import BooguImageTrainer

        trainer = MagicMock(spec=BooguImageTrainer)
        trainer.config = {"sample_every_n_steps": 0}
        trainer.device = torch.device("cpu")
        trainer.definition = MagicMock()

        assert BooguImageTrainer._create_sampler(trainer) is None
