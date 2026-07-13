"""Tests for OmniGen2Sampler / OmniGen2EditSampler — guidance semantics +
precision contract.

Invariants (pipeline_omnigen2.py at the pinned vendor/REVISION):
  1. denoise uses the LOADER-provided vendored scheduler (fail-loud when
     missing), set_timesteps(num_tokens = lat_H * lat_W).
  2. Guidance branch structure (L672-723):
     - text_g <= 1              -> 1 forward/step
     - text_g > 1, no control   -> 2 forwards/step, uncond has batch={}
     - text_g > 1, img_g > 1,
       control present          -> 3 forwards/step; combine
       uncond + img_g*(ref - uncond) + text_g*(cond - ref); the ref pass
       gets NEGATIVE embeds WITH the control batch; the uncond pass gets
       NEGATIVE embeds with batch={}.
  3. The negative prompt routes through the SAME trainer.encode_text path
     (chat template applied — pipeline L413-418), honoring
     sample_negative_prompt.
  4. NO torch.autocast anywhere in the denoise path (autocast-collapse
     gotcha); fp32 trajectory.
  5. decode: latents / scaling_factor + shift_factor -> vae.decode.
  6. Edit: control latents are VAE-encoded once per denoise and fed via
     _forward_batch; no control -> plain T2I fallback (no crash).
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import torch


_LAT_C = 4
_W = _H = 32          # -> 4x4 latents at vae_sf 8
_TXT_SEQ = 6
_TEXT_DIM = 12

_ARCH = {
    "vae.latent_channels": _LAT_C,
    "vae.vae_scale_factor": 8,
    "scheduler.num_train_timesteps": 1000,
    "scheduler.dynamic_time_shift": True,
}


def _make_vendored_scheduler():
    from app.engine.models.families.omnigen2.vendor.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )

    return FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, dynamic_time_shift=True,
    )


class _ForwardSpy:
    """Stands in for driver.forward_pass; records (embeds_id, batch_keys)."""

    def __init__(self):
        self.calls: list[tuple[str, bool]] = []
        self.pos_marker = None
        self.neg_marker = None

    def __call__(self, noisy_input, timesteps, text_embeddings, batch):
        emb = text_embeddings[0]
        kind = "pos" if emb is self.pos_marker or torch.equal(emb, self.pos_marker) else "neg"
        self.calls.append((kind, bool(batch.get("control_latents"))))
        # Deterministic distinct outputs per branch so combines are checkable.
        if kind == "pos" and batch.get("control_latents"):
            return torch.full_like(noisy_input, 4.0)
        if kind == "pos":
            return torch.full_like(noisy_input, 3.0)
        if batch.get("control_latents"):
            return torch.full_like(noisy_input, 2.0)
        return torch.full_like(noisy_input, 1.0)


def _build_pipeline_mock(spy: _ForwardSpy):
    pipeline = MagicMock()
    pipeline.device = torch.device("cpu")

    model = MagicMock()
    model.parameters = lambda: iter([torch.zeros(1, dtype=torch.float32)])
    model.to = lambda *a, **k: model
    pipeline.transformer = model
    pipeline._block_swap_managers = None

    driver = MagicMock()
    driver.scheduler = _make_vendored_scheduler()
    driver.forward_pass = spy
    pipeline.driver = driver

    pos_emb = torch.randn(1, _TXT_SEQ, _TEXT_DIM)
    neg_emb = torch.randn(1, _TXT_SEQ, _TEXT_DIM)
    spy.pos_marker = pos_emb
    spy.neg_marker = neg_emb

    encoded_prompts: list[str] = []

    def _encode_text(prompts, dtype=None):
        encoded_prompts.append(prompts[0])
        emb = neg_emb if len(encoded_prompts) > 1 else pos_emb
        return emb, torch.ones(1, _TXT_SEQ, dtype=torch.long)

    pipeline.encode_text = _encode_text
    pipeline._encoded_prompts = encoded_prompts

    vae = MagicMock()
    vae.parameters = lambda: iter([torch.zeros(1, dtype=torch.float32)])
    vae.config = MagicMock()
    vae.config.scaling_factor = 0.3611
    vae.config.shift_factor = 0.1159
    pipeline.vae = vae

    defn = MagicMock()
    defn.architecture_params = dict(_ARCH)
    defn.defaults = {
        "resolution": 1024,
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "image_guidance_scale": 2.0,
    }
    pipeline.definition = defn
    pipeline.config = {"sample_every_n_steps": 50, "sample_negative_prompt": ""}
    pipeline.components = {}
    return pipeline


def _build_sampler(spy=None):
    from app.engine.models.families.omnigen2.sampler import OmniGen2Sampler

    spy = spy or _ForwardSpy()
    pipeline = _build_pipeline_mock(spy)
    return OmniGen2Sampler(pipeline), spy


def _build_edit_sampler(spy=None):
    from app.engine.models.families.omnigen2.sampler_edit import OmniGen2EditSampler

    spy = spy or _ForwardSpy()
    pipeline = _build_pipeline_mock(spy)
    return OmniGen2EditSampler(pipeline), spy


def _noise():
    torch.manual_seed(0)
    return torch.randn(1, _LAT_C, _H // 8, _W // 8)


def _prompt_embedding(sampler):
    return sampler.encode_prompt("a prompt")


# ── Scheduler wiring ─────────────────────────────────────────────────────────


class TestSchedulerWiring:
    def test_fail_loud_without_loader_scheduler(self):
        sampler, _ = _build_sampler()
        sampler.pipeline.driver.scheduler = None
        try:
            sampler.denoise(_noise(), _prompt_embedding(sampler), 4, 1.0, 42)
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "scheduler" in str(e)

    def test_set_timesteps_receives_latent_pixel_num_tokens(self):
        sampler, _ = _build_sampler()
        sched = sampler.pipeline.driver.scheduler
        seen = {}
        original = sched.set_timesteps

        def _spy(num_inference_steps=None, device=None, timesteps=None, num_tokens=None):
            seen["num_tokens"] = num_tokens
            return original(
                num_inference_steps=num_inference_steps, device=device,
                timesteps=timesteps, num_tokens=num_tokens,
            )

        sched.set_timesteps = _spy
        sampler.denoise(_noise(), _prompt_embedding(sampler), 4, 1.0, 42)
        assert seen["num_tokens"] == (_H // 8) * (_W // 8)


# ── Guidance branch structure ────────────────────────────────────────────────


class TestGuidanceBranches:
    def test_no_cfg_single_forward_per_step(self):
        sampler, spy = _build_sampler()
        sampler.denoise(_noise(), _prompt_embedding(sampler), 4, 1.0, 42)
        assert len(spy.calls) == 4
        assert all(kind == "pos" for kind, _ in spy.calls)

    def test_text_cfg_two_forwards_uncond_without_control(self):
        sampler, spy = _build_sampler()
        sampler.denoise(_noise(), _prompt_embedding(sampler), 4, 5.0, 42)
        assert len(spy.calls) == 8
        # Per step: pos then neg; neg NEVER carries control (T2I).
        for i in range(0, 8, 2):
            assert spy.calls[i] == ("pos", False)
            assert spy.calls[i + 1] == ("neg", False)

    def test_two_pass_combine_formula(self):
        """pred = uncond + text_g*(cond - uncond) — with cond=3, uncond=1
        and g=5: pred = 1 + 5*(3-1) = 11. Verified through one Euler step:
        prev = 0 + dt*11."""
        sampler, spy = _build_sampler()
        zeros = torch.zeros(1, _LAT_C, _H // 8, _W // 8)
        out = sampler.denoise(zeros, _prompt_embedding(sampler), 1, 5.0, 42)
        # Single step: dt spans the whole [t0, 1.0] interval.
        sched = sampler.pipeline.driver.scheduler
        t0 = float(sched.timesteps[0])
        expected = (1.0 - t0) * 11.0
        assert torch.allclose(out, torch.full_like(out, expected), atol=1e-5)

    def test_edit_three_pass_structure_and_combine(self):
        """Control + text_g>1 + img_g>1 -> 3 forwards/step: cond(pos, ctrl),
        ref(NEG, ctrl), uncond(NEG, no ctrl); combine
        uncond + img_g*(ref-uncond) + text_g*(cond-ref). With cond=4,
        ref=2, uncond=1, text_g=5, img_g=2: pred = 1 + 2*(2-1) + 5*(4-2)
        = 13."""
        sampler, spy = _build_edit_sampler()
        sampler._active_prompt_cfg = {"image_guidance_scale": 2.0}
        sampler._active_control_latents = None

        # Bypass file I/O: pre-seed the control encode by stubbing it.
        sampler._encode_control_latent = lambda p, w, h: torch.randn(
            1, _LAT_C, _H // 8, _W // 8,
        )
        sampler._resolve_control_paths = lambda: ["ctrl.png"]

        zeros = torch.zeros(1, _LAT_C, _H // 8, _W // 8)
        out = sampler.denoise(zeros, _prompt_embedding(sampler), 1, 5.0, 42)

        assert len(spy.calls) == 3
        assert spy.calls[0] == ("pos", True)   # cond WITH control
        assert spy.calls[1] == ("neg", True)   # ref-CFG: neg text, WITH control
        assert spy.calls[2] == ("neg", False)  # uncond: neg text, NO control

        sched = sampler.pipeline.driver.scheduler
        t0 = float(sched.timesteps[0])
        expected = (1.0 - t0) * 13.0
        assert torch.allclose(out, torch.full_like(out, expected), atol=1e-5)

    def test_edit_image_guidance_one_collapses_to_two_pass(self):
        sampler, spy = _build_edit_sampler()
        sampler._active_prompt_cfg = {"image_guidance_scale": 1.0}
        sampler._encode_control_latent = lambda p, w, h: torch.randn(
            1, _LAT_C, _H // 8, _W // 8,
        )
        sampler._resolve_control_paths = lambda: ["ctrl.png"]

        sampler.denoise(_noise(), _prompt_embedding(sampler), 2, 5.0, 42)
        assert len(spy.calls) == 4
        for i in range(0, 4, 2):
            assert spy.calls[i] == ("pos", True)    # cond keeps the control
            assert spy.calls[i + 1] == ("neg", False)  # uncond drops it

    def test_edit_without_control_falls_back_to_t2i(self):
        sampler, spy = _build_edit_sampler()
        sampler._active_prompt_cfg = {}
        sampler.denoise(_noise(), _prompt_embedding(sampler), 2, 5.0, 42)
        # Plain 2-pass T2I, no control anywhere.
        assert len(spy.calls) == 4
        assert all(has_ctrl is False for _, has_ctrl in spy.calls)


# ── Negative prompt path ─────────────────────────────────────────────────────


class TestNegativePrompt:
    def test_negative_routes_through_same_encode_text(self):
        """sample_negative_prompt is honored and chat-templated via the SAME
        trainer.encode_text path (no boogu-style hard-pinned negative)."""
        sampler, _ = _build_sampler()
        sampler.pipeline.config["sample_negative_prompt"] = "blurry, low quality"
        sampler.denoise(_noise(), _prompt_embedding(sampler), 1, 5.0, 42)
        assert sampler.pipeline._encoded_prompts == ["a prompt", "blurry, low quality"]

    def test_no_negative_encode_when_cfg_off(self):
        sampler, _ = _build_sampler()
        sampler.denoise(_noise(), _prompt_embedding(sampler), 1, 1.0, 42)
        assert sampler.pipeline._encoded_prompts == ["a prompt"]


# ── Image-guidance resolution order ──────────────────────────────────────────


class TestImageGuidanceResolution:
    def test_prompt_key_wins(self):
        sampler, _ = _build_edit_sampler()
        sampler._active_prompt_cfg = {"image_guidance_scale": 3.5}
        sampler.pipeline.config["sample_image_guidance_scale"] = 2.5
        assert sampler._resolve_image_guidance() == 3.5

    def test_config_beats_definition_default(self):
        sampler, _ = _build_edit_sampler()
        sampler._active_prompt_cfg = {}
        sampler.pipeline.config["sample_image_guidance_scale"] = 2.5
        assert sampler._resolve_image_guidance() == 2.5

    def test_definition_default_fallback(self):
        sampler, _ = _build_edit_sampler()
        sampler._active_prompt_cfg = {}
        assert sampler._resolve_image_guidance() == 2.0

    def test_t2i_sampler_pins_image_guidance_to_one(self):
        sampler, _ = _build_sampler()
        assert sampler._resolve_image_guidance() == 1.0


# ── Precision contract ───────────────────────────────────────────────────────


class TestPrecisionContract:
    def test_no_autocast_in_sampler_sources(self):
        from app.engine.models.families.omnigen2 import sampler, sampler_edit

        for mod in (sampler, sampler_edit):
            src = inspect.getsource(mod)
            # Match invocations, not the docstrings documenting the gotcha.
            assert "torch.autocast" not in src, (
                f"{mod.__name__} uses torch.autocast — the autocast-collapse gotcha"
            )
            assert "amp.autocast" not in src and "cuda.amp" not in src, (
                f"{mod.__name__} uses AMP autocast — the autocast-collapse gotcha"
            )

    def test_denoise_returns_fp32(self):
        sampler, _ = _build_sampler()
        out = sampler.denoise(_noise(), _prompt_embedding(sampler), 2, 1.0, 42)
        assert out.dtype == torch.float32

    def test_initial_noise_shape(self):
        sampler, _ = _build_sampler()
        gen = torch.Generator().manual_seed(0)
        n = sampler._create_initial_noise(_W, _H, gen)
        assert n.shape == (1, _LAT_C, _H // 8, _W // 8)
        assert n.dtype == torch.float32


# ── Decode ───────────────────────────────────────────────────────────────────


class TestDecode:
    def test_decode_order_divide_then_shift(self):
        """latents / scaling + shift (pipeline L739-744) — order matters."""
        sampler, _ = _build_sampler()
        seen = {}

        def _decode(latents, return_dict=False):
            seen["latents"] = latents.clone()
            return (torch.zeros(1, 3, _H, _W),)

        sampler.pipeline.vae.decode = _decode

        lat = torch.ones(1, _LAT_C, 4, 4)
        sampler.decode_latents(lat)
        expected = 1.0 / 0.3611 + 0.1159
        assert torch.allclose(
            seen["latents"], torch.full_like(lat, expected), atol=1e-5,
        )
