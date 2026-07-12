"""boogu_image Edit-variant tests (task A4 fix wave).

Covers the three review findings:

1. **VL-side control-image encoding** —
   ``BooguImageDriver.encode_text_with_images`` builds the upstream TI2I
   chat-template branch (image content entries BEFORE the instruction text,
   TI2I system prompt == the DROP text, upstream alias pipeline_boogu.py:
   231-237 / :1611-1620), forwards the processor's extra tensors
   (``pixel_values``/``image_grid_thw``) to the mllm, and downscales inputs
   per upstream's dataset-matching preprocessing.
   ``BooguImageEditTrainer.encode_text`` routes control batches through it
   under a composite ``(caption, control)`` cache key; the t2i path
   (base trainer / driver ``encode_text``) stays image-free (pinned).
2. **Edit-aware previews** — ``BooguImageEditSampler`` VAE-encodes the
   sample prompt's ``control_images`` and feeds them into every
   ``driver.forward_pass`` via the ``_forward_batch`` hook (shape-level,
   same style as the driver-wiring tests); the base sampler's hook stays
   ``{}`` (Base/Turbo byte-identical regression pin).
3. Family dispatch: ``control_inputs > 0`` -> ``BooguImageEditTrainer``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from PIL import Image

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.boogu_image.driver import (
    BooguImageDriver,
    _preprocess_vlm_image,
)
from app.engine.models.families.boogu_image.family import BooguImageFamily
from app.engine.models.families.boogu_image.trainer import BooguImageTrainer
from app.engine.models.families.boogu_image.trainer_edit import (
    BooguImageEditTrainer,
    composite_te_key,
    control_files_hash,
)
from app.engine.models.families.boogu_image.vendor.schedulers.scheduling_flow_match_euler_discrete_time_shifting import (
    FlowMatchEulerDiscreteScheduler,
)

TINY_AXES_DIM_ROPE = (2, 2, 4)
TINY_AXES_LENS = (64, 64, 64)
TINY_IN_CHANNELS = 4
TINY_TEXT_DIM = 8

_DROP_TEXT = (
    "Describe the key features of the input image (color, shape, size, "
    "texture, objects, background), then explain how the user's text "
    "instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency "
    "with the original input where appropriate."
)


# ── Shared fakes ──────────────────────────────────────────────────────────


class _FakeVlmOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _FakeTextEncoder(nn.Module):
    """Captures forward kwargs; emits layer-tagged hidden states."""

    def __init__(self, dim=TINY_TEXT_DIM):
        super().__init__()
        self.p = nn.Linear(1, 1)
        self.dim = dim
        self.captured: list[dict] = []

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        self.captured.append(
            {"input_ids": input_ids, "attention_mask": attention_mask, **kwargs},
        )
        B, L = input_ids.shape
        hs = tuple(torch.full((B, L, self.dim), float(i)) for i in range(3))
        return _FakeVlmOutput(hidden_states=hs)


class _FakeProcessor:
    """Captures prompts; emits image tensors when image entries are present."""

    def __init__(self, seq_len=6):
        self.seq_len = seq_len
        self.captured: list[dict] = []

    def apply_chat_template(self, prompts, **kwargs):
        self.captured.append({"prompts": prompts, "kwargs": kwargs})
        B = len(prompts)
        out = {
            "input_ids": torch.arange(B * self.seq_len).reshape(B, self.seq_len),
            "attention_mask": torch.ones(B, self.seq_len, dtype=torch.long),
        }
        has_images = any(
            entry.get("type") == "image"
            for msgs in prompts
            for m in msgs
            for entry in m["content"]
        )
        if has_images:
            out["pixel_values"] = torch.randn(B, 3, 16, 16)
            out["image_grid_thw"] = torch.ones(B, 3, dtype=torch.long)
        return out


def _definition(control_inputs: int = 0) -> MagicMock:
    d = MagicMock(spec=ModelDefinition)
    d.family = "boogu_image"
    d.id = "boogu-image-edit-test"
    d.control_inputs = control_inputs
    d.lora_targetable_modules = ["single_stream_layers.0.attn.to_q"]
    d.architecture_params = {
        "vae.vae_scale_factor": 8,
        "vae.latent_channels": TINY_IN_CHANNELS,
    }
    d.defaults = {
        "resolution": 1024,
        "is_distilled": False,
        "guidance_scale": 4.0,
        "num_inference_steps": 50,
    }
    return d


def _wired_driver(seq_len=6) -> tuple[BooguImageDriver, _FakeProcessor, _FakeTextEncoder]:
    drv = BooguImageDriver(_definition(1), torch.device("cpu"))
    processor = _FakeProcessor(seq_len=seq_len)
    te = _FakeTextEncoder()
    drv.processor = processor
    drv.text_encoder = te
    return drv, processor, te


def _control_png(tmp_path, name="ctl.png", size=(32, 32), color=(255, 0, 0)):
    path = tmp_path / name
    Image.new("RGB", size, color).save(path)
    return str(path)


def _user_content(processor: _FakeProcessor, call: int = -1, prompt: int = 0):
    msgs = processor.captured[call]["prompts"][prompt]
    return next(m for m in msgs if m["role"] == "user")["content"]


def _system_text(processor: _FakeProcessor, call: int = -1, prompt: int = 0):
    msgs = processor.captured[call]["prompts"][prompt]
    return next(m for m in msgs if m["role"] == "system")["content"][0]["text"]


# ── Finding 3 adjunct: family dispatch ────────────────────────────────────


class TestFamilyEditDispatch:
    def test_control_inputs_positive_dispatches_edit_trainer(self):
        fam = BooguImageFamily(_definition(control_inputs=1), config={})
        assert fam.get_trainer_class() is BooguImageEditTrainer

    def test_control_inputs_zero_dispatches_base_trainer(self):
        fam = BooguImageFamily(_definition(control_inputs=0), config={})
        assert fam.get_trainer_class() is BooguImageTrainer


# ── Finding 1: driver with-image encode ───────────────────────────────────


class TestDriverEncodeTextWithImages:
    def test_image_entries_before_text_under_ti2i_system_prompt(self):
        """Upstream _apply_chat_template with-image branch: images_content
        BEFORE user_text_content (:1614-1620), TI2I system prompt (== the
        DROP text — upstream alias :231-237)."""
        drv, processor, _ = _wired_driver()
        img = Image.new("RGB", (64, 64))

        drv.encode_text_with_images(["make it red"], [[img]], torch.float32)

        content = _user_content(processor)
        assert content[0]["type"] == "image"
        assert content[-1] == {"type": "text", "text": "make it red"}
        assert _system_text(processor) == _DROP_TEXT

    def test_processor_image_tensors_forwarded_to_mllm(self):
        """pixel_values / image_grid_thw must reach the text-encoder forward
        (the **vlm_inputs splat) — this is what makes the VL encoder actually
        attend the control pixels."""
        drv, _, te = _wired_driver()
        img = Image.new("RGB", (64, 64))

        drv.encode_text_with_images(["x"], [[img]], torch.float32)

        call = te.captured[-1]
        assert isinstance(call.get("pixel_values"), torch.Tensor)
        assert isinstance(call.get("image_grid_thw"), torch.Tensor)
        assert call["output_hidden_states"] is True

    def test_text_only_encode_text_has_no_image_entries(self):
        """t2i pin: the plain encode_text path (Base/Turbo) must stay
        image-free — no image content entries, no pixel_values."""
        drv, processor, te = _wired_driver()

        drv.encode_text(["draw a cat"], torch.float32)

        content = _user_content(processor)
        assert all(entry["type"] == "text" for entry in content)
        assert "pixel_values" not in te.captured[-1]

    def test_taps_last_hidden_layer_same_as_text_path(self):
        drv, _, _ = _wired_driver()
        out = drv.encode_text_with_images(
            ["x"], [[Image.new("RGB", (64, 64))]], torch.float32,
        )
        # Fake tags layer i with value i — last layer (index 2) -> all 2.0.
        assert torch.allclose(out.embeddings, torch.full_like(out.embeddings, 2.0))

    def test_empty_image_list_raises(self):
        drv, _, _ = _wired_driver()
        with pytest.raises(ValueError, match="at least one image"):
            drv.encode_text_with_images(["x"], [[]], torch.float32)

    def test_mismatched_lengths_raise(self):
        drv, _, _ = _wired_driver()
        with pytest.raises(ValueError, match="parallel"):
            drv.encode_text_with_images(["x", "y"], [[Image.new("RGB", (16, 16))]], torch.float32)

    def test_unwired_driver_raises_assign_components(self):
        drv = BooguImageDriver(_definition(1), torch.device("cpu"))
        with pytest.raises(RuntimeError, match="assign_components"):
            drv.encode_text_with_images(
                ["x"], [[Image.new("RGB", (16, 16))]], torch.float32,
            )


class TestVlmImagePreprocessing:
    def test_oversized_image_downscaled_under_max_pixels_16_multiple(self):
        """Upstream: scale so H*W <= 1024*1024, dims rounded DOWN to
        multiples of 16 (BooguImageProcessor vae_scale_factor*2)."""
        img = Image.new("RGB", (2048, 2048))
        out = _preprocess_vlm_image(img)
        assert out.size[0] * out.size[1] <= 1024 * 1024
        assert out.size[0] % 16 == 0 and out.size[1] % 16 == 0
        assert out.size == (1024, 1024)

    def test_small_image_dims_rounded_down_to_16_multiple_no_upscale(self):
        img = Image.new("RGB", (100, 60))
        out = _preprocess_vlm_image(img)
        assert out.size == (96, 48)

    def test_conforming_image_returned_unchanged(self):
        img = Image.new("RGB", (512, 256))
        assert _preprocess_vlm_image(img) is img


# ── Finding 1: edit trainer composite TE path ─────────────────────────────


def _edit_trainer_shell(drv: BooguImageDriver) -> BooguImageEditTrainer:
    """Real BooguImageEditTrainer, no heavy __init__ (house shell pattern —
    __init__-set attrs assigned manually)."""
    t = object.__new__(BooguImageEditTrainer)
    t.device = torch.device("cpu")
    t.definition = drv.definition
    t.driver = drv
    t.config = {"cache_text_embeddings": True}
    t.logger = MagicMock()
    t.text_cache = {}
    t.text_encoder = drv.text_encoder
    t._ctrl_hash_memo = {}
    t._warned_no_vl_processor = False
    return t


class TestEditTrainerEncodeText:
    def test_control_batch_routes_through_with_image_encode(self, tmp_path):
        drv, processor, te = _wired_driver()
        trainer = _edit_trainer_shell(drv)
        ctl = _control_png(tmp_path)

        emb, mask = trainer.encode_text(
            ["make it red"], torch.float32,
            batch={"control_paths": [[ctl]]},
        )

        # The encode carried an image entry (VL attends the control).
        content = _user_content(processor)
        assert content[0]["type"] == "image"
        assert isinstance(te.captured[-1].get("pixel_values"), torch.Tensor)
        assert emb.shape[0] == 1 and mask.shape[0] == 1

    def test_composite_key_distinguishes_controls(self, tmp_path):
        """Same caption + different control image -> distinct cache entries
        (the silent shared-embedding bug this key exists to prevent)."""
        drv, _, te = _wired_driver()
        trainer = _edit_trainer_shell(drv)
        ctl_a = _control_png(tmp_path, "a.png", color=(255, 0, 0))
        ctl_b = _control_png(tmp_path, "b.png", color=(0, 255, 0))

        trainer.encode_text(["same caption"], torch.float32,
                            batch={"control_paths": [[ctl_a]]})
        trainer.encode_text(["same caption"], torch.float32,
                            batch={"control_paths": [[ctl_b]]})

        assert len(trainer.text_cache) == 2
        assert all("||ctl:" in k for k in trainer.text_cache)
        assert len(te.captured) == 2  # two distinct encodes

    def test_cache_hit_skips_reencode(self, tmp_path):
        drv, _, te = _wired_driver()
        trainer = _edit_trainer_shell(drv)
        ctl = _control_png(tmp_path)
        batch = {"control_paths": [[ctl]]}

        trainer.encode_text(["cap"], torch.float32, batch=batch)
        trainer.encode_text(["cap"], torch.float32, batch=batch)

        assert len(te.captured) == 1
        assert len(trainer.text_cache) == 1

    def test_no_control_paths_falls_back_to_text_only(self):
        """Sampler negatives / partial batches: base text-only behavior."""
        drv, processor, _ = _wired_driver()
        trainer = _edit_trainer_shell(drv)
        trainer.config = {"cache_text_embeddings": False}

        trainer.encode_text(["plain t2i"], torch.float32, batch=None)

        content = _user_content(processor)
        assert all(entry["type"] == "text" for entry in content)

    def test_image_encode_failure_falls_back_text_only_with_one_warning(self, tmp_path):
        """Degraded mode (qwen precedent): image-specific failure -> text-only
        encode + ONE loud warning; control still conditions the transformer."""
        drv, processor, _ = _wired_driver()
        trainer = _edit_trainer_shell(drv)
        missing = str(tmp_path / "nonexistent.png")

        emb, mask = trainer.encode_text(
            ["cap"], torch.float32, batch={"control_paths": [[missing]]},
        )

        assert emb.shape[0] == 1
        # Fallback encode was text-only.
        content = _user_content(processor)
        assert all(entry["type"] == "text" for entry in content)
        trainer.logger.warning.assert_called_once()

    def test_composite_key_helpers(self, tmp_path):
        ctl = _control_png(tmp_path)
        h1 = control_files_hash([ctl])
        assert composite_te_key("cap", h1) == f"cap||ctl:{h1}"
        # Unreadable path degrades to path-string hash, never raises.
        h2 = control_files_hash([str(tmp_path / "missing.png")])
        assert len(h2) == 16 and h2 != h1


# ── Finding 2: edit-aware previews ────────────────────────────────────────


class _CaptureModel(nn.Module):
    """Fake transformer capturing ref_image_hidden_states per call."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            axes_dim_rope=TINY_AXES_DIM_ROPE, axes_lens=TINY_AXES_LENS,
        )
        self.dummy = nn.Parameter(torch.zeros(1))
        self.ref_calls: list = []

    def forward(self, hidden_states, **kwargs):
        self.ref_calls.append(kwargs.get("ref_image_hidden_states"))
        return [torch.zeros_like(h) for h in hidden_states]


def _make_scheduler(seq_len: int = 64) -> FlowMatchEulerDiscreteScheduler:
    return FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        do_shift=True,
        dynamic_time_shift=False,
        time_shift_version="v1",
        seq_len=seq_len,
    )


def _build_edit_vae() -> MagicMock:
    """Mock VAE with encode (latent_dist.mode) + Boogu config factors."""
    vae = MagicMock()
    vae.dtype = torch.float32
    vae.parameters = lambda: iter([torch.zeros(1)])
    vae.config = SimpleNamespace(scaling_factor=0.3611, shift_factor=0.1159)

    def _encode(pixels):
        b, _, h, w = pixels.shape
        latent = torch.ones(b, TINY_IN_CHANNELS, h // 8, w // 8)
        dist = SimpleNamespace(mode=lambda: latent)
        return SimpleNamespace(latent_dist=dist)

    vae.encode = _encode
    return vae


def _build_edit_sampler(model: _CaptureModel, definition=None):
    from app.engine.models.families.boogu_image.sampler_edit import (
        BooguImageEditSampler,
    )

    definition = definition or _definition(1)
    drv = BooguImageDriver(definition, torch.device("cpu"))
    drv.model = model
    drv.scheduler = _make_scheduler()

    pipeline = MagicMock()
    pipeline.device = torch.device("cpu")
    pipeline.transformer = model
    pipeline.driver = drv
    pipeline.vae = _build_edit_vae()
    pipeline.definition = definition
    pipeline.config = {"sample_every_n_steps": 10}
    pipeline._block_swap_managers = None

    def encode_text_fn(captions, dtype=None, batch=None):
        b = len(captions)
        emb = torch.zeros(b, 3, TINY_TEXT_DIM)
        mask = torch.ones(b, 3, dtype=torch.long)
        return emb.to(dtype or torch.float32), mask

    pipeline.encode_text = encode_text_fn
    return BooguImageEditSampler(pipeline)


class TestEditSamplerControlFeed:
    def test_control_latents_reach_every_forward(self, tmp_path):
        """Base CFG loop (guidance 4.0 -> cond + uncond forwards): every
        model call must receive the per-item ref-image list built from the
        VAE-encoded control at the target's latent grid."""
        model = _CaptureModel()
        sampler = _build_edit_sampler(model)
        ctl = _control_png(tmp_path, size=(32, 32))
        sampler._active_prompt_cfg = {"control_images": [ctl]}

        noise = torch.randn(1, TINY_IN_CHANNELS, 4, 4)
        prompt_embedding = {
            "embeds": torch.zeros(1, 3, TINY_TEXT_DIM),
            "mask": torch.ones(1, 3, dtype=torch.long),
        }
        sampler.denoise(noise, prompt_embedding, num_steps=2,
                        guidance_scale=4.0, seed=1)

        assert len(model.ref_calls) == 4  # 2 steps x (cond + uncond)
        for ref in model.ref_calls:
            assert ref is not None
            assert len(ref) == 1 and len(ref[0]) == 1
            assert ref[0][0].shape == (TINY_IN_CHANNELS, 4, 4)

    def test_no_control_image_falls_back_to_t2i(self):
        model = _CaptureModel()
        sampler = _build_edit_sampler(model)
        sampler._active_prompt_cfg = {}  # no control_images

        noise = torch.randn(1, TINY_IN_CHANNELS, 4, 4)
        prompt_embedding = {
            "embeds": torch.zeros(1, 3, TINY_TEXT_DIM),
            "mask": torch.ones(1, 3, dtype=torch.long),
        }
        sampler.denoise(noise, prompt_embedding, num_steps=1,
                        guidance_scale=1.0, seed=1)

        assert model.ref_calls and all(r is None for r in model.ref_calls)

    def test_control_state_cleared_after_denoise(self, tmp_path):
        model = _CaptureModel()
        sampler = _build_edit_sampler(model)
        ctl = _control_png(tmp_path)
        sampler._active_prompt_cfg = {"control_images": [ctl]}

        noise = torch.randn(1, TINY_IN_CHANNELS, 4, 4)
        prompt_embedding = {
            "embeds": torch.zeros(1, 3, TINY_TEXT_DIM),
            "mask": torch.ones(1, 3, dtype=torch.long),
        }
        sampler.denoise(noise, prompt_embedding, num_steps=1,
                        guidance_scale=1.0, seed=1)

        assert sampler._active_control_latents is None
        assert sampler._forward_batch() == {}

    def test_control_encode_uses_boogu_vae_order(self, tmp_path):
        """(z - shift_factor) * scaling_factor — upstream encode_vae order
        (pipeline_boogu.py:876-892), the inverse of this family's decode."""
        model = _CaptureModel()
        sampler = _build_edit_sampler(model)
        ctl = _control_png(tmp_path, size=(32, 32))

        latent = sampler._encode_control_latent(ctl, 32, 32)

        expected = (1.0 - 0.1159) * 0.3611  # mode() returns all-ones
        assert latent.shape == (1, TINY_IN_CHANNELS, 4, 4)
        assert torch.allclose(latent, torch.full_like(latent, expected), atol=1e-6)

    def test_turbo_loop_also_receives_control(self, tmp_path):
        """The hook covers _denoise_turbo too (same _forward_batch seam)."""
        defn = _definition(1)
        defn.defaults = {
            "resolution": 1024, "is_distilled": True,
            "guidance_scale": 1.0, "num_inference_steps": 4,
        }
        model = _CaptureModel()
        sampler = _build_edit_sampler(model, definition=defn)
        ctl = _control_png(tmp_path, size=(32, 32))
        sampler._active_prompt_cfg = {"control_images": [ctl]}

        noise = torch.randn(1, TINY_IN_CHANNELS, 4, 4)
        prompt_embedding = {
            "embeds": torch.zeros(1, 3, TINY_TEXT_DIM),
            "mask": torch.ones(1, 3, dtype=torch.long),
        }
        sampler.denoise(noise, prompt_embedding, num_steps=2,
                        guidance_scale=1.0, seed=1)

        assert len(model.ref_calls) == 2
        assert all(r is not None for r in model.ref_calls)


class TestBaseSamplerForwardBatchRegression:
    def test_base_sampler_forward_batch_is_empty_dict(self):
        """Base/Turbo (t2i) regression pin: the hook must return exactly {}
        — byte-identical behavior to the pre-A4 hardcoded batch={}."""
        from app.engine.models.families.boogu_image.sampler import BooguImageSampler

        model = _CaptureModel()
        drv = BooguImageDriver(_definition(0), torch.device("cpu"))
        drv.model = model
        pipeline = MagicMock()
        pipeline.device = torch.device("cpu")
        pipeline.driver = drv
        pipeline.definition = _definition(0)
        pipeline.config = {"sample_every_n_steps": 10}
        sampler = BooguImageSampler(pipeline)

        assert sampler._forward_batch() == {}
