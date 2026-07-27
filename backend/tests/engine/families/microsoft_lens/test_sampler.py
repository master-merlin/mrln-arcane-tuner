"""microsoft_lens in-training sampler tests.

These exercise the sampler with a tiny DiT fixture + a fake VAE so the full
denoise→decode path runs on CPU without GPT-OSS or real weights. The text
encoder is bypassed by handing ``denoise`` a pre-built prompt-embedding dict
(the same structure ``encode_prompt`` produces).
"""
from types import SimpleNamespace

import torch
from PIL import Image

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.microsoft_lens.driver import MicrosoftLensDriver
from app.engine.models.families.microsoft_lens.sampler import (
    MicrosoftLensSampler,
    _calculate_shift,
)


def _defn():
    return ModelDefinition(
        id="microsoft-lens-base", family="microsoft_lens", name="Lens Base",
        defaults={}, components={},
    )


def _tiny_dit():
    from app.engine.models.families.microsoft_lens.vendor.transformer import (
        LensTransformer2DModel,
    )
    return LensTransformer2DModel(
        patch_size=2, in_channels=128, out_channels=32, num_layers=1,
        attention_head_dim=8, num_attention_heads=2, inner_dim=16,
        enc_hidden_dim=2880, axes_dims_rope=(2, 2, 4),
        gate_mlp=True, rms_norm=True, multi_layer_encoder_feature=True,
        selected_layer_index=(5, 11, 17, 23),
    )


class _FakeBN:
    running_mean = torch.full((128,), 2.0)
    running_var = torch.full((128,), 4.0)
    eps = 1e-5


class _FakeVAE:
    """Stands in for AutoencoderKLFlux2: BN stats + a decode that maps
    [B,32,H,W] -> [B,3,H*8,W*8] (spatial upscale by 8, like the real VAE)."""

    bn = _FakeBN()
    dtype = torch.float32

    def decode(self, latents, return_dict=False):
        b, _, h, w = latents.shape
        img = torch.zeros((b, 3, h * 8, w * 8), dtype=latents.dtype)
        return (img,)


def _make_sampler(device=torch.device("cpu")):
    dit = _tiny_dit().eval()
    vae = _FakeVAE()
    drv = MicrosoftLensDriver(_defn(), device)
    drv.transformer = dit
    drv.vae = vae

    # W5.T10: denoise() now encodes the CFG unconditional ("") embedding
    # LAZILY, only when guidance_scale > 1 — via a real driver.encode_text()
    # call, not a value threaded through prompt_embedding. Stub it so the
    # do_cfg=True test path below doesn't need a real GPT-OSS text encoder.
    def _stub_encode_text(captions, dtype, s_txt=5):
        from app.engine.core.text_encoding import TextEncoderOutput

        return TextEncoderOutput(
            embeddings=torch.randn(
                len(captions), 4, s_txt, 2880, dtype=dtype, device=device
            ),
            attention_mask=torch.ones(
                len(captions), s_txt, dtype=torch.bool, device=device
            ),
        )

    drv.encode_text = _stub_encode_text
    pipeline = SimpleNamespace(
        config={},
        device=device,
        driver=drv,
        transformer=dit,
        vae=vae,
        components={},
        _block_swap_managers=None,
    )
    return MicrosoftLensSampler(pipeline)


def _fake_prompt_embedding(device, dtype, s_txt=5):
    """Build the dict encode_prompt would return.

    Only ``cond`` is needed — the CFG unconditional embedding is no longer
    threaded through prompt_embedding; denoise() encodes it lazily itself
    (via driver.encode_text, stubbed in _make_sampler above) only when
    guidance_scale > 1 actually engages CFG."""

    def _pair():
        stacked = torch.randn(1, 4, s_txt, 2880, dtype=dtype, device=device)
        mask = torch.ones(1, s_txt, dtype=torch.bool, device=device)
        return stacked, mask

    return {"cond": _pair()}


# ── scheduler / shift math ───────────────────────────────────────────────

def test_calculate_shift_endpoints():
    # At base_image_seq_len -> base_shift; at max_image_seq_len -> max_shift.
    assert abs(_calculate_shift(256) - 0.5) < 1e-6
    assert abs(_calculate_shift(4096) - 1.15) < 1e-6


def test_scheduler_has_lens_config_and_step_count():
    s = _make_sampler()
    sched = s._get_scheduler()
    assert sched.config.shift == 3.0
    assert sched.config.use_dynamic_shifting is True
    # Full Lens-Base scheduler parity (matches scheduler_config.json).
    assert sched.config.time_shift_type == "exponential"
    assert sched.config.base_shift == 0.5 and sched.config.max_shift == 1.15
    assert sched.config.base_image_seq_len == 256
    assert sched.config.max_image_seq_len == 4096
    sched.set_timesteps(num_inference_steps=4, mu=_calculate_shift(16))
    assert len(sched.timesteps) == 4


# ── noise creation ───────────────────────────────────────────────────────

def test_create_initial_noise_shape_and_grid():
    s = _make_sampler()
    gen = torch.Generator(device="cpu").manual_seed(0)
    # 64px -> VAE/8 = 8 latent px -> /2 patchify = 4x4 grid -> S=16, 128ch.
    noise = s._create_initial_noise(64, 64, gen)
    assert noise.shape == (1, 16, 128)
    assert (s._latent_h, s._latent_w) == (4, 4)


def test_create_initial_noise_nonsquare():
    s = _make_sampler()
    gen = torch.Generator(device="cpu").manual_seed(0)
    # 64 wide x 32 tall -> latent 8x4 -> grid 4x2 -> S=8.
    noise = s._create_initial_noise(64, 32, gen)
    assert noise.shape == (1, 8, 128)
    assert (s._latent_h, s._latent_w) == (2, 4)  # (h_grid, w_grid)


# ── full denoise loop (tiny DiT) ─────────────────────────────────────────

def test_denoise_returns_unpatchified_latent():
    s = _make_sampler()
    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = s._create_initial_noise(64, 64, gen)  # S=16, grid 4x4
    emb = _fake_prompt_embedding(s.device, torch.float32)
    out = s.denoise(noise, emb, num_steps=2, guidance_scale=4.0, seed=0)
    # unpatchified: [1, 32, h_grid*2, w_grid*2] = [1, 32, 8, 8]
    assert out.shape == (1, 32, 8, 8)


def test_denoise_without_cfg_runs():
    s = _make_sampler()
    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = s._create_initial_noise(32, 32, gen)  # S=4, grid 2x2
    emb = _fake_prompt_embedding(s.device, torch.float32)
    out = s.denoise(noise, emb, num_steps=2, guidance_scale=1.0, seed=0)
    assert out.shape == (1, 32, 4, 4)


def test_denoise_skips_uncond_encode_when_cfg_off():
    """W5.T10: guidance_scale <= 1 (CFG off) must NOT pay for an
    unconditional ("") text-encoder forward at all — it used to be encoded
    unconditionally inside encode_prompt() every sampling round."""
    s = _make_sampler()
    calls: list[list[str]] = []
    real_encode = s.pipeline.driver.encode_text

    def _counting_encode_text(captions, dtype):
        calls.append(list(captions))
        return real_encode(captions, dtype)

    s.pipeline.driver.encode_text = _counting_encode_text

    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = s._create_initial_noise(32, 32, gen)
    emb = _fake_prompt_embedding(s.device, torch.float32)
    s.denoise(noise, emb, num_steps=2, guidance_scale=1.0, seed=0)

    assert calls == []  # cond comes from prompt_embedding, no live encode at all


def test_denoise_encodes_uncond_lazily_when_cfg_on():
    """The mirror case: guidance_scale > 1 must trigger EXACTLY one
    encode_text(['']) call for the unconditional pass — the lazy
    counterpart of the old unconditional encode_prompt()-time call."""
    s = _make_sampler()
    calls: list[list[str]] = []
    real_encode = s.pipeline.driver.encode_text

    def _counting_encode_text(captions, dtype):
        calls.append(list(captions))
        return real_encode(captions, dtype)

    s.pipeline.driver.encode_text = _counting_encode_text

    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = s._create_initial_noise(32, 32, gen)
    emb = _fake_prompt_embedding(s.device, torch.float32)
    s.denoise(noise, emb, num_steps=2, guidance_scale=4.0, seed=0)

    assert calls == [[""]]


def test_denoise_uncond_encode_is_bracketed_around_real_encode_text():
    """Regression for the W5 blocking bug: the T10 lazy uncond encode in
    denoise() calls ``driver.encode_text`` DIRECTLY, exactly like
    production. ``MicrosoftLensDriver.encode_text`` moves its inputs to
    ``self.device`` and forwards ``self.text_encoder`` — if denoise() ever
    calls it WITHOUT re-bracketing the text encoder to the sampling device
    first, Phase 1 of ``_sample_single`` (``needs_live_te``) has already
    offloaded the TE back to CPU by the time denoise() runs, and this call
    raises a device-mismatch RuntimeError with guidance_scale > 1 (the
    3.5 default) — silently swallowed by pipeline_train.py's per-round
    try/except, producing zero preview images for the whole run.

    Unlike every other sampler test in this file, this one does NOT stub
    ``driver.encode_text`` — it runs the real driver method against a fake
    (but real ``nn.Module``) text encoder + tokenizer, so a regression that
    drops the bracket is caught by the bracket's ABSENCE, not hidden behind
    a stub that never touches device residency at all. It doesn't need a
    real CUDA device to catch the regression: it spies on
    ``_ensure_on_gpu``/``_offload_to_cpu`` and asserts the real encode call
    lands strictly between them.
    """
    device = torch.device("cpu")
    defn = ModelDefinition(
        id="microsoft-lens-base",
        family="microsoft_lens",
        name="Lens Base",
        defaults={},
        components={},
        # Offset 0 so the fake tokenizer's short sequence isn't dropped to
        # zero length by drop_txt_offset (irrelevant to the bug — keeps the
        # DiT forward well-formed).
        architecture_params={"te.txt_offset": 0},
    )
    drv = MicrosoftLensDriver(defn, device)
    drv.transformer = _tiny_dit().eval()
    drv.vae = _FakeVAE()

    class _FakeTokenizer:
        def apply_chat_template(
            self,
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        ):
            return "rendered prompt<|return|>tail"

        def __call__(
            self,
            texts,
            padding=True,
            truncation=True,
            max_length=None,
            return_tensors="pt",
            add_special_tokens=True,
        ):
            n, seq_len = len(texts), 5
            return {
                "input_ids": torch.zeros(n, seq_len, dtype=torch.long),
                "attention_mask": torch.ones(n, seq_len, dtype=torch.long),
            }

    class _FakeTextEncoderModule(torch.nn.Module):
        """A REAL nn.Module (unlike the stubs above) so encode_text's
        ``next(self.text_encoder.parameters()).device`` residency check has
        something genuine to inspect."""

        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        def forward(
            self,
            input_ids,
            attention_mask,
            output_hidden_states=True,
            use_cache=False,
        ):
            b, s_txt = input_ids.shape
            hidden_states = tuple(torch.zeros(b, s_txt, 2880) for _ in range(25))
            return SimpleNamespace(hidden_states=hidden_states)

    drv.tokenizer = _FakeTokenizer()
    drv.text_encoder = _FakeTextEncoderModule().eval()

    pipeline = SimpleNamespace(
        config={},
        device=device,
        driver=drv,
        transformer=drv.transformer,
        vae=drv.vae,
        components={},
        _block_swap_managers=None,
    )
    s = MicrosoftLensSampler(pipeline)

    events: list[str] = []
    real_ensure_on_gpu = s._ensure_on_gpu
    real_offload_to_cpu = s._offload_to_cpu
    real_encode_text = drv.encode_text

    def _spy_ensure_on_gpu(names):
        events.append(f"ensure:{names}")
        return real_ensure_on_gpu(names)

    def _spy_offload_to_cpu(names):
        events.append(f"offload:{names}")
        return real_offload_to_cpu(names)

    def _spy_encode_text(captions, dtype):
        events.append(f"encode:{captions}")
        return real_encode_text(captions, dtype)

    s._ensure_on_gpu = _spy_ensure_on_gpu
    s._offload_to_cpu = _spy_offload_to_cpu
    drv.encode_text = _spy_encode_text

    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = s._create_initial_noise(32, 32, gen)
    emb = _fake_prompt_embedding(s.device, torch.float32)

    # guidance_scale > 1 -> do_cfg=True -> the lazy uncond encode fires.
    s.denoise(noise, emb, num_steps=1, guidance_scale=4.0, seed=0)

    assert "encode:['']" in events, (
        f"the lazy uncond encode_text(['']) call never ran: {events}"
    )
    encode_idx = events.index("encode:['']")
    # _ensure_on_gpu only reports a component as "moved" (and _offload_to_cpu
    # only acts on it) when a real device move actually happened — on this
    # CPU-only host the text encoder is already CPU-resident, so both calls
    # fire with an empty moved-list. What must hold regardless of host is
    # that the bracket METHODS were invoked, straddling the real encode.
    ensure_calls = [i for i, e in enumerate(events) if e.startswith("ensure:")]
    offload_calls = [i for i, e in enumerate(events) if e.startswith("offload:")]
    assert ensure_calls, f"_ensure_on_gpu was never called around the encode: {events}"
    assert offload_calls, (
        f"_offload_to_cpu was never called around the encode: {events}"
    )
    assert ensure_calls[0] < encode_idx < offload_calls[-1], (
        "the real driver.encode_text call must run strictly between "
        f"_ensure_on_gpu and _offload_to_cpu — got order {events}"
    )
    # And it must have specifically requested the text encoder, not some
    # other component — a bracket around the wrong name would still pass a
    # bare "was something called" check.
    assert any("text_encoder" in events[i] for i in ensure_calls)


def test_denoise_bn_denormalizes_output():
    """With BN mean=2/std=2, denormalized latents should not be ~zero-mean."""
    s = _make_sampler()
    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = s._create_initial_noise(32, 32, gen)
    emb = _fake_prompt_embedding(s.device, torch.float32)
    out = s.denoise(noise, emb, num_steps=2, guidance_scale=1.0, seed=0)
    # bn_denormalize multiplies by std(=2) and adds mean(=2); the additive
    # mean shift makes the output mean clearly positive vs a ~0-mean latent.
    assert out.mean().item() > 0.5


# ── decode ───────────────────────────────────────────────────────────────

def test_decode_latents_returns_pil_image():
    s = _make_sampler()
    latents = torch.zeros(1, 32, 8, 8)
    img = s.decode_latents(latents)
    assert isinstance(img, Image.Image)
    assert img.size == (64, 64)  # 8*8 spatial upscale


# ── trainer wiring ───────────────────────────────────────────────────────

def test_trainer_creates_sampler_only_when_interval_positive():
    from app.engine.models.families.microsoft_lens.trainer import (
        MicrosoftLensTrainer,
    )

    trainer = MicrosoftLensTrainer.__new__(MicrosoftLensTrainer)
    trainer.device = torch.device("cpu")
    trainer.driver = MicrosoftLensDriver(_defn(), torch.device("cpu"))
    trainer.config = {"sample_every_n_steps": 0}
    assert trainer._create_sampler() is None

    trainer.config = {"sample_every_n_steps": 50}
    sampler = trainer._create_sampler()
    assert isinstance(sampler, MicrosoftLensSampler)


def test_assign_components_restores_offloaded_te_for_sampling():
    """A re-assignment after offload (TE popped from components) must not
    drop the sampling text encoder when sampling is enabled."""
    from app.engine.models.families.microsoft_lens.trainer import (
        MicrosoftLensTrainer,
    )

    trainer = MicrosoftLensTrainer.__new__(MicrosoftLensTrainer)
    trainer.config = {"sample_every_n_steps": 50, "unload_text_encoder": False}
    trainer.device = torch.device("cpu")
    trainer.driver = MicrosoftLensDriver(_defn(), torch.device("cpu"))
    # Components no longer hold the TE/tokenizer (offload popped them).
    trainer.components = {"unet": torch.nn.Linear(2, 2), "vae": torch.nn.Linear(2, 2)}
    # Strong refs captured at offload time.
    te = torch.nn.Linear(4, 4)
    tok = object()
    trainer._sampling_text_encoder = te
    trainer._sampling_tokenizer = tok

    trainer._assign_components()

    assert trainer.driver.text_encoder is te
    assert trainer.driver.tokenizer is tok
    assert trainer.text_encoder is te


def test_offload_captures_te_only_when_sampling_and_not_unloading():
    """The TE strong-ref is captured only when sampling is on and the TE is
    offloaded (not unloaded)."""
    from unittest import mock

    from app.engine.core.pipeline import GenericTrainingPipeline
    from app.engine.models.families.microsoft_lens.trainer import (
        MicrosoftLensTrainer,
    )

    def _make(cfg):
        t = MicrosoftLensTrainer.__new__(MicrosoftLensTrainer)
        t.config = cfg
        t.device = torch.device("cpu")
        t.driver = MicrosoftLensDriver(_defn(), torch.device("cpu"))
        t.driver.text_encoder = torch.nn.Linear(4, 4)
        t.driver.tokenizer = object()
        return t

    with mock.patch.object(
        GenericTrainingPipeline, "_offload_text_encoders", lambda self: None
    ):
        # sampling on + offloaded (not unloaded) -> captures
        t1 = _make({"sample_every_n_steps": 50, "unload_text_encoder": False})
        t1._offload_text_encoders()
        assert t1._sampling_text_encoder is t1.driver.text_encoder

        # unloading -> no capture (new-prompt sampling is impossible anyway)
        t2 = _make({"sample_every_n_steps": 50, "unload_text_encoder": True})
        t2._offload_text_encoders()
        assert getattr(t2, "_sampling_text_encoder", None) is None

        # sampling off -> no capture (avoid pinning the TE needlessly)
        t3 = _make({"sample_every_n_steps": 0})
        t3._offload_text_encoders()
        assert getattr(t3, "_sampling_text_encoder", None) is None


def test_assign_components_no_restore_when_sampling_disabled():
    """Without a captured sampling ref, _assign_components leaves TE as-is."""
    from app.engine.models.families.microsoft_lens.trainer import (
        MicrosoftLensTrainer,
    )

    trainer = MicrosoftLensTrainer.__new__(MicrosoftLensTrainer)
    trainer.config = {"sample_every_n_steps": 0}
    trainer.device = torch.device("cpu")
    trainer.driver = MicrosoftLensDriver(_defn(), torch.device("cpu"))
    trainer.components = {"unet": torch.nn.Linear(2, 2), "vae": torch.nn.Linear(2, 2)}

    trainer._assign_components()

    assert trainer.driver.text_encoder is None
