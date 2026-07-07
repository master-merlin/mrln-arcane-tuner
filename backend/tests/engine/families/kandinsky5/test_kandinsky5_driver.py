"""Kandinsky 5.0 driver tests — tiny REAL transformer, no GPU.

Covers the family's load-bearing quirks:

- channels-LAST ⇄ channels-first transpose contract (both directions),
- cu_seqlens int32 construction (the padding-mask replacement),
- flow-match raw [0,1000] timestep contract (÷1000 exactly once, RAW timestep
  to the transformer),
- visual_cond zero-concat (T2V) + first-frame conditioning concat (I2V),
- i2v add_noise frame-0 clean,
- fully-indexed LoRA targets verified on a real tiny Kandinsky5Transformer3DModel
  (text_transformer_blocks must stay LoRA-free),
- RoPE pos + scale_factor helpers replicated from the pipeline.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from app.engine.core.text_encoding import TextEncoderOutput
from app.engine.models.families.kandinsky5.driver import (
    FLOWMATCH_SCALE,
    K5_LORA_TARGET_SUFFIXES,
    KANDINSKY5_CROP_START,
    KANDINSKY5_PROMPT_TEMPLATE,
    Kandinsky5Driver,
    build_cu_seqlens,
    get_scale_factor,
    prompt_clean,
    tiny_transformer_config,
    to_channels_first,
    to_channels_last,
)
from tests.engine.precision_contracts import assert_flowmatch_timestep_contract


def _definition(mode: str = "t2v", **arch_extra) -> MagicMock:
    d = MagicMock()
    d.family = "kandinsky5"
    d.id = "k5-test"
    d.lora_targetable_modules = []
    d.architecture_params = {
        "mode": mode,
        "transformer.in_visual_dim": 4,
        "transformer.num_visual_blocks": 1,
        "transformer.visual_cond": True,
        "video.vae_spatial": 8,
        **arch_extra,
    }
    return d


def _driver(mode: str = "t2v", **arch_extra) -> Kandinsky5Driver:
    return Kandinsky5Driver(_definition(mode, **arch_extra), torch.device("cpu"))


def _tiny_transformer(**overrides):
    from diffusers import Kandinsky5Transformer3DModel

    return Kandinsky5Transformer3DModel(**tiny_transformer_config(**overrides))


def _fake_text(batch: int, length: int = 6) -> TextEncoderOutput:
    return TextEncoderOutput(
        embeddings=torch.randn(batch, length, 16),
        attention_mask=build_cu_seqlens([length] * batch),
        pooled=torch.randn(batch, 8),
    )


# ── Transpose contract ─────────────────────────────────────────────────────


def test_channels_last_first_round_trip():
    x = torch.randn(2, 4, 3, 8, 8)  # [B, C, F, H, W]
    cl = to_channels_last(x)
    assert cl.shape == (2, 3, 8, 8, 4)  # [B, F, H, W, C]
    back = to_channels_first(cl)
    assert back.shape == x.shape
    assert torch.equal(back, x)


def test_transpose_moves_channel_content_not_just_shape():
    """Channel c of frame f must land at [..., c] of frame f (not a reshape)."""
    x = torch.zeros(1, 4, 2, 2, 2)
    x[0, 3, 1] = 7.0  # channel 3, frame 1
    cl = to_channels_last(x)
    assert torch.all(cl[0, 1, :, :, 3] == 7.0)
    assert cl[0, 0].abs().sum() == 0


def test_transpose_rejects_non_5d():
    with pytest.raises(ValueError):
        to_channels_last(torch.randn(2, 4, 8, 8))
    with pytest.raises(ValueError):
        to_channels_first(torch.randn(2, 4, 8, 8))


def test_prepare_latents_transposes_and_records_grid():
    drv = _driver()
    out = drv.prepare_latents(torch.randn(2, 4, 3, 8, 6))
    assert out.shape == (2, 3, 8, 6, 4)
    assert drv._latent_shape == (3, 8, 6)


def test_prepare_latents_lifts_4d_still_to_one_frame():
    drv = _driver()
    out = drv.prepare_latents(torch.randn(2, 4, 8, 6))
    assert out.shape == (2, 1, 8, 6, 4)
    assert drv._latent_shape == (1, 8, 6)


# ── cu_seqlens construction ────────────────────────────────────────────────


def test_build_cu_seqlens_matches_pipeline_math():
    """Replicates F.pad(cumsum(mask[:,129:].sum(1)), (1,0)).to(int32)."""
    cu = build_cu_seqlens(torch.tensor([3, 5, 2]))
    assert cu.dtype == torch.int32
    assert cu.tolist() == [0, 3, 8, 10]


def test_build_cu_seqlens_from_attention_mask():
    """End-to-end from a padded post-crop attention mask."""
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    cu = build_cu_seqlens(mask.sum(1))
    assert cu.tolist() == [0, 3, 8]
    assert cu.dtype == torch.int32


def test_build_cu_seqlens_single_caption():
    cu = build_cu_seqlens([7])
    assert cu.tolist() == [0, 7]


def test_text_rope_pos_is_arange_of_max_length():
    cu = build_cu_seqlens([3, 5])
    pos = Kandinsky5Driver.build_text_rope_pos(cu, "cpu")
    assert pos.tolist() == list(range(5))


# ── Flow-match timestep contract ───────────────────────────────────────────


def test_add_noise_flowmatch_contract():
    drv = _driver()
    assert_flowmatch_timestep_contract(
        drv.add_noise, shape=(2, 3, 4, 4, 4), scale=FLOWMATCH_SCALE
    )


def test_forward_receives_raw_timestep_not_divided():
    """The transformer must be conditioned on the RAW [0,1000] value —
    dividing again is the pure-noise-LoRA gotcha."""
    drv = _driver()
    captured = {}

    def fake_transformer(**kwargs):
        captured.update(kwargs)
        hs = kwargs["hidden_states"]
        return (hs[..., :4].clone(),)

    drv.transformer = fake_transformer
    drv.visual_cond = True
    latents = torch.randn(2, 4, 2, 8, 8)
    prepared = drv.prepare_latents(latents)
    t = torch.tensor([500.0, 999.0])
    drv.forward_pass(prepared, t, _fake_text(2), {})
    assert torch.equal(captured["timestep"], t)  # raw, NOT /1000


def test_add_noise_divides_exactly_once():
    drv = _driver()
    latents = torch.zeros(1, 2, 4, 4, 4)
    noise = torch.ones(1, 2, 4, 4, 4)
    noisy = drv.add_noise(latents, noise, torch.tensor([500.0]))
    # (500/1000) * 1 + (1 - 500/1000) * 0 = 0.5 — an extra /1000 would give 5e-4.
    assert torch.allclose(noisy, torch.full_like(noisy, 0.5))


# ── visual_cond concat (T2V zero-cond) ─────────────────────────────────────


def test_t2v_forward_concats_zero_cond_and_mask():
    """visual_cond=True checkpoints get [latents, zeros(C), zeros(1)] for T2V."""
    drv = _driver()
    drv.visual_cond = True
    captured = {}

    def fake_transformer(**kwargs):
        captured.update(kwargs)
        return (kwargs["hidden_states"][..., :4].clone(),)

    drv.transformer = fake_transformer
    prepared = drv.prepare_latents(torch.randn(2, 4, 2, 8, 8))
    out = drv.forward_pass(prepared, torch.tensor([10.0, 20.0]), _fake_text(2), {})

    hs = captured["hidden_states"]
    assert hs.shape == (2, 2, 8, 8, 9)  # C + C + 1 on the LAST dim
    assert torch.equal(hs[..., :4], prepared)
    assert hs[..., 4:].abs().sum() == 0  # zero cond + zero mask
    assert out.shape == (2, 2, 8, 8, 4)


def test_non_visual_cond_forward_passes_bare_latents():
    drv = _driver()
    drv.visual_cond = False
    captured = {}

    def fake_transformer(**kwargs):
        captured.update(kwargs)
        return (kwargs["hidden_states"].clone(),)

    drv.transformer = fake_transformer
    prepared = drv.prepare_latents(torch.randn(1, 4, 2, 8, 8))
    drv.forward_pass(prepared, torch.tensor([10.0]), _fake_text(1), {})
    assert captured["hidden_states"].shape == (1, 2, 8, 8, 4)


# ── I2V conditioning ───────────────────────────────────────────────────────


def _engaged_i2v_driver() -> Kandinsky5Driver:
    drv = _driver(mode="i2v")
    drv._i2v_active = True
    return drv


def test_i2v_attach_conditioning_stashes_clean_frame0():
    drv = _engaged_i2v_driver()
    batch: dict = {}
    latents = torch.randn(2, 4, 3, 8, 8)  # channels-FIRST raw latents
    drv.attach_conditioning(batch, latents)
    ff = batch[Kandinsky5Driver.BATCH_FIRST_FRAME_LATENT]
    assert ff.shape == (2, 4, 1, 8, 8)
    assert torch.equal(ff, latents[:, :, :1])


def test_t2v_attach_conditioning_is_noop():
    drv = _driver(mode="t2v")
    batch: dict = {}
    drv.attach_conditioning(batch, torch.randn(2, 4, 3, 8, 8))
    assert Kandinsky5Driver.BATCH_FIRST_FRAME_LATENT not in batch


def test_i2v_add_noise_keeps_frame0_clean():
    drv = _engaged_i2v_driver()
    latents_cf = torch.randn(2, 4, 3, 8, 8)
    prepared = drv.prepare_latents(latents_cf)  # engages (F=3 > 1)
    noise = torch.randn_like(prepared)
    noisy = drv.add_noise(prepared, noise, torch.tensor([700.0, 700.0]))
    assert torch.equal(noisy[:, :1], prepared[:, :1])  # frame 0 untouched
    assert not torch.equal(noisy[:, 1:], prepared[:, 1:])  # rest noised


def test_i2v_single_frame_disengages_conditioning():
    """A still (F=1) trains T2V-style even on an i2v run — no clean-everything
    degenerate step (mirrors the LTX-2 guard)."""
    drv = _engaged_i2v_driver()
    prepared = drv.prepare_latents(torch.randn(1, 4, 8, 8))  # still → F=1
    assert not drv._i2v_conditioning_engaged()
    noise = torch.randn_like(prepared)
    noisy = drv.add_noise(prepared, noise, torch.tensor([500.0]))
    assert not torch.equal(noisy, prepared)  # frame 0 IS noised


def test_i2v_forward_concats_first_frame_cond_and_mask():
    drv = _engaged_i2v_driver()
    captured = {}

    def fake_transformer(**kwargs):
        captured.update(kwargs)
        return (kwargs["hidden_states"][..., :4].clone(),)

    drv.transformer = fake_transformer
    latents_cf = torch.randn(2, 4, 3, 8, 8)
    batch: dict = {}
    drv.attach_conditioning(batch, latents_cf)
    prepared = drv.prepare_latents(latents_cf)
    noisy = drv.add_noise(prepared, torch.randn_like(prepared), torch.tensor([600.0, 600.0]))
    drv.forward_pass(noisy, torch.tensor([600.0, 600.0]), _fake_text(2), batch)

    hs = captured["hidden_states"]
    assert hs.shape == (2, 3, 8, 8, 9)
    # cond channels: frame 0 = the first-frame latent (channels-last), rest 0.
    expected_ff = to_channels_last(batch[Kandinsky5Driver.BATCH_FIRST_FRAME_LATENT])
    assert torch.equal(hs[:, :1, :, :, 4:8], expected_ff)
    assert hs[:, 1:, :, :, 4:8].abs().sum() == 0
    # mask channel: 1 in frame 0, 0 elsewhere.
    assert torch.all(hs[:, :1, :, :, 8] == 1.0)
    assert hs[:, 1:, :, :, 8].abs().sum() == 0
    # frame 0 of the latent stream is the CLEAN latent (never noised).
    assert torch.equal(hs[:, :1, :, :, :4], prepared[:, :1])


def test_i2v_forward_without_stash_raises():
    drv = _engaged_i2v_driver()
    drv.transformer = lambda **kw: (kw["hidden_states"][..., :4],)
    prepared = drv.prepare_latents(torch.randn(1, 4, 3, 8, 8))
    with pytest.raises(ValueError, match="first-frame latent"):
        drv.forward_pass(prepared, torch.tensor([500.0]), _fake_text(1), {})


# ── Real tiny-transformer forward (shape + finiteness) ─────────────────────


def test_real_tiny_transformer_forward_t2v():
    model = _tiny_transformer(visual_cond=False)
    drv = _driver()
    drv.visual_cond = False
    drv.transformer = model
    prepared = drv.prepare_latents(torch.randn(1, 4, 2, 8, 8))
    noise = torch.randn_like(prepared)
    t = torch.tensor([500.0])
    noisy = drv.add_noise(prepared, noise, t)
    out = drv.forward_pass(noisy, t, _fake_text(1), {})
    assert out.shape == (1, 2, 8, 8, 4)  # channels-last velocity
    assert torch.isfinite(out).all()
    # target lives in the same channels-last space
    target = drv.compute_target(prepared, noise, t)
    assert target.shape == out.shape


def test_real_tiny_transformer_forward_visual_cond_i2v():
    model = _tiny_transformer(visual_cond=True)
    drv = _driver(mode="i2v")
    drv.transformer = model
    drv.visual_cond = True
    latents_cf = torch.randn(1, 4, 3, 8, 8)
    drv._i2v_active = True
    batch: dict = {}
    drv.attach_conditioning(batch, latents_cf)
    prepared = drv.prepare_latents(latents_cf)
    t = torch.tensor([250.0])
    noisy = drv.add_noise(prepared, torch.randn_like(prepared), t)
    out = drv.forward_pass(noisy, t, _fake_text(1), batch)
    assert out.shape == (1, 3, 8, 8, 4)
    assert torch.isfinite(out).all()


# ── LoRA targets on the real tiny model ────────────────────────────────────


def test_lora_targets_are_fully_indexed_visual_paths():
    drv = _driver()
    targets = drv.get_lora_targets()
    assert len(targets) == 1 * len(K5_LORA_TARGET_SUFFIXES)
    assert all(t.startswith("visual_transformer_blocks.") for t in targets)
    assert "visual_transformer_blocks.0.self_attention.to_query" in targets
    assert "visual_transformer_blocks.0.feed_forward.out_layer" in targets


def test_lora_targets_expand_to_lite_block_count():
    drv = _driver(**{"transformer.num_visual_blocks": 32})
    targets = drv.get_lora_targets()
    assert len(targets) == 32 * 10  # 320 modules → 640 lora_A/B tensors


def test_peft_wraps_visual_blocks_only_on_real_model():
    """The exact-name expansion must keep text_transformer_blocks (which share
    the self_attention/feed_forward sub-module names) 100% LoRA-free."""
    from peft import LoraConfig, get_peft_model

    model = _tiny_transformer()
    drv = _driver()
    peft_model = get_peft_model(
        model, LoraConfig(r=4, lora_alpha=4, target_modules=drv.get_lora_targets())
    )

    lora_modules = [
        name
        for name, module in peft_model.named_modules()
        if hasattr(module, "lora_A") and getattr(module, "lora_A", None)
    ]
    assert lora_modules, "PEFT wrapped no modules — targets matched nothing"
    assert all("visual_transformer_blocks" in n for n in lora_modules), (
        f"LoRA bled outside visual blocks: "
        f"{[n for n in lora_modules if 'visual_transformer_blocks' not in n][:5]}"
    )
    assert not any("text_transformer_blocks" in n for n in lora_modules)
    assert not any("time_embeddings" in n for n in lora_modules)
    # 10 targets on the 1-block tiny model.
    assert len(lora_modules) == 10


def test_definition_suffixes_override_but_stay_visual_scoped():
    d = _definition()
    d.lora_targetable_modules = ["self_attention.to_query"]
    drv = Kandinsky5Driver(d, torch.device("cpu"))
    assert drv.get_lora_targets() == [
        "visual_transformer_blocks.0.self_attention.to_query"
    ]


# ── RoPE helpers + scale factor ────────────────────────────────────────────


def test_scale_factor_replicates_pipeline():
    assert get_scale_factor(512, 768) == (1, 2, 2)
    assert get_scale_factor(480, 854) == (1, 2, 2)
    assert get_scale_factor(479, 768) == (1, 3.16, 3.16)
    assert get_scale_factor(512, 855) == (1, 3.16, 3.16)
    assert get_scale_factor(1024, 1024) == (1, 3.16, 3.16)


def test_visual_rope_pos_halves_spatial_dims():
    pos = Kandinsky5Driver.build_visual_rope_pos(3, 8, 6, "cpu")
    assert pos[0].tolist() == [0, 1, 2]
    assert pos[1].tolist() == [0, 1, 2, 3]  # 8 // 2
    assert pos[2].tolist() == [0, 1, 2]  # 6 // 2


def test_forward_passes_pipeline_scale_factor():
    """Latent 64x96 → pixel 512x768 → (1, 2, 2)."""
    drv = _driver()
    drv.visual_cond = False
    captured = {}

    def fake_transformer(**kwargs):
        captured.update(kwargs)
        return (kwargs["hidden_states"].clone(),)

    drv.transformer = fake_transformer
    prepared = drv.prepare_latents(torch.randn(1, 4, 2, 64, 96))
    drv.forward_pass(prepared, torch.tensor([10.0]), _fake_text(1), {})
    assert captured["scale_factor"] == (1, 2, 2)
    assert captured["sparse_params"] is None
    assert captured["visual_rope_pos"][1].numel() == 32  # 64 // 2
    assert captured["text_rope_pos"].numel() == 6


# ── Text encoding (fake dual TE — pins template/crop/cu math) ──────────────


class _FakeProcessor:
    """Qwen2VLProcessor stand-in: 129 template tokens + 1 token per word."""

    def __call__(self, text, images=None, videos=None, max_length=None,
                 truncation=False, return_tensors="pt", padding=True):
        lengths = [KANDINSKY5_CROP_START + max(len(t.split("user\n")[-1].split()), 1)
                   for t in text]
        max_len = min(max(lengths), max_length or 10**9)
        ids = torch.zeros(len(text), max_len, dtype=torch.long)
        mask = torch.zeros(len(text), max_len, dtype=torch.long)
        for i, ln in enumerate(lengths):
            mask[i, : min(ln, max_len)] = 1
        return _FakeBatch({"input_ids": ids, "attention_mask": mask})


class _FakeBatch(dict):
    def to(self, *_args, **_kw):
        return self


class _FakeQwen:
    def __init__(self, hidden: int = 16):
        self.hidden = hidden

    def __call__(self, input_ids=None, return_dict=True, output_hidden_states=True):
        b, length = input_ids.shape
        return {"hidden_states": [torch.randn(b, length, self.hidden)]}


class _FakeClipTokenizer:
    def __call__(self, prompts, max_length=77, truncation=True,
                 add_special_tokens=True, padding="max_length", return_tensors="pt"):
        return _FakeBatch({
            "input_ids": torch.zeros(len(prompts), max_length, dtype=torch.long),
            "attention_mask": torch.ones(len(prompts), max_length, dtype=torch.long),
        })


class _FakeClip:
    def __call__(self, input_ids=None, attention_mask=None):
        return {"pooler_output": torch.randn(input_ids.shape[0], 8)}


def test_encode_text_returns_qwen_clip_cu_triple():
    drv = _driver()
    drv.tokenizer = _FakeProcessor()
    drv.text_encoder = _FakeQwen()
    drv.tokenizer_2 = _FakeClipTokenizer()
    drv.text_encoder_2 = _FakeClip()

    out = drv.encode_text(["a cat runs", "a dog"], torch.float32)
    assert isinstance(out, TextEncoderOutput)
    # crop_start slicing: only post-template tokens remain (3 words max).
    assert out.embeddings.shape == (2, 3, 16)
    assert out.pooled.shape == (2, 8)
    assert out.attention_mask.dtype == torch.int32
    assert out.attention_mask.tolist() == [0, 3, 5]  # cu_seqlens [0, 3, 3+2]


def test_prompt_template_has_crop_start_prefix_marker():
    """The template embeds the user prompt after the system block; crop_start
    129 is the pipeline's fixed template-prefix token count."""
    assert "{}" in KANDINSKY5_PROMPT_TEMPLATE
    assert KANDINSKY5_PROMPT_TEMPLATE.count("<|im_start|>") == 2
    assert KANDINSKY5_CROP_START == 129


def test_prompt_clean_collapses_whitespace_and_html():
    assert prompt_clean("a  cat\n runs&amp;jumps ") == "a cat runs&jumps"


# ── Misc driver surface ────────────────────────────────────────────────────


def test_get_text_encoders_lists_both():
    drv = _driver()
    drv.text_encoder = object()
    drv.text_encoder_2 = object()
    assert set(drv.get_text_encoders()) == {"text_encoder", "text_encoder_2"}


def test_resolve_loading_dtype_bf16():
    assert _driver().resolve_loading_dtype() is torch.bfloat16


def test_assign_components_prefers_transformer_config():
    drv = _driver()
    model = _tiny_transformer(visual_cond=True)
    drv.assign_components({"unet": model})
    assert drv.visual_cond is True
    assert drv.in_visual_dim == 4  # tiny config's in_visual_dim
