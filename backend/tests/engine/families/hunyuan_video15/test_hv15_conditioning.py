"""hv15 65-channel input builders + glyph extraction + upstream parity.

Weight-free unit tests for the concat contract
``[latents(32), cond(32), mask(1)]`` (T2V zeros / I2V first-frame+mask), the
zero ``image_embeds`` helper, and the ByT5 glyph text extraction — including
byte-parity with the installed diffusers pipeline's ``extract_glyph_texts``
and the chat-template system message.
"""

import pytest
import torch

from app.engine.models.families.hunyuan_video15.driver import (
    HV15_SYSTEM_MESSAGE,
    IMAGE_EMBED_DIM,
    IN_CHANNELS,
    NOISE_CHANNELS,
    VISION_NUM_SEMANTIC_TOKENS,
    build_i2v_cond_and_mask,
    build_model_input,
    build_t2v_cond_and_mask,
    extract_glyph_text,
    zero_image_embeds,
)


def _noisy(b=2, c=NOISE_CHANNELS, f=3, h=4, w=5) -> torch.Tensor:
    return torch.randn(b, c, f, h, w)


# ── T2V builders ───────────────────────────────────────────────────────────


def test_t2v_cond_and_mask_are_zeros():
    noisy = _noisy()
    cond, mask = build_t2v_cond_and_mask(noisy)
    assert cond.shape == (2, 32, 3, 4, 5)
    assert mask.shape == (2, 1, 3, 4, 5)
    assert torch.all(cond == 0) and torch.all(mask == 0)
    assert cond.dtype == noisy.dtype


def test_t2v_rejects_4d():
    with pytest.raises(ValueError, match="5D"):
        build_t2v_cond_and_mask(torch.randn(2, 32, 4, 5))


def test_model_input_is_65_channels_with_noisy_first():
    noisy = _noisy()
    cond, mask = build_t2v_cond_and_mask(noisy)
    x = build_model_input(noisy, cond, mask)
    assert x.shape == (2, IN_CHANNELS, 3, 4, 5)
    assert IN_CHANNELS == 65
    # The first 32 channels ARE the noised latent (the diffusion variable) —
    # the trainer computes the velocity target over exactly those channels.
    assert torch.equal(x[:, :32], noisy)


def test_model_input_rejects_wrong_channel_sum():
    noisy = _noisy()
    cond, mask = build_t2v_cond_and_mask(noisy)
    with pytest.raises(ValueError, match="65"):
        build_model_input(noisy, cond[:, :16], mask)


# ── I2V builders ───────────────────────────────────────────────────────────


def test_i2v_cond_carries_first_frame_only():
    noisy = _noisy()
    first = torch.randn(2, 32, 1, 4, 5)
    cond, mask = build_i2v_cond_and_mask(noisy, first)
    assert torch.equal(cond[:, :, 0], first[:, :, 0])
    assert torch.all(cond[:, :, 1:] == 0)  # frames 1: zeroed (upstream :626)
    assert mask.shape == (2, 1, 3, 4, 5)
    assert torch.all(mask[:, :, 0] == 1.0)
    assert torch.all(mask[:, :, 1:] == 0.0)


def test_i2v_accepts_4d_first_frame():
    noisy = _noisy()
    first = torch.randn(2, 32, 4, 5)  # no frame axis
    cond, _ = build_i2v_cond_and_mask(noisy, first)
    assert torch.equal(cond[:, :, 0], first)


def test_i2v_multiframe_stash_keeps_only_frame0():
    noisy = _noisy()
    multi = torch.randn(2, 32, 3, 4, 5)
    cond, _ = build_i2v_cond_and_mask(noisy, multi)
    assert torch.equal(cond[:, :, 0], multi[:, :, 0])


def test_i2v_rejects_wrong_noise_channels():
    with pytest.raises(ValueError, match="32"):
        build_i2v_cond_and_mask(torch.randn(2, 16, 3, 4, 5), torch.randn(2, 16, 1, 4, 5))


# ── image_embeds ───────────────────────────────────────────────────────────


def test_zero_image_embeds_shape_and_zeroness():
    z = zero_image_embeds(3, dtype=torch.bfloat16)
    assert z.shape == (3, VISION_NUM_SEMANTIC_TOKENS, IMAGE_EMBED_DIM) == (3, 729, 1152)
    assert torch.all(z == 0)
    assert z.dtype == torch.bfloat16


# ── Glyph extraction (ByT5 channel) ────────────────────────────────────────


def test_glyph_none_without_quotes():
    # The common training-caption case → zero te2 downstream.
    assert extract_glyph_text("a cat walks on the grass, realistic") is None


def test_glyph_straight_quotes():
    assert extract_glyph_text('a sign reading "OPEN"') == 'Text "OPEN". '


def test_glyph_curly_quotes():
    assert extract_glyph_text("a neon sign “CLOSED” at night") == 'Text "CLOSED". '


def test_glyph_multiple_and_dedupe():
    out = extract_glyph_text('"A" then "B" then "A" again')
    assert out == 'Text "A". Text "B". '


def test_glyph_parity_with_upstream_pipeline():
    """Byte-parity with the installed diffusers extract_glyph_texts."""
    from diffusers.pipelines.hunyuan_video1_5.pipeline_hunyuan_video1_5 import (
        extract_glyph_texts,
    )

    cases = [
        "no quotes at all",
        'one "WORD" here',
        "curly “QUOTES” too",
        'mixed "A" and “B” and "A"',
        'empty "" quotes',
        "",
    ]
    for prompt in cases:
        assert extract_glyph_text(prompt) == extract_glyph_texts(prompt), prompt


def test_system_message_matches_upstream_default():
    """Byte-equality with the pipeline's chat-template system message (the
    embedding cache key space depends on it)."""
    import inspect

    from diffusers import HunyuanVideo15Pipeline

    upstream_default = inspect.signature(
        HunyuanVideo15Pipeline._get_mllm_prompt_embeds
    ).parameters["system_message"].default
    assert HV15_SYSTEM_MESSAGE == upstream_default
