"""Tests for the Lens DiT joint-attention mask fast path.

The attention mask is the source of a training-time VRAM blow-up: passing any
explicit ``attn_mask`` to ``scaled_dot_product_attention`` disables
FlashAttention and forces the math backend, which materialises the O(S^2) score
matrix per layer. ``_build_joint_attention_mask`` therefore returns ``None``
when nothing is padded so FlashAttention can run. These tests pin that
behaviour and prove the no-mask path is numerically identical to the explicit
all-valid mask it replaces.
"""
import torch

from app.engine.models.families.microsoft_lens.vendor.transformer import (
    LensJointAttention,
    LensTransformer2DModel,
)


def _tiny_dit():
    return LensTransformer2DModel(
        patch_size=2, in_channels=128, out_channels=32, num_layers=1,
        attention_head_dim=8, num_attention_heads=2, inner_dim=16,
        enc_hidden_dim=2880, axes_dims_rope=(2, 2, 4),
        gate_mlp=True, rms_norm=True, multi_layer_encoder_feature=True,
        selected_layer_index=(5, 11, 17, 23),
    )


def test_build_mask_returns_none_when_all_valid():
    """No padding (all text positions valid) -> None, so SDPA uses Flash."""
    text_mask = torch.ones(4, 7, dtype=torch.bool)
    assert LensTransformer2DModel._build_joint_attention_mask(text_mask, 16) is None


def test_build_mask_returns_none_for_batch_size_one():
    """Batch 1 never has padding -> always the Flash path."""
    text_mask = torch.ones(1, 13, dtype=torch.bool)
    assert LensTransformer2DModel._build_joint_attention_mask(text_mask, 32) is None


def test_build_mask_masks_only_padded_text_positions():
    """A ragged batch keeps the additive mask with -inf at padded text slots."""
    img_len = 3
    text_mask = torch.tensor(
        [[True, True, True], [True, False, False]], dtype=torch.bool
    )
    mask = LensTransformer2DModel._build_joint_attention_mask(text_mask, img_len)
    assert mask is not None
    assert mask.shape == (2, 1, 1, img_len + 3)  # [B, 1, 1, img + S_txt]
    flat = mask[:, 0, 0, :]
    # Image positions (first img_len) are always valid for both rows.
    assert torch.isfinite(flat[:, :img_len]).all()
    # Row 0: all text valid. Row 1: last two text positions are -inf.
    assert torch.isfinite(flat[0, img_len:]).all()
    assert torch.isfinite(flat[1, img_len]).item()
    assert flat[1, img_len + 1].item() == float("-inf")
    assert flat[1, img_len + 2].item() == float("-inf")


def test_build_mask_casts_non_bool_input():
    """A float/int mask (1.0 = valid) is accepted and treated as boolean."""
    text_mask = torch.tensor([[1.0, 1.0, 0.0]])
    mask = LensTransformer2DModel._build_joint_attention_mask(text_mask, 2)
    assert mask is not None
    assert mask[0, 0, 0, -1].item() == float("-inf")


def test_attention_none_mask_equals_all_valid_additive_mask():
    """Dropping an all-valid mask is numerically safe.

    Adding an all-zero additive mask to the attention scores is a no-op, so the
    output with ``attn_mask=None`` must equal the output with the explicit
    all-zero mask the fast path replaces.
    """
    torch.manual_seed(0)
    bsz, seq_img, seq_txt, dim, heads, head_dim = 2, 5, 4, 16, 2, 8
    attn = LensJointAttention(
        query_dim=dim, added_kv_proj_dim=dim, dim_head=head_dim,
        heads=heads, out_dim=dim,
    ).eval()

    hidden = torch.randn(bsz, seq_img, dim)
    enc = torch.randn(bsz, seq_txt, dim)
    # RoPE freqs sized for the joint sequence (complex, head_dim/2 per token).
    img_freqs = torch.polar(
        torch.ones(seq_img, head_dim // 2), torch.randn(seq_img, head_dim // 2)
    )
    txt_freqs = torch.polar(
        torch.ones(seq_txt, head_dim // 2), torch.randn(seq_txt, head_dim // 2)
    )
    rope = (img_freqs, txt_freqs)

    all_valid_additive = torch.zeros(bsz, 1, 1, seq_img + seq_txt)

    with torch.no_grad():
        img_none, txt_none = attn(hidden, enc, rope, attention_mask=None)
        img_mask, txt_mask = attn(hidden, enc, rope, attention_mask=all_valid_additive)

    assert torch.allclose(img_none, img_mask, atol=1e-5)
    assert torch.allclose(txt_none, txt_mask, atol=1e-5)


def test_forward_all_valid_runs_without_explicit_mask():
    """End-to-end: a fully-valid batch forwards through the DiT (Flash path)."""

    dit = _tiny_dit().eval()
    noisy = torch.randn(1, 4, 128)  # latent_h=latent_w=2 -> S_img=4
    text = [torch.randn(1, 5, 2880) for _ in range(4)]
    mask = torch.ones(1, 5, dtype=torch.bool)
    with torch.no_grad():
        out = dit(
            hidden_states=noisy,
            encoder_hidden_states=text,
            encoder_hidden_states_mask=mask,
            timestep=torch.tensor([0.5]),
            img_shapes=[(1, 2, 2)],
        )
    assert out.shape == (1, 4, 128)


def test_forward_ragged_batch_still_masks_and_runs():
    """A padded batch keeps the mask path and produces correct output shape."""

    dit = _tiny_dit().eval()
    noisy = torch.randn(2, 4, 128)
    text = [torch.randn(2, 5, 2880) for _ in range(4)]
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[1, 3:] = False  # row 1 has two padded text positions
    with torch.no_grad():
        out = dit(
            hidden_states=noisy,
            encoder_hidden_states=text,
            encoder_hidden_states_mask=mask,
            timestep=torch.tensor([0.5, 0.5]),
            img_shapes=[(1, 2, 2)],
        )
    assert out.shape == (2, 4, 128)
