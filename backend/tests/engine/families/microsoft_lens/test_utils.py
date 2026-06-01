"""microsoft_lens latent/text utility tests."""
import torch

from app.engine.models.families.microsoft_lens import utils


def test_patchify_unpatchify_round_trip():
    x = torch.randn(2, 32, 8, 6)
    seq = utils.patchify_to_seq(x)
    assert seq.shape == (2, (8 // 2) * (6 // 2), 128)
    back = utils.unpatchify_from_seq(seq, latent_h=8 // 2, latent_w=6 // 2)
    assert back.shape == x.shape
    assert torch.allclose(back, x, atol=1e-6)


def test_selected_hidden_state_indices():
    assert utils.lens_layers_to_hf_indices((5, 11, 17, 23)) == [6, 12, 18, 24]


def test_bn_normalize_inverts_denormalize():
    class _BN:
        running_mean = torch.randn(128)
        running_var = torch.rand(128) + 0.5
        eps = 1e-5

    class _VAE:
        bn = _BN()

    vae = _VAE()
    x = torch.randn(2, 10, 128)
    normed = utils.bn_normalize_seq(x, vae)
    denormed = utils.bn_denormalize_seq(normed, vae)
    assert torch.allclose(denormed, x, atol=1e-4)


def test_drop_txt_offset_slices_prefix():
    feats = [torch.randn(1, 150, 4) for _ in range(4)]
    mask = torch.ones(1, 150, dtype=torch.bool)
    out_feats, out_mask = utils.drop_txt_offset(feats, mask, offset=97)
    assert all(f.shape[1] == 150 - 97 for f in out_feats)
    assert out_mask.shape[1] == 150 - 97


def test_drop_txt_offset_short_sequence_returns_empty():
    feats = [torch.randn(1, 50, 4) for _ in range(4)]
    mask = torch.ones(1, 50, dtype=torch.bool)
    out_feats, out_mask = utils.drop_txt_offset(feats, mask, offset=97)
    assert all(f.shape[1] == 0 for f in out_feats)
    assert out_mask.shape[1] == 0


def test_drop_txt_offset_exact_boundary_returns_empty():
    # seq_len == offset: the strict > condition means no content survives.
    feats = [torch.randn(1, 97, 4) for _ in range(4)]
    mask = torch.ones(1, 97, dtype=torch.bool)
    out_feats, out_mask = utils.drop_txt_offset(feats, mask, offset=97)
    assert all(f.shape[1] == 0 for f in out_feats)
    assert out_mask.shape[1] == 0


def test_render_chat_prompt_splits_on_return_sentinel():
    captured = {}

    class _StubTokenizer:
        def apply_chat_template(self, conversation, **kwargs):
            captured["conversation"] = conversation
            return "<sys>PROMPT<|return|>trailing"

    stub = _StubTokenizer()
    result = utils.render_chat_prompt("PROMPT", stub)

    assert result == "<sys>PROMPT"

    conv = captured["conversation"]
    roles = [turn["role"] for turn in conv]
    assert roles == ["system", "user", "assistant"]

    user_turn = next(t for t in conv if t["role"] == "user")
    assert user_turn["content"] == "PROMPT"
