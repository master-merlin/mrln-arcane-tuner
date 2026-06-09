"""Smoke tests for the Ideogram 4 model family (no weights downloaded)."""
from __future__ import annotations

import pytest
import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.models.registry import ModelRegistry


def test_family_is_discovered():
    registry = ModelRegistry()
    registry.discover_families()
    family_cls = registry.get_family_class("ideogram4")
    assert family_cls.family_name == "ideogram4"


def test_family_returns_trainer_class():
    from app.engine.models.families.ideogram4.family import IdeogramV4Family
    from app.engine.models.families.ideogram4.trainer import IdeogramV4Trainer

    definition = ModelDefinition(id="x", family="ideogram4", name="X")
    family = IdeogramV4Family(definition, {})
    assert family.get_trainer_class() is IdeogramV4Trainer


def test_dequantize_fp8_state_dict_applies_scale():
    from app.engine.models.families.ideogram4.utils import dequantize_fp8_state_dict

    w = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float8_e4m3fn)
    scale = torch.tensor([2.0, 0.5], dtype=torch.float32)
    sd = {"blk.weight": w, "blk.weight_scale": scale}

    out = dequantize_fp8_state_dict(sd)

    assert "blk.weight_scale" not in out          # scale consumed
    expected = w.to(torch.float32) * scale[:, None]
    assert torch.allclose(out["blk.weight"].float(), expected)


def test_dequantize_passes_through_unscaled():
    from app.engine.models.families.ideogram4.utils import dequantize_fp8_state_dict

    sd = {"norm.weight": torch.ones(4)}
    out = dequantize_fp8_state_dict(sd)
    assert torch.allclose(out["norm.weight"], torch.ones(4))


def test_patchify_roundtrip():
    from app.engine.models.families.ideogram4.utils import (
        patchify_to_seq, unpatchify_from_seq,
    )
    x = torch.randn(2, 32, 16, 16)
    seq = patchify_to_seq(x)               # [2, (8*8), 128]
    assert seq.shape == (2, 64, 128)
    back = unpatchify_from_seq(seq, 8, 8)
    assert torch.allclose(back, x)


def test_encode_text_concats_selected_layers():
    from app.engine.core.definitions import ModelDefinition
    from app.engine.models.families.ideogram4.driver import IdeogramV4Driver
    from app.engine.models.families.ideogram4.utils import QWEN3VL_SELECTED_LAYERS

    HID = 8  # tiny stand-in for Qwen3-VL hidden size

    class _Tok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return messages[0]["content"]
        def __call__(self, texts, **kw):
            n = max(len(t.split()) for t in texts) or 1
            import torch
            return {
                "input_ids": torch.ones(len(texts), n, dtype=torch.long),
                "attention_mask": torch.ones(len(texts), n, dtype=torch.long),
            }

    class _Out:
        def __init__(self, hs):
            self.hidden_states = hs

    import torch

    class _TE(torch.nn.Module):
        def forward(self, input_ids, attention_mask, output_hidden_states, **kw):
            b, n = input_ids.shape
            # +2: HF prepends the embedding output at [0], and the driver taps
            # post-layer index k at HF hidden_states[k+1], so the top tap needs
            # index max+1 -> the tuple must be length max+2. Returning exactly
            # max+2 also asserts the driver never reads past the +1 shift.
            n_hs = max(QWEN3VL_SELECTED_LAYERS) + 2
            return _Out([torch.randn(b, n, HID) for _ in range(n_hs)])

    defn = ModelDefinition(id="x", family="ideogram4", name="X")
    drv = IdeogramV4Driver(defn, torch.device("cpu"))
    drv.text_encoder = _TE()
    drv.tokenizer = _Tok()

    out = drv.encode_text(["a cat sitting"], torch.float32)
    assert out.embeddings.shape[-1] == len(QWEN3VL_SELECTED_LAYERS) * HID
    assert out.embeddings.shape[0] == 1


def test_latent_norm_roundtrip():
    from app.engine.models.families.ideogram4.utils import (
        denormalize_latents, normalize_latents,
    )
    x = torch.randn(2, 7, 128)
    back = denormalize_latents(normalize_latents(x))
    assert torch.allclose(back, x, atol=1e-5)


def test_latent_norm_constants_match_upstream():
    """Constants must equal upstream get_latent_norm() exactly (no approximation)."""
    from app.engine.models.families.ideogram4 import utils

    assert len(utils.LATENT_SHIFT) == 128
    assert len(utils.LATENT_SCALE) == 128
    # spot-check the documented upstream endpoints
    assert utils.LATENT_SHIFT[0] == 0.01984364
    assert utils.LATENT_SHIFT[-1] == -0.01760592
    assert utils.LATENT_SCALE[0] == 1.63933691
    assert utils.LATENT_SCALE[-1] == 1.68533454


def _make_driver_with_stub_dit(feat_dim=8, latent_h=4, latent_w=4):
    from app.engine.models.families.ideogram4.driver import IdeogramV4Driver

    captured = {}

    class _DiT(torch.nn.Module):
        def forward(self, *, llm_features, x, t, position_ids, segment_ids, indicator):
            captured["t"] = t.detach().clone()
            captured["L"] = x.shape[1]
            captured["position_ids"] = position_ids.detach().clone()
            captured["segment_ids"] = segment_ids.detach().clone()
            captured["indicator"] = indicator.detach().clone()
            captured["llm_features"] = llm_features.detach().clone()
            captured["x"] = x.detach().clone()
            return torch.zeros(x.shape[0], x.shape[1], x.shape[2])

    defn = ModelDefinition(id="x", family="ideogram4", name="X")
    drv = IdeogramV4Driver(defn, torch.device("cpu"))
    drv.transformer = _DiT()
    drv._latent_h, drv._latent_w = latent_h, latent_w
    return drv, captured


def test_forward_pass_timestep_scale_divides_by_1000():
    """Trainer passes [0,1000]; DiT wants [0,1]. Guards the prior ×1000 noise bug."""
    drv, captured = _make_driver_with_stub_dit(feat_dim=8, latent_h=4, latent_w=4)

    image_seq = torch.randn(1, 16, 128)            # S_img = 4*4 = 16
    feats = torch.randn(1, 5, 8)
    text = (feats, torch.ones(1, 5, dtype=torch.bool))
    ts = torch.tensor([500.0])                     # [0,1000] convention in

    out = drv.forward_pass(image_seq, ts, text, {"latent_h": 4, "latent_w": 4})

    assert torch.allclose(captured["t"], ts / 1000.0), \
        "driver must divide [0,1000] timestep by 1000"
    # image-position outputs sliced back to [B, S_img, 128]
    assert out.shape == (1, 16, 128)
    assert captured["L"] == 5 + 16


def test_build_packed_inputs_shapes_and_role_counts():
    """_build_packed_inputs: pos last-dim 3, indicator/segment len L, role counts."""
    drv, _ = _make_driver_with_stub_dit(latent_h=2, latent_w=3)

    s_text, s_img = 4, 6  # 2*3
    text_feats = torch.randn(1, s_text, 8)
    # one padded text token at the end
    text_mask = torch.tensor([[True, True, True, False]])
    image_seq = torch.randn(1, s_img, 128)

    packed = drv._build_packed_inputs(text_feats, text_mask, image_seq, 2, 3)
    L = s_text + s_img

    assert packed["position_ids"].shape == (1, L, 3)
    assert packed["indicator"].shape == (1, L)
    assert packed["segment_ids"].shape == (1, L)
    assert packed["x"].shape == (1, L, 128)
    assert packed["llm_features"].shape == (1, L, 8)

    ind = packed["indicator"][0]
    # 3 real text tokens -> LLM_TOKEN_INDICATOR(3); 6 image -> OUTPUT_IMAGE(2);
    # 1 padded text -> 0.
    assert int((ind == 3).sum()) == 3
    assert int((ind == 2).sum()) == 6
    assert int((ind == 0).sum()) == 1

    # image positions are offset; text positions are < offset
    pos = packed["position_ids"][0]
    assert (pos[:s_text] < 65536).all()
    assert (pos[s_text:] >= 65536).all()
    # padded text position has zeroed llm_features
    assert torch.allclose(packed["llm_features"][0, 3], torch.zeros(8))
    # image latents land at image positions
    assert torch.allclose(packed["x"][0, s_text:], image_seq[0])
    assert torch.allclose(packed["x"][0, :s_text], torch.zeros(s_text, 128))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
