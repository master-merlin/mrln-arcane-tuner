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
            n_hs = max(QWEN3VL_SELECTED_LAYERS) + 1
            return _Out([torch.randn(b, n, HID) for _ in range(n_hs)])

    defn = ModelDefinition(id="x", family="ideogram4", name="X")
    drv = IdeogramV4Driver(defn, torch.device("cpu"))
    drv.text_encoder = _TE()
    drv.tokenizer = _Tok()

    out = drv.encode_text(["a cat sitting"], torch.float32)
    assert out.embeddings.shape[-1] == len(QWEN3VL_SELECTED_LAYERS) * HID
    assert out.embeddings.shape[0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
