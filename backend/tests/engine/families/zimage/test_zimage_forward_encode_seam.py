"""Z-Image trainer→driver seam (P2a dedup).

Refactor-TDD pins for eliminating the duplicated trainer-level
``forward_pass`` / ``_encode_text_direct`` bodies.  Z-Image's forward uses the
INVERTED flow-matching timestep convention (``(1000 - t) / 1000``) and the
per-sample variable-length list API; the pins verify both survive delegation.
"""

from __future__ import annotations

import structlog
import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.zimage.driver import ZImageDriver
from app.engine.models.families.zimage.trainer import ZImageTrainer


class _StubTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                            enable_thinking):
        return messages[0]["content"]

    def __call__(self, templated, *, padding, max_length, truncation, return_tensors):
        b = len(templated)
        out = type("_T", (), {})()
        out.input_ids = torch.zeros(b, 6, dtype=torch.long)
        out.attention_mask = torch.ones(b, 6, dtype=torch.long)
        return out


class _StubTE(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, *, input_ids, attention_mask, output_hidden_states):
        b, seq = input_ids.shape
        out = type("_O", (), {})()
        out.hidden_states = (torch.randn(b, seq, self.dim), torch.randn(b, seq, self.dim))
        return out


class _CaptureModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_t: torch.Tensor | None = None

    def forward(self, *, x, t, cap_feats, return_dict):
        self.seen_t = t.detach().clone()
        return ([xi for xi in x],)  # echo per-sample [C,1,H,W]


def _defn() -> ModelDefinition:
    return ModelDefinition(
        id="zimage-base", family="zimage", name="Z-Image", defaults={}, components={},
    )


def _trainer(dim: int = 8) -> ZImageTrainer:
    t = object.__new__(ZImageTrainer)
    t.logger = structlog.get_logger("test")
    t.device = torch.device("cpu")
    t.config = {"cache_text_embeddings": False}
    t.text_cache = {}
    t.max_length = 512

    drv = ZImageDriver(_defn(), torch.device("cpu"))
    drv.max_length = 512
    tok, te = _StubTokenizer(), _StubTE(dim)
    for obj in (t, drv):
        obj.tokenizer = tok
        obj.text_encoder = te
    t.driver = drv
    return t


def test_encode_returns_variable_length_list():
    t = _trainer()
    out = t.encode_text(["a fox", "a cat"], torch.float32)
    assert isinstance(out, list) and len(out) == 2
    assert all(e.ndim == 2 for e in out)  # [Li, D] per sample


def test_forward_inverts_timestep_and_keeps_shape():
    t = _trainer()
    model = _CaptureModel()
    t.model = model
    t.driver.model = model

    emb = t.encode_text(["a fox"], torch.float32)
    B, C, H, W = 1, 4, 8, 8
    noisy = torch.randn(B, C, H, W)
    with torch.no_grad():
        pred = t.forward_pass(noisy, torch.tensor([250.0]), emb, {})

    assert pred.shape == (B, C, H, W)
    # Inverted convention: (1000 - 250) / 1000 = 0.75.
    assert torch.allclose(model.seen_t, torch.tensor([0.75]))
