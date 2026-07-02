"""Qwen-Image trainer→driver seam (P2a dedup).

Refactor-TDD pins for eliminating the duplicated trainer-level
``forward_pass`` / ``_encode_text_direct`` bodies in favour of the driver
(the base ``PipelineBaseMixin.forward_pass`` delegates to
``driver.forward_pass``; the trainer's text path delegates to
``driver.encode_text`` the way krea2 does).

These exercise the REAL trainer→driver seam (no mocking of the seam itself):
the real ``QwenImageTrainer`` methods run against a real ``QwenImageDriver``
with stub tokenizer/TE/transformer.  They pass BEFORE the refactor (the
trainer owns the body) and MUST still pass AFTER (the driver owns it).

Load-bearing reconcile pinned here: the trainer's production ``max_length``
is 1024 (``QwenImageTrainer.TOKENIZER_MAX_LENGTH``); the driver's copy drifted
to 512.  Trainer semantics win → the delegated encode must tokenize with
``max_length = 1024 + 34``.
"""

from __future__ import annotations

import structlog
import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.qwen_image.driver import QwenImageDriver
from app.engine.models.families.qwen_image.trainer import (
    PROMPT_TEMPLATE_DROP_IDX,
    TOKENIZER_MAX_LENGTH,
    QwenImageTrainer,
)


class _StubTokenizer:
    def __init__(self) -> None:
        self.seen_max_length: int | None = None

    def __call__(self, txt, *, max_length, padding, truncation, return_tensors):
        self.seen_max_length = max_length
        b = len(txt)
        seq = 40  # > drop idx (34) so ≥1 real token survives the preamble drop
        out = type("_Tok", (), {})()
        out.input_ids = torch.zeros(b, seq, dtype=torch.long)
        out.attention_mask = torch.ones(b, seq, dtype=torch.long)
        return out


class _StubTE(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, *, input_ids, attention_mask, output_hidden_states):
        b, seq = input_ids.shape
        out = type("_O", (), {})()
        out.hidden_states = (torch.randn(b, seq, self.dim),)
        return out


class _CaptureTransformer(torch.nn.Module):
    """Captures the timestep it is fed; returns a correctly-shaped patch seq."""

    def __init__(self, out_channels: int, patch_size: int = 2) -> None:
        super().__init__()
        self.config = type("_C", (), {})()
        self.config.patch_size = patch_size
        self.config.out_channels = out_channels
        self.seen_timestep: torch.Tensor | None = None

    def forward(self, *, hidden_states, encoder_hidden_states,
                encoder_hidden_states_mask, timestep, img_shapes,
                txt_seq_lens, return_dict):
        self.seen_timestep = timestep.detach().clone()
        b, seq, _ = hidden_states.shape
        pdim = self.config.out_channels * self.config.patch_size ** 2
        return (torch.zeros(b, seq, pdim),)


def _defn() -> ModelDefinition:
    return ModelDefinition(
        id="qwen-image", family="qwen_image", name="Qwen-Image",
        defaults={}, components={},
    )


def _trainer(te_dim: int = 16) -> QwenImageTrainer:
    t = object.__new__(QwenImageTrainer)
    t.logger = structlog.get_logger("test")
    t.device = torch.device("cpu")
    t.config = {"cache_text_embeddings": False}
    t.text_cache = {}
    t.max_length = TOKENIZER_MAX_LENGTH  # 1024 — set by _assign_components live
    drv = QwenImageDriver(_defn(), torch.device("cpu"))
    drv.max_length = t.max_length  # the sync _assign_components performs
    tok, te = _StubTokenizer(), _StubTE(te_dim)
    drv.tokenizer = tok
    drv.text_encoder = te
    # Base _assign_components aliases these onto the trainer too; set both so
    # the pin holds whether the pre- or post-refactor path runs.
    t.tokenizer = tok
    t.text_encoder = te
    t.driver = drv
    return t


def test_encode_delegates_to_driver_with_trainer_max_length():
    """Trainer max_length (1024) must win: tokenizer sees 1024 + 34, not 512."""
    t = _trainer()
    emb, mask = t.encode_text(["a fox in snow"], torch.float32)

    assert t.driver.tokenizer.seen_max_length == TOKENIZER_MAX_LENGTH + PROMPT_TEMPLATE_DROP_IDX
    assert TOKENIZER_MAX_LENGTH == 1024
    assert isinstance(emb, torch.Tensor) and emb.ndim == 3
    assert mask.ndim == 2 and mask.shape[0] == emb.shape[0]


def test_encode_to_forward_real_seam_shape_and_timestep():
    """encode_text → forward_pass round-trip: driver forward is exercised."""
    t = _trainer()
    B, C, H, W = 1, 16, 4, 4
    t.model = _CaptureTransformer(out_channels=C).eval()
    t.driver.model = t.model

    emb, mask = t.encode_text(["a fox"], torch.float32)
    noisy = torch.randn(B, C, H, W)
    with torch.no_grad():
        pred = t.forward_pass(noisy, torch.tensor([500.0]), (emb, mask), {})

    assert pred.shape == (B, C, H, W)
    # Qwen-Image transformer wants [0,1]; forward must divide [0,1000] by 1000.
    assert torch.allclose(t.model.seen_timestep, torch.tensor([0.5]))
