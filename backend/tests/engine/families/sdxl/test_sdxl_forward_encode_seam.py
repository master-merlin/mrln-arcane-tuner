"""SDXL trainer→driver seam (P2a dedup).

Refactor-TDD pins for eliminating the duplicated trainer-level
``forward_pass`` / ``_encode_text_direct`` bodies.  SDXL's forward needs the
CLIP-2 pooled embedding (``_pooled_embeds``) that ``encode_text`` stashes — so
the pin verifies the pooled produced by encode reaches the UNet through the
delegated driver forward, and that the sampler-facing ``pipeline._pooled_embeds``
is still populated.
"""

from __future__ import annotations

import structlog
import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.sdxl.driver import SDXLDriver
from app.engine.models.families.sdxl.trainer import SDXLTrainer


class _Toks:
    def __call__(self, captions, *, padding, max_length, truncation, return_tensors):
        b = len(captions)
        out = type("_T", (), {})()
        out.input_ids = torch.zeros(b, max_length, dtype=torch.long)
        return out


class _TE(torch.nn.Module):
    def __init__(self, dim: int, pooled: bool) -> None:
        super().__init__()
        self.dim = dim
        self.pooled = pooled
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, output_hidden_states):
        b, seq = input_ids.shape
        out = type("_O", (), {})()
        out.hidden_states = (torch.randn(b, seq, self.dim), torch.randn(b, seq, self.dim))
        if self.pooled:
            out.text_embeds = torch.randn(b, self.dim)
        return out


class _CaptureUNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_pooled: torch.Tensor | None = None
        self.seen_time_ids: torch.Tensor | None = None

    def forward(self, noisy_input, timesteps, *, encoder_hidden_states, added_cond_kwargs):
        self.seen_pooled = added_cond_kwargs["text_embeds"].detach().clone()
        self.seen_time_ids = added_cond_kwargs["time_ids"]
        out = type("_S", (), {})()
        out.sample = torch.zeros_like(noisy_input)
        return out


def _defn() -> ModelDefinition:
    return ModelDefinition(
        id="sdxl-base", family="sdxl", name="SDXL", defaults={}, components={},
    )


def _trainer(d1: int = 8, d2: int = 8) -> SDXLTrainer:
    t = object.__new__(SDXLTrainer)
    t.logger = structlog.get_logger("test")
    t.device = torch.device("cpu")
    t.config = {"cache_text_embeddings": False}
    t.text_cache = {}
    t.te_max_length = 77
    t._pooled_cache = {}

    drv = SDXLDriver(_defn(), torch.device("cpu"))
    drv.te_max_length = 77
    tok1, tok2 = _Toks(), _Toks()
    te1, te2 = _TE(d1, pooled=False), _TE(d2, pooled=True)
    comps = {"tokenizer_1": tok1, "tokenizer_2": tok2}
    t.components = dict(comps)
    drv._components = dict(comps)
    for obj in (t, drv):
        obj.text_encoder_1 = te1
        obj.text_encoder_2 = te2
    t.driver = drv
    return t


def test_encode_returns_prompt_embeds_and_stashes_pooled():
    t = _trainer()
    prompt = t.encode_text(["a fox"], torch.float32)
    assert prompt.ndim == 3
    # sampler reads pipeline._pooled_embeds — must be populated post-encode.
    assert getattr(t, "_pooled_embeds", None) is not None
    assert t._pooled_embeds.shape[-1] == 8


def test_encode_to_forward_pooled_reaches_unet():
    """The pooled produced by encode must flow into the UNet's added_cond_kwargs."""
    t = _trainer()
    unet = _CaptureUNet()
    t.unet = unet
    t.driver.unet = unet

    prompt = t.encode_text(["a fox"], torch.float32)
    pooled_from_encode = t._pooled_embeds.clone()

    B, C, H, W = 1, 4, 8, 8
    noisy = torch.randn(B, C, H, W)
    time_ids = torch.zeros(B, 6)
    out = t.forward_pass(noisy, torch.tensor([10]), prompt, {"time_ids": time_ids})

    assert out.shape == (B, C, H, W)
    assert torch.allclose(unet.seen_pooled, pooled_from_encode)
    assert unet.seen_time_ids is time_ids
