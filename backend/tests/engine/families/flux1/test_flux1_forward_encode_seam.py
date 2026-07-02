"""FLUX.1 trainer→driver seam (P2a dedup).

Refactor-TDD pins for eliminating the duplicated trainer-level
``forward_pass`` / ``_encode_text_direct`` bodies.  The driver's copy had
PROVEN drift: it hardcoded ``guidance_scale = 3.5`` and used
``noisy_input.dtype``, while the LIVE trainer path reads
``config["guidance_scale"]`` and uses ``autocast_dtype``.  Trainer semantics
win → the delegated driver forward must be config-driven and autocast-typed.

These exercise the REAL trainer→driver seam and pass BOTH before the refactor
(trainer owns the body) and after (driver owns it).
"""

from __future__ import annotations

import structlog
import torch

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.flux1.driver import Flux1Driver
from app.engine.models.families.flux1.trainer import Flux1Trainer


class _Toks:
    def __call__(self, captions, *, padding, max_length, truncation, return_tensors):
        b = len(captions)
        out = type("_T", (), {})()
        out.input_ids = torch.zeros(b, 8, dtype=torch.long)
        return out


class _T5(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        b, seq = input_ids.shape
        out = type("_O", (), {})()
        out.last_hidden_state = torch.randn(b, seq, self.dim)
        return out


class _Clip(torch.nn.Module):
    def __init__(self, pooled_dim: int) -> None:
        super().__init__()
        self.pooled_dim = pooled_dim
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, output_hidden_states):
        b = input_ids.shape[0]
        out = type("_O", (), {})()
        out.pooler_output = torch.randn(b, self.pooled_dim)
        return out


class _CaptureTransformer(torch.nn.Module):
    def __init__(self, pooled_dim: int) -> None:
        super().__init__()
        self.config = type("_C", (), {})()
        self.config.pooled_projection_dim = pooled_dim
        self.seen_guidance: torch.Tensor | None = None
        self.seen_timestep: torch.Tensor | None = None

    def forward(self, *, hidden_states, encoder_hidden_states, pooled_projections,
                timestep, img_ids, txt_ids, guidance, return_dict):
        self.seen_guidance = None if guidance is None else guidance.detach().clone()
        self.seen_timestep = timestep.detach().clone()
        return (torch.zeros_like(hidden_states),)


def _defn() -> ModelDefinition:
    return ModelDefinition(
        id="flux1-dev", family="flux1", name="FLUX.1 Dev", defaults={}, components={},
    )


def _trainer(guidance_scale: float, dtype: torch.dtype) -> Flux1Trainer:
    t = object.__new__(Flux1Trainer)
    t.logger = structlog.get_logger("test")
    t.device = torch.device("cpu")
    t.config = {"cache_text_embeddings": False, "guidance_scale": guidance_scale}
    t.text_cache = {}
    t.autocast_dtype = dtype
    t.use_guidance_embed = True
    t.te_t5_max_length = 512
    t.te_clip_max_length = 77
    t._clip_pooled_cache = {}

    drv = Flux1Driver(_defn(), torch.device("cpu"))
    drv.use_guidance_embed = True
    drv.te_t5_max_length = 512
    drv.te_clip_max_length = 77
    t5_tok, clip_tok, t5, clip = _Toks(), _Toks(), _T5(32), _Clip(8)
    for obj in (t, drv):
        obj.t5_tokenizer = t5_tok
        obj.clip_tokenizer = clip_tok
        obj.t5_encoder = t5
        obj.clip_encoder = clip
    t.driver = drv
    return t


def test_forward_reads_guidance_from_config_and_autocast_dtype():
    """The drift fix: guidance == config value @ autocast_dtype (not 3.5 @ input dtype)."""
    t = _trainer(guidance_scale=7.0, dtype=torch.bfloat16)
    transformer = _CaptureTransformer(pooled_dim=8)
    t.transformer = transformer
    t.driver.transformer = transformer

    t5_emb = t.encode_text(["a fox"], torch.bfloat16)

    latents = torch.randn(1, 16, 8, 8)
    prepared = t.prepare_latents_for_training(latents)
    with torch.no_grad():
        t.forward_pass(prepared, torch.tensor([500.0]), t5_emb, {})

    assert transformer.seen_guidance is not None
    assert float(transformer.seen_guidance[0]) == 7.0, "guidance must come from config"
    assert transformer.seen_guidance.dtype == torch.bfloat16, "guidance dtype = autocast_dtype"
    assert torch.allclose(transformer.seen_timestep, torch.tensor([0.5]))


def test_encode_delegates_and_stashes_clip_pooled():
    """encode_text returns the T5 sequence and stashes CLIP pooled for forward."""
    t = _trainer(guidance_scale=3.5, dtype=torch.float32)
    t5_emb = t.encode_text(["a fox"], torch.float32)
    assert isinstance(t5_emb, torch.Tensor) and t5_emb.ndim == 3
    assert getattr(t, "_clip_pooled", None) is not None
    assert t._clip_pooled.shape[-1] == 8  # CLIP pooled dim
