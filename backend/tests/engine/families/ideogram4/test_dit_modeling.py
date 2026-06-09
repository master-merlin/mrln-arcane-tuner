"""Vendored Ideogram4 DiT: instantiation + module-name + forward contract.

Uses a tiny synthetic config (no gated weights) to lock the submodule names
the LoRA target list depends on, and the forward/timestep contract the driver
must honour.
"""

from __future__ import annotations

from accelerate import init_empty_weights

from app.engine.models.families.ideogram4.vendor.modeling_ideogram4 import (
    Ideogram4Transformer2DModel,
)

# Tiny synthetic config. Every key matches a kwarg of the ported
# `Ideogram4Transformer2DModel.__init__`. Values are kept small so the model
# instantiates in seconds on CPU under `init_empty_weights`.
#
# Constraints honoured by these values:
#   - emb_dim % num_heads == 0  (head_dim = emb_dim // num_heads = 8, even).
#   - MRoPE uses indices up to max(mrope_section) * 3 into an inv_freq vector of
#     length head_dim // 2 == 4, so each mrope_section entry must satisfy
#     section * 3 <= 4  ->  section == 1. Hence mrope_section=(1, 1, 1).
TINY_CONFIG = {
    "emb_dim": 16,
    "num_layers": 2,
    "num_heads": 2,
    "intermediate_size": 32,
    "adanln_dim": 8,
    "in_channels": 8,
    "llm_features_dim": 12,
    "rope_theta": 10000,
    "mrope_section": (1, 1, 1),
    "norm_eps": 1e-5,
}


def _tiny_model():
    with init_empty_weights():
        return Ideogram4Transformer2DModel.from_config(TINY_CONFIG)


def test_model_instantiates_from_config():
    model = _tiny_model()
    assert model is not None


def test_block_count_matches_config():
    model = _tiny_model()
    blocks = getattr(model, "layers", None) or getattr(model, "transformer_blocks", None)
    assert blocks is not None
    assert len(blocks) == TINY_CONFIG["num_layers"]


def test_lora_target_submodule_names_exist():
    model = _tiny_model()
    names = {n for n, _ in model.named_modules()}
    # Upstream names the attention block `attention` and the SwiGLU MLP
    # `feed_forward` (NOT `attn` / `mlp`).
    for pattern in ("attention", "feed_forward"):
        assert any(pattern in n for n in names), f"no module matches {pattern!r}"


# ---------------------------------------------------------------------------
# LoRA target candidates -- canonical leaf nn.Linear names inside ONE block.
#
# Pinned from `Ideogram4Transformer2DModel.layers[0].named_modules()` (verified
# against the tiny config; identical structure at full scale). A later task's
# target-module YAML and the driver default reference THIS list.
#
#   layers.<i>.attention.qkv         fused Q/K/V projection (bias=False)
#   layers.<i>.attention.o           attention output projection (bias=False)
#   layers.<i>.feed_forward.w1       SwiGLU gate projection      (bias=False)
#   layers.<i>.feed_forward.w2       SwiGLU down projection      (bias=False)
#   layers.<i>.feed_forward.w3       SwiGLU up projection        (bias=False)
#   layers.<i>.adaln_modulation      per-block adaLN modulation  (bias=True)
#
# peft-matchable target_modules patterns (suffixes):
#   ["qkv", "o", "w1", "w2", "w3"]   (attention + SwiGLU MLP; the usual LoRA set)
#   add "adaln_modulation" only if conditioning adaptation is desired.
#
# NOTE: attention uses a FUSED qkv projection (one Linear -> 3*hidden), not
# separate to_q/to_k/to_v, so the attention LoRA target is the single `qkv`
# (plus output `o`).
# ---------------------------------------------------------------------------
