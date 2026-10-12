"""diffusers 0.41.0's Ideogram4 DiT: instantiation + module-name + LoRA-target contract.

Uses a tiny synthetic config (no gated weights) to lock the submodule names
the LoRA target list depends on (LANE-150: the vendored copy is retired).
"""

from __future__ import annotations

import torch
from accelerate import init_empty_weights
from diffusers import Ideogram4Transformer2DModel

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.ideogram4.driver import IdeogramV4Driver

# Tiny synthetic config in the release spelling (transformer_ideogram4.py:316-328).
# Values are kept small so the model instantiates in seconds on CPU under
# `init_empty_weights`.
#
# Constraints honoured by these values:
#   - head_dim = attention_head_dim = 8 (even).
#   - MRoPE uses indices up to max(mrope_section) * 3 into an inv_freq vector of
#     length head_dim // 2 == 4, so each mrope_section entry must satisfy
#     section * 3 <= 4  ->  section == 1. Hence mrope_section=(1, 1, 1).
TINY_CONFIG = {
    "attention_head_dim": 8,
    "num_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 32,
    "adaln_dim": 8,
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
    assert len(model.layers) == TINY_CONFIG["num_layers"]


def test_lora_target_submodule_names_exist():
    model = _tiny_model()
    names = {n for n, _ in model.named_modules()}
    # Upstream names the attention block `attention` and the SwiGLU MLP
    # `feed_forward` (NOT `attn` / `mlp`).
    for pattern in ("attention", "feed_forward"):
        assert any(pattern in n for n in names), f"no module matches {pattern!r}"


# ---------------------------------------------------------------------------
# LoRA target candidates -- canonical leaf nn.Linear names inside ONE block of
# the release class (transformer_ideogram4.py:146-151, :175-181):
#
#   layers.<i>.attention.to_q / to_k / to_v   split Q/K/V projections (bias=False)
#   layers.<i>.attention.to_out.0             attention output projection (bias=False)
#   layers.<i>.feed_forward.w1 / w2 / w3      SwiGLU gate / down / up (bias=False)
#   layers.<i>.adaln_modulation               per-block adaLN modulation (bias=True)
#
# The definition's released list ["qkv", "o", "w1", "w2", "w3"] names the ported
# FUSED projections; ``IdeogramV4Driver.get_lora_targets`` expands it to the
# split names, and the saver re-fuses the exported keys.
# ---------------------------------------------------------------------------


def test_every_driver_lora_target_is_a_linear_of_the_release_class():
    model = _tiny_model()
    drv = IdeogramV4Driver(
        ModelDefinition(id="x", family="ideogram4", name="X", lora_targetable_modules=["qkv", "o", "w1", "w2", "w3"]),
        torch.device("cpu"),
    )
    linears = {n for n, m in model.layers[0].named_modules() if isinstance(m, torch.nn.Linear)}
    for target in drv.get_lora_targets():
        assert any(n == target or n.endswith("." + target) for n in linears), (
            f"LoRA target {target!r} names no Linear in a release block: {sorted(linears)}"
        )
    # The ported fused names no longer exist on the release class.
    assert not any(n.endswith(".qkv") or n.endswith(".o") for n in linears)
