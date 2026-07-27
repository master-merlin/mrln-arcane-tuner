"""WAN 2.1 LoRA-target tests.

The driver returns the WAN self/cross-attn + ffn targets. T2V and I2V share
the exact same pattern-default surface (W3.T9): the image cross-attention
projections (``add_k_proj`` / ``add_v_proj``) are DEAD on wan21 i2v — nothing
ever computes a CLIP image embedding to feed them (the loader no longer even
loads the CLIP encoder), so targeting them trained nothing but zero-delta
tensors and wasted optimizer slots. The actual ``WanTransformerBlock`` module
names are asserted to match the shipped targets so PEFT's suffix matching
will hit real modules.
"""

import torch

from app.engine.models.families.wan21.driver import Wan21Driver
from app.engine.models.families.wan_shared.driver_base import WAN_T2V_LORA_TARGETS


class _Defn:
    def __init__(self, mode: str):
        self.architecture_params = {"mode": mode, "te.max_length": 512}
        self.lora_targetable_modules: list[str] = []


def test_t2v_targets():
    driver = Wan21Driver(_Defn("t2v"), torch.device("cpu"))
    targets = driver.get_lora_targets()
    assert set(targets) == set(WAN_T2V_LORA_TARGETS)
    # No image cross-attn for t2v.
    assert "attn2.add_k_proj" not in targets
    assert "attn2.add_v_proj" not in targets


def test_i2v_targets_match_t2v_no_dead_image_projections():
    """I2V's pattern-default targets are IDENTICAL to t2v's (W3.T9) — the
    image cross-attn projections are dead weight, not a real i2v feature."""
    driver = Wan21Driver(_Defn("i2v"), torch.device("cpu"))
    targets = driver.get_lora_targets()
    assert set(targets) == set(WAN_T2V_LORA_TARGETS)
    assert "attn2.add_k_proj" not in targets
    assert "attn2.add_v_proj" not in targets


def test_definition_targets_override_defaults():
    defn = _Defn("t2v")
    defn.lora_targetable_modules = ["custom.module.a", "custom.module.b"]
    driver = Wan21Driver(defn, torch.device("cpu"))
    assert driver.get_lora_targets() == ["custom.module.a", "custom.module.b"]


def test_targets_match_real_wan_block_module_names():
    """The target suffixes must correspond to real WanTransformerBlock modules."""
    from diffusers.models.transformers.transformer_wan import WanTransformerBlock

    blk = WanTransformerBlock(
        dim=16, ffn_dim=32, num_heads=2, cross_attn_norm=True, added_kv_proj_dim=16
    )
    module_names = {n for n, _ in blk.named_modules()}

    for target in WAN_T2V_LORA_TARGETS:
        assert any(n == target or n.endswith("." + target) for n in module_names), (
            f"LoRA target {target!r} matches no module in WanTransformerBlock"
        )
