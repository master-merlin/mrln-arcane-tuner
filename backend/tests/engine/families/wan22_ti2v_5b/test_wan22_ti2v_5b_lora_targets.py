"""WAN 2.2 TI2V-5B LoRA-target tests (mirrors wan21/wan22's equivalents).

Unlike wan21 i2v / wan22 i2v (which add ``attn2.add_k_proj``/``add_v_proj``),
TI2V-5B never adds the image cross-attention targets — the driver keeps
``is_i2v`` at the base False (see ``driver.py``'s ``__init__`` docstring), and
the real checkpoint has no ``added_kv_proj_dim`` at all (no such modules exist
on the transformer to target).
"""

from __future__ import annotations

import torch

from app.engine.models.families.wan22_ti2v_5b.driver import Wan22Ti2v5bDriver
from app.engine.models.families.wan_shared.driver_base import WAN_T2V_LORA_TARGETS


class _Defn:
    def __init__(self):
        self.architecture_params = {"mode": "both", "te.max_length": 512}
        self.lora_targetable_modules: list[str] = []


def test_default_targets_are_plain_t2v_set_no_image_cross_attn():
    driver = Wan22Ti2v5bDriver(_Defn(), torch.device("cpu"))
    targets = driver.get_lora_targets()
    assert set(targets) == set(WAN_T2V_LORA_TARGETS)
    assert "attn2.add_k_proj" not in targets
    assert "attn2.add_v_proj" not in targets


def test_definition_targets_override_defaults():
    defn = _Defn()
    defn.lora_targetable_modules = ["custom.module.a", "custom.module.b"]
    driver = Wan22Ti2v5bDriver(defn, torch.device("cpu"))
    assert driver.get_lora_targets() == ["custom.module.a", "custom.module.b"]


def test_targets_match_real_wan_block_module_names():
    """The target suffixes must correspond to real WanTransformerBlock modules
    (the 5B has no added_kv_proj_dim, so this block is built without it)."""
    from diffusers.models.transformers.transformer_wan import WanTransformerBlock

    blk = WanTransformerBlock(
        dim=16, ffn_dim=32, num_heads=2, cross_attn_norm=True, added_kv_proj_dim=None
    )
    module_names = {n for n, _ in blk.named_modules()}

    for target in WAN_T2V_LORA_TARGETS:
        assert any(n == target or n.endswith("." + target) for n in module_names), (
            f"LoRA target {target!r} matches no module in WanTransformerBlock"
        )
    # And no image cross-attn modules exist without added_kv_proj_dim.
    assert not any("add_k_proj" in n or "add_v_proj" in n for n in module_names)
