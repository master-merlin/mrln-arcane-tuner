"""WAN 2.2 LoRA-target tests.

Mirror of ``wan21/test_wan21_lora_targets.py``. WAN 2.2's :class:`Wan22Driver`
inherits ``get_lora_targets`` from the shared :class:`WanDriverBase`, so the
same self/cross-attn + ffn surface applies, with I2V adding the image
cross-attention projections (``add_k_proj`` / ``add_v_proj``). This test was
absent for wan22 (W2-C recon 2026-07-11) — added so the WAN 2.2 target surface
is pinned exactly like WAN 2.1's, and a definition-list override is proven to
win over the driver defaults (the enrich-clobber guard's runtime half).
"""

import torch

from app.engine.models.families.wan22.driver import Wan22Driver
from app.engine.models.families.wan_shared.driver_base import (
    WAN_I2V_EXTRA_LORA_TARGETS,
    WAN_T2V_LORA_TARGETS,
)


class _Defn:
    def __init__(self, mode: str):
        self.architecture_params = {"mode": mode, "te.max_length": 512}
        self.lora_targetable_modules: list[str] = []


def test_t2v_targets():
    driver = Wan22Driver(_Defn("t2v"), torch.device("cpu"))
    targets = driver.get_lora_targets()
    assert set(targets) == set(WAN_T2V_LORA_TARGETS)
    # No image cross-attn for t2v.
    assert "attn2.add_k_proj" not in targets
    assert "attn2.add_v_proj" not in targets


def test_i2v_targets_add_image_projections():
    driver = Wan22Driver(_Defn("i2v"), torch.device("cpu"))
    targets = driver.get_lora_targets()
    for t in WAN_T2V_LORA_TARGETS:
        assert t in targets
    for t in WAN_I2V_EXTRA_LORA_TARGETS:
        assert t in targets
    assert "attn2.add_k_proj" in targets
    assert "attn2.add_v_proj" in targets


def test_definition_targets_override_defaults():
    defn = _Defn("t2v")
    defn.lora_targetable_modules = ["custom.module.a", "custom.module.b"]
    driver = Wan22Driver(defn, torch.device("cpu"))
    assert driver.get_lora_targets() == ["custom.module.a", "custom.module.b"]


def test_targets_match_real_wan_block_module_names():
    """The target suffixes must correspond to real WanTransformerBlock modules."""
    from diffusers.models.transformers.transformer_wan import WanTransformerBlock

    blk = WanTransformerBlock(
        dim=16, ffn_dim=32, num_heads=2, cross_attn_norm=True, added_kv_proj_dim=16
    )
    module_names = {n for n, _ in blk.named_modules()}

    for target in WAN_T2V_LORA_TARGETS + WAN_I2V_EXTRA_LORA_TARGETS:
        assert any(n == target or n.endswith("." + target) for n in module_names), (
            f"LoRA target {target!r} matches no module in WanTransformerBlock"
        )
