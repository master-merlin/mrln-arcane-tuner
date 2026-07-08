"""hv15 LoRA-target tests against a REAL (tiny) instantiated transformer.

The driver's default targets are FULL ``transformer_blocks.{i}.*`` paths (not
bare suffixes): the 1.5 transformer's token refiner
(``context_embedder.token_refiner.refiner_blocks.N.attn.to_q`` …) carries the
SAME ``attn.to_q``/``ff.net.*`` module names, so suffix-only targets would
silently wrap the refiner too. PEFT is applied to a tiny real model and the
wrapped module set is asserted exactly.
"""

import pytest
import torch

from app.engine.models.families.hunyuan_video15.driver import (
    HV15_BLOCK_LORA_SUFFIXES,
    HV15_NUM_LAYERS_DEFAULT,
    Hv15Driver,
    hv15_lora_target_paths,
)

# Tiny config — real class, laptop-sized (rope_axes_dim sums to head_dim).
TINY_CFG = dict(
    in_channels=9,  # 4 + 4 + 1 (keeps the 65-ch latents/cond/mask ratio)
    out_channels=4,
    num_attention_heads=2,
    attention_head_dim=8,
    num_layers=1,
    num_refiner_layers=1,
    mlp_ratio=2.0,
    patch_size=1,
    patch_size_t=1,
    text_embed_dim=16,
    text_embed_2_dim=8,
    image_embed_dim=8,
    rope_axes_dim=(2, 2, 4),
    rope_theta=256.0,
    task_type="t2v",
    use_meanflow=False,
)


class _Defn:
    def __init__(self, mode: str = "t2v", num_layers: int = 1):
        self.architecture_params = {
            "mode": mode,
            "transformer.num_layers": num_layers,
        }
        self.lora_targetable_modules: list[str] = []


def _tiny_model():
    from diffusers import HunyuanVideo15Transformer3DModel

    return HunyuanVideo15Transformer3DModel(**TINY_CFG)


def test_default_targets_are_full_paths_54_blocks():
    driver = Hv15Driver(_Defn(num_layers=HV15_NUM_LAYERS_DEFAULT), torch.device("cpu"))
    targets = driver.get_lora_targets()
    # 54 blocks x 12 modules = 648 full-path targets.
    assert len(targets) == 54 * 12 == 648
    assert "transformer_blocks.0.attn.to_q" in targets
    assert "transformer_blocks.53.ff_context.net.2" in targets
    # No bare suffixes (they would leak into the token refiner).
    assert all(t.startswith("transformer_blocks.") for t in targets)


def test_targets_match_modes_identically():
    """t2v and i2v share ONE transformer layout — identical targets (the
    portability precondition)."""
    t2v = Hv15Driver(_Defn("t2v", 54), torch.device("cpu")).get_lora_targets()
    i2v = Hv15Driver(_Defn("i2v", 54), torch.device("cpu")).get_lora_targets()
    assert t2v == i2v


def test_definition_targets_override_defaults():
    defn = _Defn()
    defn.lora_targetable_modules = ["custom.module.a"]
    driver = Hv15Driver(defn, torch.device("cpu"))
    assert driver.get_lora_targets() == ["custom.module.a"]


def test_targets_match_real_tiny_model_modules():
    """Every generated target path exists as a Linear in the real model."""
    model = _tiny_model()
    linear_names = {
        n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
    }
    for target in hv15_lora_target_paths(1):
        assert target in linear_names, f"target {target!r} matches no Linear"


def test_peft_wraps_exactly_the_block_modules_and_not_the_refiner():
    """PEFT-wrap the tiny model → exactly the 12 per-block modules get LoRA;
    the token refiner / embedders / norms / proj_out stay untouched."""
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import LoraLayer

    model = _tiny_model()
    targets = hv15_lora_target_paths(1)
    peft_model = get_peft_model(
        model, LoraConfig(r=2, lora_alpha=2, target_modules=targets)
    )

    wrapped = {
        n.replace("base_model.model.", "")
        for n, m in peft_model.named_modules()
        if isinstance(m, LoraLayer)
    }
    assert wrapped == set(targets)
    # The refiner's look-alike attn/ff modules were NOT wrapped.
    assert not any("token_refiner" in n for n in wrapped)
    assert not any("context_embedder" in n for n in wrapped)
    assert not any("image_embedder" in n for n in wrapped)

    # Pinned key count: 12 modules x (lora_A + lora_B) = 24 tensors per block.
    from peft import get_peft_model_state_dict

    sd = get_peft_model_state_dict(peft_model)
    lora_keys = [k for k in sd if "lora_A" in k or "lora_B" in k]
    assert len(lora_keys) == 24


@pytest.mark.parametrize("suffix", HV15_BLOCK_LORA_SUFFIXES)
def test_each_suffix_present_per_block(suffix):
    assert f"transformer_blocks.0.{suffix}" in hv15_lora_target_paths(1)
