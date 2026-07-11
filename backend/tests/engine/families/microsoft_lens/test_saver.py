"""microsoft_lens saver: lora_unet_* keys + ss_ metadata."""
from unittest.mock import patch

import pytest
from peft import LoraConfig, get_peft_model
from safetensors.torch import safe_open

from app.engine.models.families.microsoft_lens.saver import MicrosoftLensSaver
from app.engine.models.families.microsoft_lens.vendor.transformer import (
    LensTransformer2DModel,
)


def _tiny_peft_dit():
    model = LensTransformer2DModel(
        patch_size=2, in_channels=128, out_channels=32, num_layers=1,
        attention_head_dim=8, num_attention_heads=2, inner_dim=16,
        enc_hidden_dim=2880, axes_dims_rope=(2, 2, 4),
        gate_mlp=True, rms_norm=True, multi_layer_encoder_feature=True,
        selected_layer_index=(5, 11, 17, 23),
    )
    cfg = LoraConfig(
        r=8, lora_alpha=8,
        target_modules=["img_qkv", "txt_qkv", "to_out.0", "to_add_out", "w1", "w2", "w3"],
    )
    return get_peft_model(model, cfg)


def test_saver_writes_lora_unet_keys_and_ss_metadata(tmp_path):
    unet = _tiny_peft_dit()
    out = tmp_path / "lens_lora.safetensors"
    MicrosoftLensSaver().save(
        {"unet": unet, "config": {
            "network_rank": 8, "network_alpha": 8, "save_precision": "bf16",
            "learning_rate": 1e-4, "optimizer_type": "adamw",
        }},
        out, metadata={},
    )
    assert out.exists()
    with safe_open(str(out), framework="pt") as f:
        keys = list(f.keys())
        meta = f.metadata()
    assert any(k.startswith("lora_unet_") for k in keys)
    assert any(k.endswith(".lora_down.weight") for k in keys)
    assert any(k.endswith(".lora_up.weight") for k in keys)
    assert any(k.endswith(".alpha") for k in keys)
    assert meta.get("ss_network_dim") == "8"
    assert "ss_network_alpha" in meta
    assert meta.get("modelspec.architecture") == "microsoft_lens"
    down_mods = {k[: -len(".lora_down.weight")] for k in keys if k.endswith(".lora_down.weight")}
    up_mods = {k[: -len(".lora_up.weight")] for k in keys if k.endswith(".lora_up.weight")}
    assert down_mods == up_mods and len(down_mods) > 0


def test_saver_bails_on_non_peft_model(tmp_path):
    import torch.nn as nn
    out = tmp_path / "nope.safetensors"
    MicrosoftLensSaver().save({"unet": nn.Linear(4, 4)}, out, metadata={})
    assert not out.exists()


def test_save_failure_propagates(tmp_path):
    """A safetensors write failure must raise out of save(), not be swallowed.

    Regression test for the silent-failure bug class: a training job must
    not "succeed" while writing no LoRA file.
    """
    unet = _tiny_peft_dit()
    out = tmp_path / "lens_lora.safetensors"

    with patch(
        "app.engine.models.families.microsoft_lens.saver.safe_save_file",
        side_effect=OSError("disk full"),
    ):
        with pytest.raises(OSError, match="disk full"):
            MicrosoftLensSaver().save(
                {"unet": unet, "config": {
                    "network_rank": 8, "network_alpha": 8, "save_precision": "bf16",
                }},
                out, metadata={},
            )
    assert not out.exists()
