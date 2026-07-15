"""Bernini-R saver test — wan-canonical keys, bernini-r arch label.

Bernini-R adds ZERO new weight modules (recon §9): its checkpoints are 100%
stock Wan, and ComfyUI's official Bernini-R workflow loads STOCK Wan LoRA keys.
So the exported tensor key set MUST be byte-identical to the wan21 single-expert
export for the same module set. Only the ``modelspec.architecture`` metadata
label differs (provenance).

We PEFT-wrap one fake 2-block module mirroring ``WanTransformerBlock`` submodule
names, save it with BOTH savers, and assert the key sets are equal.
"""

import torch.nn as nn
from peft import LoraConfig, get_peft_model
from safetensors.torch import safe_open

from app.engine.models.families.bernini_r.saver import BerniniRSaver
from app.engine.models.families.wan21.saver import Wan21Saver


class _Attn(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Identity()])


class _FFN(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        proj_mod = nn.Module()
        proj_mod.proj = nn.Linear(dim, dim * 2)
        self.net = nn.ModuleList([proj_mod, nn.Identity(), nn.Linear(dim * 2, dim)])


class _Block(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.attn1 = _Attn(dim)
        self.attn2 = _Attn(dim)
        self.ffn = _FFN(dim)


class _FakeWan(nn.Module):
    def __init__(self, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])


def _peft_wrap(model):
    targets = [
        "attn1.to_q",
        "attn1.to_k",
        "attn1.to_v",
        "attn1.to_out.0",
        "attn2.to_q",
        "attn2.to_k",
        "attn2.to_v",
        "attn2.to_out.0",
        "ffn.net.0.proj",
        "ffn.net.2",
    ]
    cfg = LoraConfig(r=8, lora_alpha=8, target_modules=targets)
    return get_peft_model(model, cfg)


def _save_and_read(saver, model, path):
    saver.save({"unet": model, "config": {}}, path, metadata={})
    with safe_open(str(path), framework="pt") as f:
        return set(f.keys()), dict(f.metadata())


def test_bernini_saver_keys_equal_wan21(tmp_path):
    model = _peft_wrap(_FakeWan(n_blocks=2))

    keys_b, meta_b = _save_and_read(
        BerniniRSaver(mode="t2v"), model, tmp_path / "bernini_r_lora.safetensors"
    )
    keys_w, meta_w = _save_and_read(
        Wan21Saver(mode="t2v"), model, tmp_path / "wan21_lora.safetensors"
    )

    # Byte-identical tensor key set → wan LoRA loads into Bernini-R verbatim.
    assert keys_b == keys_w
    assert keys_b, "no keys exported"
    # And they are the wan-canonical ComfyUI diffusion_model.blocks.* keys.
    assert all(k.startswith("diffusion_model.blocks.") for k in keys_b)

    # Only the provenance label differs.
    assert meta_b.get("modelspec.architecture") == "bernini-r-t2v"
    assert meta_w.get("modelspec.architecture") == "wan2.1-t2v"


def test_bernini_saver_bails_on_non_peft_model(tmp_path):
    out = tmp_path / "nope.safetensors"
    BerniniRSaver().save({"unet": nn.Linear(4, 4)}, out, metadata={})
    assert not out.exists()
