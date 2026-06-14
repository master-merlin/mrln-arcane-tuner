"""WAN 2.1 saver test — ComfyUI ``diffusion_model.blocks.*`` keys + metadata.

A tiny fake 2-block module mirroring the ``WanTransformerBlock`` submodule names
is PEFT-wrapped and saved. We assert the converted ComfyUI key naming
(``self_attn`` / ``cross_attn`` / ``ffn.{0,2}`` + ``lora_{down,up}``) and the
``modelspec.architecture`` metadata. No real transformer needed.
"""

import torch.nn as nn
from peft import LoraConfig, get_peft_model
from safetensors.torch import safe_open

from app.engine.models.families.wan21.saver import Wan21Saver


class _Attn(nn.Module):
    def __init__(self, dim=16, i2v=False):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Identity()])
        if i2v:
            self.add_k_proj = nn.Linear(dim, dim)
            self.add_v_proj = nn.Linear(dim, dim)


class _FFN(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        # FeedForward.net == [proj-act(GELU wrapper), dropout, linear] in
        # diffusers; we only need .net.0.proj and .net.2 to exist as Linears.
        proj_mod = nn.Module()
        proj_mod.proj = nn.Linear(dim, dim * 2)
        self.net = nn.ModuleList([proj_mod, nn.Identity(), nn.Linear(dim * 2, dim)])


class _Block(nn.Module):
    def __init__(self, dim=16, i2v=False):
        super().__init__()
        self.attn1 = _Attn(dim, i2v=False)
        self.attn2 = _Attn(dim, i2v=i2v)
        self.ffn = _FFN(dim)


class _FakeWan(nn.Module):
    def __init__(self, n_blocks=2, i2v=False):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(i2v=i2v) for _ in range(n_blocks)])


def _peft_wrap(model, i2v=False):
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
    if i2v:
        targets += ["attn2.add_k_proj", "attn2.add_v_proj"]
    cfg = LoraConfig(r=8, lora_alpha=8, target_modules=targets)
    return get_peft_model(model, cfg)


def test_saver_writes_comfy_keys_and_metadata(tmp_path):
    model = _peft_wrap(_FakeWan(n_blocks=2))
    out = tmp_path / "wan21_t2v_lora.safetensors"
    Wan21Saver(mode="t2v").save(
        {"unet": model, "config": {"save_precision": "bf16", "learning_rate": 1e-4}},
        out,
        metadata={},
    )
    assert out.exists()

    with safe_open(str(out), framework="pt") as f:
        keys = list(f.keys())
        meta = f.metadata()

    # All keys are ComfyUI-format diffusion_model.blocks.*
    assert all(k.startswith("diffusion_model.blocks.") for k in keys), keys
    assert any(".lora_down.weight" in k for k in keys)
    assert any(".lora_up.weight" in k for k in keys)
    # Self-attn / cross-attn / ffn sub-names present.
    assert any(".self_attn.q.lora_down.weight" in k for k in keys)
    assert any(".cross_attn.o.lora_up.weight" in k for k in keys)
    assert any(".ffn.0.lora_down.weight" in k for k in keys)
    assert any(".ffn.2.lora_up.weight" in k for k in keys)
    # No leftover diffusers attn1/attn2/to_q naming.
    assert not any("attn1" in k or "to_q" in k for k in keys)

    assert meta.get("modelspec.architecture") == "wan2.1-t2v"
    assert meta.get("ss_network_dim") == "8"
    assert "ss_network_alpha" in meta

    # down/up modules pair up.
    down = {
        k[: -len(".lora_down.weight")] for k in keys if k.endswith(".lora_down.weight")
    }
    up = {k[: -len(".lora_up.weight")] for k in keys if k.endswith(".lora_up.weight")}
    assert down == up and len(down) > 0


def test_saver_i2v_includes_image_projections_and_arch(tmp_path):
    model = _peft_wrap(_FakeWan(n_blocks=2, i2v=True), i2v=True)
    out = tmp_path / "wan21_i2v_lora.safetensors"
    Wan21Saver(mode="i2v").save({"unet": model, "config": {}}, out, metadata={})
    assert out.exists()

    with safe_open(str(out), framework="pt") as f:
        keys = list(f.keys())
        meta = f.metadata()

    assert meta.get("modelspec.architecture") == "wan2.1-i2v"
    # Image cross-attn projections mapped to k_img / v_img.
    assert any(".cross_attn.k_img." in k for k in keys)
    assert any(".cross_attn.v_img." in k for k in keys)


def test_saver_bails_on_non_peft_model(tmp_path):
    out = tmp_path / "nope.safetensors"
    Wan21Saver().save({"unet": nn.Linear(4, 4)}, out, metadata={})
    assert not out.exists()
