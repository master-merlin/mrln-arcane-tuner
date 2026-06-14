"""WAN 2.2 saver test — TWO ComfyUI-format files (high + low) + per-file metadata.

Two tiny fake transformers (mirroring ``WanTransformerBlock`` submodule names)
are PEFT-wrapped and saved together. We assert that the saver writes
``{stem}_high_noise`` and ``{stem}_low_noise`` files, each with the ComfyUI
``diffusion_model.blocks.*`` keys and the per-file ``modelspec.architecture``
(``wan2.2-{t2v}-{high,low}``).
"""

import torch.nn as nn
from peft import LoraConfig, get_peft_model
from safetensors.torch import safe_open

from app.engine.models.families.wan22.saver import Wan22Saver


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


def test_saver_writes_two_files_with_comfy_keys_and_metadata(tmp_path):
    high = _peft_wrap(_FakeWan(n_blocks=2))
    low = _peft_wrap(_FakeWan(n_blocks=2))
    out = tmp_path / "wan22_t2v_lora.safetensors"

    Wan22Saver(mode="t2v").save(
        {
            "unet_high": high,
            "unet_low": low,
            "config": {"save_precision": "bf16", "learning_rate": 1e-4},
        },
        out,
        metadata={},
    )

    high_path = tmp_path / "wan22_t2v_lora_high_noise.safetensors"
    low_path = tmp_path / "wan22_t2v_lora_low_noise.safetensors"
    assert high_path.exists(), "high-noise file not written"
    assert low_path.exists(), "low-noise file not written"
    # The single original path must NOT be written (only the two expert files).
    assert not out.exists()

    for path, expert in ((high_path, "high"), (low_path, "low")):
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
            meta = f.metadata()
        assert all(k.startswith("diffusion_model.blocks.") for k in keys), keys
        assert any(".self_attn.q.lora_down.weight" in k for k in keys)
        assert any(".cross_attn.o.lora_up.weight" in k for k in keys)
        assert any(".ffn.0.lora_down.weight" in k for k in keys)
        assert any(".ffn.2.lora_up.weight" in k for k in keys)
        assert not any("attn1" in k or "to_q" in k for k in keys)
        assert meta.get("modelspec.architecture") == f"wan2.2-t2v-{expert}"
        assert meta.get("wan22_expert") == expert
        assert meta.get("ss_network_dim") == "8"
        # down/up modules pair up.
        down = {
            k[: -len(".lora_down.weight")]
            for k in keys
            if k.endswith(".lora_down.weight")
        }
        up = {
            k[: -len(".lora_up.weight")] for k in keys if k.endswith(".lora_up.weight")
        }
        assert down == up and len(down) > 0


def test_saver_i2v_includes_image_projections(tmp_path):
    high = _peft_wrap(_FakeWan(n_blocks=2, i2v=True), i2v=True)
    low = _peft_wrap(_FakeWan(n_blocks=2, i2v=True), i2v=True)
    out = tmp_path / "wan22_i2v_lora.safetensors"
    Wan22Saver(mode="i2v").save(
        {"unet_high": high, "unet_low": low, "config": {}}, out, metadata={}
    )

    for expert in ("high", "low"):
        path = tmp_path / f"wan22_i2v_lora_{expert}_noise.safetensors"
        assert path.exists()
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
            meta = f.metadata()
        assert meta.get("modelspec.architecture") == f"wan2.2-i2v-{expert}"
        assert any(".cross_attn.k_img." in k for k in keys)
        assert any(".cross_attn.v_img." in k for k in keys)


def test_saver_skips_non_peft_expert(tmp_path):
    # A non-PEFT high + a real PEFT low → only the low file is written.
    low = _peft_wrap(_FakeWan(n_blocks=1))
    out = tmp_path / "x.safetensors"
    Wan22Saver(mode="t2v").save(
        {"unet_high": nn.Linear(4, 4), "unet_low": low, "config": {}}, out, metadata={}
    )
    assert not (tmp_path / "x_high_noise.safetensors").exists()
    assert (tmp_path / "x_low_noise.safetensors").exists()
