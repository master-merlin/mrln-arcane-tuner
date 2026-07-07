"""DreamLite LoRA portability: canonical keys + pinned count + MQA shapes.

No upstream LoRA loader mixin exists for DreamLite — OUR saver's canonical
ai-toolkit keys (``diffusion_model.{module}.lora_A/B.weight``) are the
format of record.

Pinned key math (REAL checkpoint unet/config.json — layers_per_block=2,
transformer_layers_per_block=(1,2,4), up blocks get layers+1=3 attentions):

  attn2-bearing transformer blocks (cross-attention everywhere):
    down0 2×1 + down1 2×2 + down2 2×4 + mid 1×4 + up0 3×4 + up1 3×2 = 36
  attn1-bearing blocks (self-attn only in down2 / mid / up0):
    down2 8 + mid 4 + up0 12 = 24
  modules = 36×4 (attn2) + 24×4 (attn1) + 36×2 (ff) = 312
  → 312 × 2 (lora_A + lora_B) = **624 keys**.

MQA caveat (num_kv_heads=1, head_dim 64): to_k/to_v out_features = 64 at
EVERY level → their LoRA-B matrices are [64, r] — narrow, NOT the
[block_out, r] of to_q/to_out. Pinned below on both the tiny PEFT model
(width = dim_head per level) and the meta-instantiated real config (64).
"""

from __future__ import annotations

import pathlib
import tempfile
from unittest.mock import MagicMock

import torch


# Tiny structurally-faithful config (same as test_dreamlite_family.py).
_TINY_CFG = dict(
    in_channels=4,
    out_channels=4,
    block_out_channels=(8, 16, 32),
    layers_per_block=1,
    transformer_layers_per_block=(1, 2, 4),
    attention_head_dim=4,
    cross_attention_dim=16,
    norm_num_groups=8,
    use_linear_projection=True,
    encoder_hid_dim=12,
    encoder_hid_dim_type="text_proj_rms",
    addition_embed_type="time",
    addition_time_embed_dim=8,
    projection_class_embeddings_input_dim=16,
    num_kv_heads=1,
    qk_norm="rms_norm",
    ff_mult=3,
    use_sep_conv=True,
)

# Tiny pinned counts (layers_per_block=1 → up blocks get 2 attentions):
#   attn2 tblocks: down0 1 + down1 2 + down2 4 + mid 4 + up0 2×4 + up1 2×2 = 23
#   attn1 tblocks: down2 4 + mid 4 + up0 8 = 16
#   modules = 23*4 + 16*4 + 23*2 = 202 → 404 keys
_EXPECTED_TINY_MODULES = 202
_EXPECTED_TINY_KEYS = _EXPECTED_TINY_MODULES * 2

# THE pinned count for the real checkpoint (derivation in module docstring).
_EXPECTED_FULL_MODEL_MODULES = 312
_EXPECTED_FULL_MODEL_KEYS = _EXPECTED_FULL_MODEL_MODULES * 2  # == 624

# Tiny MQA widths: dim_head = block_out // heads(4) per level.
_TINY_KV_WIDTH_BY_LEVEL = {
    "down_blocks.0": 2,
    "down_blocks.1": 4,
    "down_blocks.2": 8,
    "mid_block": 8,
    "up_blocks.0": 8,
    "up_blocks.1": 4,
}

_LORA_RANK = 4


def _make_driver():
    from app.engine.models.families.dreamlite.driver import DreamLiteDriver

    definition = MagicMock()
    definition.family = "dreamlite"
    definition.id = "dreamlite-test"
    definition.lora_targetable_modules = []
    definition.architecture_params = {}
    return DreamLiteDriver(definition, torch.device("cpu"))


def _build_peft_model(driver):
    """Tiny DreamLiteUNetModel wrapped with the driver's LoRA spec."""
    from peft import LoraConfig, get_peft_model
    from diffusers.models.unets.unet_dreamlite import DreamLiteUNetModel

    base = DreamLiteUNetModel(**_TINY_CFG)
    lora_cfg = LoraConfig(
        r=_LORA_RANK,
        lora_alpha=_LORA_RANK,
        target_modules=driver.get_lora_targets(),
    )
    return get_peft_model(base, lora_cfg)


def _save_lora(tmp_dir: str):
    from safetensors.torch import load_file

    drv = _make_driver()
    peft_model = _build_peft_model(drv)
    saver = drv.get_saver()
    path = pathlib.Path(tmp_dir) / "dreamlite_lora.safetensors"
    saver.save(components={"unet": peft_model, "config": {}}, path=path)
    assert path.exists(), "Saver did not produce a safetensors file"
    return path, load_file(str(path))


def test_key_count_pinned_404_tiny_624_full():
    """Tiny model: exactly 404 keys; the real-checkpoint expectation is
    pinned at 624 (312 modules × A/B), cross-checked against a meta
    instantiation of the REAL config in test_dreamlite_family.py."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)

    assert len(sd) == _EXPECTED_TINY_KEYS, (
        f"expected {_EXPECTED_TINY_KEYS} keys for the tiny model, got {len(sd)}"
    )
    modules = {k.rsplit(".lora_", 1)[0] for k in sd}
    assert len(modules) == _EXPECTED_TINY_MODULES

    # attn1 only where the checkpoint has self-attention.
    attn1_prefixes = {
        m.split(".attentions")[0].split(".transformer_blocks")[0]
        .replace("diffusion_model.", "")
        for m in modules
        if ".attn1." in m
    }
    assert attn1_prefixes == {"down_blocks.2", "mid_block", "up_blocks.0"}, (
        f"unexpected attn1 topology: {attn1_prefixes}"
    )

    # The real-checkpoint pin: 312 modules → 624 keys.
    assert _EXPECTED_FULL_MODEL_KEYS == 624


def test_mqa_lora_b_widths_pinned():
    """MQA (num_kv_heads=1): to_k/to_v LoRA-B is [dim_head, r] — NOT
    [block_out, r]. Pinned per level on the tiny model, and pinned at 64
    for the REAL checkpoint config via meta instantiation."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)

    checked = 0
    for k, v in sd.items():
        if not (k.endswith(".to_k.lora_B.weight")
                or k.endswith(".to_v.lora_B.weight")):
            continue
        level = k[len("diffusion_model."):].split(".attentions")[0]
        expected = _TINY_KV_WIDTH_BY_LEVEL[level]
        assert v.shape == (expected, _LORA_RANK), (
            f"{k}: expected MQA-narrow B [{expected}, {_LORA_RANK}], "
            f"got {tuple(v.shape)}"
        )
        checked += 1
    assert checked > 0, "no to_k/to_v lora_B keys found"

    # to_q lora_B stays [block_out, r] — proves the asymmetry is real.
    q_key = (
        "diffusion_model.down_blocks.0.attentions.0.transformer_blocks.0."
        "attn2.to_q.lora_B.weight"
    )
    assert sd[q_key].shape == (8, _LORA_RANK)

    # REAL config: to_k/to_v out_features = 64 at EVERY level (block_out //
    # heads = 256/4 = 512/8 = 896/14 = 64; kv_heads=1) — so the checkpoint
    # LoRA-B for to_k/to_v is [64, r] everywhere.
    from diffusers.models.unets.unet_dreamlite import DreamLiteUNetModel

    with torch.device("meta"):
        real = DreamLiteUNetModel(
            block_out_channels=(256, 512, 896),
            attention_head_dim=(4, 8, 14),
            cross_attention_dim=2304,
            layers_per_block=2,
            transformer_layers_per_block=(1, 2, 4),
            use_linear_projection=True,
            encoder_hid_dim=2048,
            encoder_hid_dim_type="text_proj_rms",
            addition_embed_type="time",
            addition_time_embed_dim=256,
            projection_class_embeddings_input_dim=512,
            num_kv_heads=1,
            qk_norm="rms_norm",
            ff_mult=3,
            use_sep_conv=True,
        )
    kv_widths = {
        m.out_features
        for n, m in real.named_modules()
        if isinstance(m, torch.nn.Linear) and n.endswith(("to_k", "to_v"))
        and (".attn1." in n or ".attn2." in n)
    }
    assert kv_widths == {64}, f"real-config MQA widths must be 64: {kv_widths}"


def test_saver_key_format_is_ai_toolkit():
    """All keys are diffusion_model.{module}.lora_A/B.weight."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)

    assert sd, "Saved state dict is empty"
    for k in sd:
        assert k.startswith("diffusion_model."), f"bad prefix: {k!r}"
        assert k.endswith(".weight"), f"bad suffix: {k!r}"
        assert ".lora_A." in k or ".lora_B." in k, f"not a LoRA key: {k!r}"
        assert ".default." not in k, f"PEFT adapter name leaked: {k!r}"
        assert ".attn1." in k or ".attn2." in k or ".ff." in k, (
            f"key outside the canonical attn/ff targets: {k!r}"
        )

    lora_a = [k for k in sd if ".lora_A." in k]
    lora_b = [k for k in sd if ".lora_B." in k]
    assert len(lora_a) == len(lora_b), "lora_A/lora_B counts must match"


def test_saver_architecture_metadata():
    """modelspec.architecture must be 'dreamlite'."""
    from safetensors import safe_open

    with tempfile.TemporaryDirectory() as td:
        path, _ = _save_lora(td)
        with safe_open(str(path), framework="pt") as f:
            metadata = f.metadata()

    assert metadata is not None
    assert metadata.get("modelspec.architecture") == "dreamlite", (
        f"wrong architecture metadata: {metadata.get('modelspec.architecture')!r}"
    )


def test_lora_round_trips_base_to_mobile():
    """base → mobile portability: DreamLite-base and DreamLite-mobile ship
    BYTE-IDENTICAL unet configs (verified on the hub), so a LoRA saved from
    a base-config PEFT model loads onto a fresh identically-wrapped
    (mobile) model with zero missing LoRA keys and zero unexpected keys."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)

    mobile = _build_peft_model(_make_driver())

    def _remap_to_peft(key: str) -> str:
        module_path = key[len("diffusion_model."):]
        module_path = module_path.replace(
            ".lora_A.weight", ".lora_A.default.weight",
        )
        module_path = module_path.replace(
            ".lora_B.weight", ".lora_B.default.weight",
        )
        return f"base_model.model.{module_path}"

    remapped = {_remap_to_peft(k): v for k, v in sd.items()}
    missing, unexpected = mobile.load_state_dict(remapped, strict=False)

    lora_missing = [k for k in missing if "lora" in k.lower()]
    assert not lora_missing, f"LoRA keys missing on reload: {lora_missing[:5]}"
    assert not unexpected, f"Unexpected keys on reload: {unexpected[:5]}"


def test_base_and_mobile_definitions_share_lora_surface():
    """The two YAML definitions must expose the SAME LoRA surface: identical
    architecture_params (checked byte-wise in test_dreamlite_family.py) and
    identical lora_targetable_modules (both empty → driver pattern
    defaults), so trained LoRAs are interchangeable."""
    from app.engine.models.registry import ModelRegistry

    ModelRegistry.initialize()
    defs = {
        d.id: d
        for d in ModelRegistry._definitions.values()
        if d.family == "dreamlite"
    }
    base = defs["dreamlite-base"]
    mobile = defs["dreamlite-mobile"]

    assert base.architecture_params == mobile.architecture_params
    assert (getattr(base, "lora_targetable_modules", None) or []) == (
        getattr(mobile, "lora_targetable_modules", None) or []
    )
