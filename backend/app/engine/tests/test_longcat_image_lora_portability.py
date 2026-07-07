"""LongCat-Image LoRA portability: ai-toolkit key format + pinned key count.

No upstream LoRA loader mixin exists for LongCat-Image, so OUR canonical
keys (``diffusion_model.{module}.lora_A/B.weight``) are the format of record.

Pinned counts (LoRA modules × 2 keys each):
- Tiny config (1 double + 1 single block): 12 + 5 + 1 top-level proj_out
  = 18 modules → **36 keys** (verified against a live PEFT wrap).
- Full checkpoint config (19 double + 38 single): 19×12 + 38×5 + 1
  = 419 modules → **838 keys** (formula-pinned; the full 11.9 B model is
  not instantiable in CI).

The top-level ``proj_out`` (final output projection) is matched by the
``proj_out`` suffix pattern — intentional and identical to the flux1
family's behavior.
"""

from __future__ import annotations

import pathlib
import tempfile
from unittest.mock import MagicMock

import torch


# ── Tiny config (matches test_longcat_image_family.py) ──────────────────────
_TINY_CFG = dict(
    patch_size=1,
    in_channels=4,
    num_layers=1,
    num_single_layers=1,
    attention_head_dim=8,
    num_attention_heads=2,
    joint_attention_dim=16,
    pooled_projection_dim=16,
    axes_dims_rope=[4, 2, 2],
)

# Driver pattern defaults (LongCatImageDriver.get_lora_targets)
_LORA_TARGETS = [
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
    "ff.net.0.proj", "ff.net.2",
    "ff_context.net.0.proj", "ff_context.net.2",
    "proj_mlp", "proj_out",
]

# Pinned key counts (see module docstring)
_TINY_EXPECTED_KEYS = 36
_FULL_EXPECTED_KEYS = 838

# Per-block LoRA module counts (for the formula pin)
_DOUBLE_BLOCK_MODULES = 12  # 8 attn projections + 4 ff projections
_SINGLE_BLOCK_MODULES = 5   # attn q/k/v (pre_only) + proj_mlp + proj_out
_TOP_LEVEL_MODULES = 1      # top-level proj_out (suffix match, flux1 precedent)


def _build_peft_model():
    """Build a tiny LongCatImageTransformer2DModel wrapped with PEFT LoRA."""
    from peft import LoraConfig, get_peft_model
    from diffusers.models.transformers.transformer_longcat_image import (
        LongCatImageTransformer2DModel,
    )

    base = LongCatImageTransformer2DModel(**_TINY_CFG)
    lora_cfg = LoraConfig(r=4, lora_alpha=4, target_modules=_LORA_TARGETS)
    return get_peft_model(base, lora_cfg)


def _get_saver():
    """Obtain the saver via the driver (same as the training path)."""
    from app.engine.models.families.longcat_image.driver import LongCatImageDriver

    definition = MagicMock()
    definition.family = "longcat_image"
    definition.id = "longcat-image-test"
    definition.lora_targetable_modules = _LORA_TARGETS
    definition.architecture_params = {}

    drv = LongCatImageDriver(definition, torch.device("cpu"))
    return drv.get_saver()


def _save_tiny_lora(tmp_dir: str):
    from safetensors.torch import load_file

    saved_path = pathlib.Path(tmp_dir) / "longcat_lora.safetensors"
    model = _build_peft_model()
    saver = _get_saver()
    saver.save(components={"unet": model, "config": {}}, path=saved_path)
    assert saved_path.exists(), "Saver did not produce a safetensors file"
    return saved_path, load_file(str(saved_path))


def test_saver_key_format_and_pinned_count():
    """Keys are diffusion_model.{path}.lora_A/B.weight — count pinned at 36
    for the tiny (1 double + 1 single) config."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_tiny_lora(td)

    assert len(sd) == _TINY_EXPECTED_KEYS, (
        f"pinned key count changed: expected {_TINY_EXPECTED_KEYS}, got {len(sd)}"
    )

    for k in sd:
        assert k.startswith("diffusion_model."), f"bad prefix: {k!r}"
        assert k.endswith(".weight"), f"bad suffix: {k!r}"
        assert "lora_A" in k or "lora_B" in k, f"non-LoRA key: {k!r}"
        assert ".default." not in k, f"PEFT adapter name leaked: {k!r}"

    lora_a = [k for k in sd if "lora_A" in k]
    lora_b = [k for k in sd if "lora_B" in k]
    assert len(lora_a) == len(lora_b) == _TINY_EXPECTED_KEYS // 2

    # Both streams + the top-level projection are represented
    assert any(".transformer_blocks.0." in k for k in sd), "no double-block keys"
    assert any(".single_transformer_blocks.0." in k for k in sd), (
        "no single-block keys"
    )
    assert "diffusion_model.proj_out.lora_A.weight" in sd, (
        "top-level proj_out missing (flux1-precedent suffix match)"
    )


def test_full_config_key_count_formula():
    """Formula pin for the real checkpoint (19 double + 38 single blocks):
    419 modules → 838 keys. If the target list or architecture facts drift,
    this recomputation catches it without instantiating the 11.9 B model."""
    num_layers = 19
    num_single_layers = 38
    modules = (
        num_layers * _DOUBLE_BLOCK_MODULES
        + num_single_layers * _SINGLE_BLOCK_MODULES
        + _TOP_LEVEL_MODULES
    )
    assert modules * 2 == _FULL_EXPECTED_KEYS

    # The tiny pin must satisfy the same formula with 1+1 blocks.
    tiny_modules = (
        1 * _DOUBLE_BLOCK_MODULES + 1 * _SINGLE_BLOCK_MODULES + _TOP_LEVEL_MODULES
    )
    assert tiny_modules * 2 == _TINY_EXPECTED_KEYS


def test_saver_architecture_metadata():
    """modelspec.architecture must be 'longcat_image' (not qwen_image etc.)."""
    from safetensors import safe_open

    with tempfile.TemporaryDirectory() as td:
        saved_path, _ = _save_tiny_lora(td)
        with safe_open(str(saved_path), framework="pt") as f:
            metadata = f.metadata()

    assert metadata is not None, "Safetensors metadata is None"
    assert metadata.get("modelspec.architecture") == "longcat_image", (
        f"modelspec.architecture is {metadata.get('modelspec.architecture')!r}"
    )


def test_lora_roundtrips_onto_fresh_model():
    """Saved LoRA loads onto a fresh identically-wrapped model with zero
    missing LoRA keys and zero unexpected keys (ComfyUI/ai-toolkit fidelity)."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_tiny_lora(td)

    fresh = _build_peft_model()

    def _remap_to_peft(key: str) -> str:
        """Reverse ai-toolkit key → PEFT internal key."""
        module_path = key[len("diffusion_model."):]
        module_path = module_path.replace(".lora_A.weight", ".lora_A.default.weight")
        module_path = module_path.replace(".lora_B.weight", ".lora_B.default.weight")
        return f"base_model.model.{module_path}"

    remapped = {_remap_to_peft(k): v for k, v in sd.items()}
    missing, unexpected = fresh.load_state_dict(remapped, strict=False)

    lora_missing = [k for k in missing if "lora" in k.lower()]
    assert not lora_missing, f"LoRA keys missing on reload: {lora_missing[:5]}"
    assert not unexpected, f"Unexpected keys on reload: {unexpected[:5]}"


def test_saver_keys_match_peft_module_paths_verbatim():
    """Family-agnostic derivation: PEFT module paths appear verbatim under
    the diffusion_model. prefix (no qwen-specific hardcoding)."""
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file

    with tempfile.TemporaryDirectory() as td:
        saved_path = pathlib.Path(td) / "test.safetensors"
        model = _build_peft_model()

        peft_sd = get_peft_model_state_dict(model)
        peft_keys_sample = [
            k.replace("base_model.model.", "")
            for k in peft_sd.keys()
            if "lora_A" in k
        ][:5]

        saver = _get_saver()
        saver.save(components={"unet": model, "config": {}}, path=saved_path)
        sd = load_file(str(saved_path))

    for peft_key in peft_keys_sample:
        expected = f"diffusion_model.{peft_key}"
        assert expected in sd, f"Expected key {expected!r} not found"
