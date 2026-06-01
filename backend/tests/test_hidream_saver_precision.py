"""HiDream-O1 saver precision resolution (PR8a, Task 6).

Locks the fix for the dead ``definition.save_dtype`` read in
``HiDreamO1Driver.get_saver``. ``ModelDefinition`` has no ``save_dtype``
field, so the old code always fell back to ``"bf16"`` regardless of the
training config. The saver must instead resolve precision from
``config["save_precision"]`` (the established pattern across sdxl/flux2/
qwen_image), keeping ``bf16`` as the fallback for this bf16-native model.

These tests use a minimal LoRA-wrapped module — no real HiDream weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from safetensors import safe_open

from app.engine.core.definitions import ModelDefinition
from app.engine.models.families.hidream_o1.driver import HiDreamO1Driver
from app.engine.models.families.hidream_o1.lora_wrapper import inject_lora_layers
from app.engine.models.families.hidream_o1.saver import (
    HiDreamO1Saver,
    _resolve_dtype,
)


def _make_definition() -> ModelDefinition:
    """Minimal definition with NO ``save_dtype`` attribute (it never existed)."""
    return ModelDefinition(
        id="hidream-o1-test",
        family="hidream_o1",
        name="HiDream-O1 Test",
    )


def _make_lora_module() -> nn.Module:
    class Mini(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = nn.Sequential(nn.Linear(8, 8))

    m = Mini()
    inject_lora_layers(m, rank=4, alpha=8.0)
    return m


def _saved_dtype(
    saver: HiDreamO1Saver, model: nn.Module, tmp_path, config
) -> torch.dtype:
    out_dir = tmp_path / "out"
    saver.save(
        components={"unet": model, "config": config},
        path=out_dir / "lora.safetensors",
    )
    with safe_open(str(out_dir / "lora.safetensors"), framework="pt") as f:
        key = next(k for k in f.keys() if k.endswith(".lora_down.weight"))
        return f.get_tensor(key).dtype


# ── Regression: the definition never had save_dtype ──────────────────────


def test_model_definition_has_no_save_dtype_field():
    """Guards the root cause: reading definition.save_dtype was always dead."""
    definition = _make_definition()
    assert not hasattr(definition, "save_dtype")
    assert "save_dtype" not in ModelDefinition.model_fields


# ── Driver get_saver no longer depends on the missing attribute ──────────


def test_get_saver_default_fallback_is_bf16():
    """With no config, the driver's saver must default to bf16 (bf16-native)."""
    driver = HiDreamO1Driver(_make_definition(), torch.device("cpu"))
    saver = driver.get_saver()
    assert isinstance(saver, HiDreamO1Saver)
    assert saver.save_dtype == torch.bfloat16


# ── Precision is resolved from config["save_precision"] per-call ─────────


def test_saver_resolves_fp32_from_config(tmp_path):
    driver = HiDreamO1Driver(_make_definition(), torch.device("cpu"))
    saver = driver.get_saver()
    model = _make_lora_module()
    dtype = _saved_dtype(saver, model, tmp_path, {"save_precision": "fp32"})
    assert dtype == torch.float32


def test_saver_resolves_fp16_from_config(tmp_path):
    driver = HiDreamO1Driver(_make_definition(), torch.device("cpu"))
    saver = driver.get_saver()
    model = _make_lora_module()
    dtype = _saved_dtype(saver, model, tmp_path, {"save_precision": "fp16"})
    assert dtype == torch.float16


def test_saver_falls_back_to_bf16_when_save_precision_absent(tmp_path):
    """No save_precision key → bf16 (the safe default for a bf16-native model)."""
    driver = HiDreamO1Driver(_make_definition(), torch.device("cpu"))
    saver = driver.get_saver()
    model = _make_lora_module()
    dtype = _saved_dtype(saver, model, tmp_path, {})
    assert dtype == torch.bfloat16


# ── string→dtype helper in isolation ─────────────────────────────────────


def test_resolve_dtype_mapping():
    assert _resolve_dtype("fp16") == torch.float16
    assert _resolve_dtype("bf16") == torch.bfloat16
    assert _resolve_dtype("fp32") == torch.float32
    assert _resolve_dtype("float32") == torch.float32
    assert _resolve_dtype(None) == torch.bfloat16
    # Unknown string → safe bf16 fallback
    assert _resolve_dtype("nonsense") == torch.bfloat16
    # Passthrough when already a torch.dtype
    assert _resolve_dtype(torch.float16) == torch.float16
