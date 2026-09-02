"""Unit tests for Phase 4 — layer manifest, precision spec, block swapping, targeted training."""

import pytest
import torch
import torch.nn as nn

from app.engine.core.layer_manifest import (
    BlockInfo,
    ModelLayerManifest,
    PrecisionSpec,
)


# ---------------------------------------------------------------------------
# PrecisionSpec
# ---------------------------------------------------------------------------


class TestPrecisionSpec:
    def test_bf16(self):
        spec = PrecisionSpec.from_config("bf16")
        assert spec.autocast_dtype == torch.bfloat16
        assert spec.use_amp is True
        assert spec.grad_scaler_enabled is False

    def test_fp16(self):
        spec = PrecisionSpec.from_config("fp16")
        assert spec.autocast_dtype == torch.float16
        assert spec.use_amp is True
        assert spec.grad_scaler_enabled is True

    def test_fp16_adaptive(self):
        spec = PrecisionSpec.from_config("fp16", is_adaptive_optimizer=True)
        assert spec.grad_scaler_enabled is False

    def test_fp32(self):
        spec = PrecisionSpec.from_config("fp32")
        assert spec.autocast_dtype == torch.float32
        assert spec.use_amp is False
        assert spec.grad_scaler_enabled is False


# ---------------------------------------------------------------------------
# BlockInfo + ModelLayerManifest
# ---------------------------------------------------------------------------


class TestBlockInfo:
    def test_fields(self):
        b = BlockInfo(name="transformer_blocks.3", block_type="joint", param_count=1000, depth_index=3)
        assert b.name == "transformer_blocks.3"
        assert b.block_type == "joint"
        assert b.param_count == 1000
        assert b.depth_index == 3


class TestModelLayerManifest:
    def test_empty(self):
        m = ModelLayerManifest()
        assert m.block_count == 0
        assert m.total_block_params == 0
        assert m.lora_targets == []

    def test_with_blocks(self):
        blocks = [
            BlockInfo("tb.0", "joint", 500, 0),
            BlockInfo("tb.1", "joint", 600, 1),
            BlockInfo("stb.0", "single", 300, 2),
        ]
        m = ModelLayerManifest(
            transformer_blocks=blocks,
            lora_targets=["to_q", "to_k"],
            te_lora_targets=["q_proj"],
        )
        assert m.block_count == 3
        assert m.total_block_params == 1400
        assert len(m.blocks_by_type("joint")) == 2
        assert len(m.blocks_by_type("single")) == 1
        assert m.te_lora_targets == ["q_proj"]


# ---------------------------------------------------------------------------
# BlockSwappingManager
# ---------------------------------------------------------------------------


class TestBlockSwappingManager:
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason=(
            "BlockSwappingManager.apply() stages every block into PINNED host "
            "memory (block_swapping.py _get_or_create_pinned_shadow -> "
            "tensor.pin_memory()), and a CPU-only torch raises 'Cannot access "
            "accelerator device when none is available' (CI run 33687356291). "
            "The product only builds the manager for CUDA training — the sole "
            "caller, pipeline_optimization.py, passes the pipeline's execution "
            "device, and swapping to CPU from CPU saves nothing — so the "
            "test needs the accelerator, not a product fallback."
        ),
    )
    def test_apply_and_remove(self):
        """Hooks are registered and blocks move to CPU on apply."""
        from app.engine.core.optimization.block_swapping import BlockSwappingManager

        block1 = nn.Linear(4, 4)
        block2 = nn.Linear(4, 4)

        mgr = BlockSwappingManager([block1, block2], device=torch.device("cpu"))
        mgr.apply()

        # Hooks should be registered (2 per block: pre + post)
        assert len(mgr._hooks) == 4

        # Blocks should be on CPU after apply
        assert next(block1.parameters()).device.type == "cpu"
        assert next(block2.parameters()).device.type == "cpu"

        mgr.remove()
        assert len(mgr._hooks) == 0


# ---------------------------------------------------------------------------
# TargetedLayerManager
# ---------------------------------------------------------------------------


class TestTargetedLayerManager:
    def test_noop_when_no_patterns(self):
        """No patterns means no changes."""
        from app.engine.core.optimization.targeted_training import TargetedLayerManager

        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        for p in model.parameters():
            p.requires_grad_(True)

        mgr = TargetedLayerManager()
        mgr.apply(model)

        # All should still be trainable
        for p in model.parameters():
            assert p.requires_grad is True

    def test_selective_freeze(self):
        """Only LoRA adapters on matched modules stay trainable."""
        from app.engine.core.optimization.targeted_training import TargetedLayerManager

        # Build a nested module mimicking PEFT structure so named_parameters()
        # produces paths like "base_model.model.blocks.0.attn.to_q.lora_A.weight"

        # Helper leaf module that holds a .weight parameter
        class Leaf(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4))

        # Build one "attn.to_q" submodule with a base weight + lora_A + lora_B
        class LoraLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4))  # base weight
                self.lora_A = Leaf()
                self.lora_B = Leaf()

        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = LoraLinear()

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attn()

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                bm_model = nn.Module()
                bm_model.blocks = nn.ModuleList([Block(), Block()])
                bm = nn.Module()
                bm.model = bm_model
                self.base_model = bm

        model = FakeModel()

        # Enable all
        for p in model.parameters():
            p.requires_grad_(True)

        # Only match block 0
        mgr = TargetedLayerManager([r"blocks\.0\."])
        mgr.apply(model)

        params = dict(model.named_parameters())
        # Base param — untouched (still True)
        assert params["base_model.model.blocks.0.attn.to_q.weight"].requires_grad is True
        # Matched LoRA — trainable
        assert params["base_model.model.blocks.0.attn.to_q.lora_A.weight"].requires_grad is True
        assert params["base_model.model.blocks.0.attn.to_q.lora_B.weight"].requires_grad is True
        # Non-matched LoRA — frozen
        assert params["base_model.model.blocks.1.attn.to_q.lora_A.weight"].requires_grad is False
        assert params["base_model.model.blocks.1.attn.to_q.lora_B.weight"].requires_grad is False


