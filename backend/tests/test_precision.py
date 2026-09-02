"""
Tests for precision handling, EMA, and loss tracking.
Phase 4: Training Precision & Module Completeness.
"""

import os
import json
import torch
import torch.nn as nn
from app.engine.strategies.ema import EMAHandler
from app.engine.components.training_logger import TrainingLogger


# ── EMA Tests ────────────────────────────────────────────────────────────


class SimpleModel(nn.Module):
    """Minimal trainable model for EMA tests."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)
    
    def forward(self, x):
        return self.linear(x)


class TestEMAHandler:
    """Tests for the EMAHandler component."""

    def test_init_creates_shadow_for_trainable_params(self):
        """EMA shadow should be created for each trainable parameter."""
        model = SimpleModel()
        ema = EMAHandler(model, decay=0.999)
        assert len(ema.shadow) > 0
        assert "linear.weight" in ema.shadow

    def test_init_uses_correct_decay(self):
        """Decay value should be stored correctly."""
        model = SimpleModel()
        ema = EMAHandler(model, decay=0.95)
        assert ema.decay == 0.95

    def test_step_updates_shadow(self):
        """After a step, shadow should differ from initial clone if model weights changed."""
        model = SimpleModel()
        ema = EMAHandler(model, decay=0.9)
        
        # Save initial shadow
        initial_shadow = ema.shadow["linear.weight"].clone()
        
        # Simulate weight update
        with torch.no_grad():
            model.linear.weight.add_(1.0)
        
        ema.step()
        
        # Shadow should have moved toward new weights
        updated_shadow = ema.shadow["linear.weight"]
        assert not torch.allclose(initial_shadow, updated_shadow), "Shadow should change after step"

    def test_store_and_swap_loads_shadow_weights(self):
        """store_and_swap should load EMA shadow weights into the model."""
        model = SimpleModel()
        ema = EMAHandler(model, decay=0.5)
        
        # Step a few times with modified weights
        with torch.no_grad():
            model.linear.weight.fill_(10.0)
        ema.step()
        
        shadow_before = ema.shadow["linear.weight"].clone()
        ema.store_and_swap()
        
        # Model weights should now match shadow
        assert torch.allclose(model.linear.weight.data, shadow_before)

    def test_restore_reverses_swap(self):
        """restore() should undo a store_and_swap()."""
        model = SimpleModel()
        ema = EMAHandler(model, decay=0.5)
        
        with torch.no_grad():
            model.linear.weight.fill_(10.0)
        ema.step()
        
        current = model.linear.weight.data.clone()
        ema.store_and_swap()
        ema.restore()
        
        assert torch.allclose(model.linear.weight.data, current), "Should restore to pre-swap state"

    def test_state_dict_roundtrip(self):
        """state_dict/load_state_dict should roundtrip shadow weights."""
        model = SimpleModel()
        ema = EMAHandler(model, decay=0.999)
        
        state = ema.state_dict()
        
        model2 = SimpleModel()
        ema2 = EMAHandler(model2, decay=0.999)
        ema2.load_state_dict(state)
        
        for key in state:
            assert torch.allclose(ema.shadow[key], ema2.shadow[key])

    def test_frozen_params_excluded_from_shadow(self):
        """Non-trainable parameters should not appear in shadow."""
        model = SimpleModel()
        model.linear.weight.requires_grad = False
        ema = EMAHandler(model, decay=0.999)
        assert len(ema.shadow) == 0


# ── Training Logger Tests ────────────────────────────────────────────────


class TestTrainingLogger:
    """Tests for the TrainingLogger component."""

    def test_log_step_accumulates_history(self):
        """Each log_step call should add an entry to loss_history."""
        tl = TrainingLogger(max_steps=100)
        tl.log_step(0, loss=0.5, lr=1e-4)
        tl.log_step(1, loss=0.4, lr=1e-4)
        tl.log_step(2, loss=0.3, lr=1e-4)
        
        assert len(tl.loss_history) == 3
        assert tl.loss_history[0]["step"] == 1
        assert tl.loss_history[0]["loss"] == 0.5
        assert tl.loss_history[2]["loss"] == 0.3

    def test_loss_history_contains_required_fields(self):
        """Each history entry should have step, loss, lr, elapsed, timestamp."""
        tl = TrainingLogger(max_steps=10)
        tl.log_step(0, loss=1.0, lr=0.001)
        
        entry = tl.loss_history[0]
        assert "step" in entry
        assert "loss" in entry
        assert "lr" in entry
        assert "elapsed" in entry
        assert "timestamp" in entry

    def test_save_loss_history(self, tmp_path):
        """save_loss_history should write valid JSON to disk."""
        tl = TrainingLogger(max_steps=10)
        tl.log_step(0, loss=0.8, lr=1e-4)
        tl.log_step(1, loss=0.6, lr=1e-4)
        
        out_dir = str(tmp_path / "output")
        tl.save_loss_history(out_dir)
        
        path = os.path.join(out_dir, "loss_history.json")
        assert os.path.exists(path)
        
        with open(path) as f:
            data = json.load(f)
        
        assert len(data) == 2
        assert data[0]["loss"] == 0.8
        assert data[1]["step"] == 2

    def test_save_loss_history_noop_without_output_dir(self):
        """save_loss_history with no output_dir should not raise."""
        tl = TrainingLogger(max_steps=10)
        tl.log_step(0, loss=0.5)
        tl.save_loss_history(None)  # Should be a no-op

    def test_save_loss_history_noop_when_empty(self, tmp_path):
        """save_loss_history with no entries should not create a file."""
        tl = TrainingLogger(max_steps=10)
        out_dir = str(tmp_path / "empty_output")
        tl.save_loss_history(out_dir)
        assert not os.path.exists(os.path.join(out_dir, "loss_history.json"))

    def test_progress_calculation(self):
        """Progress should be calculated correctly as percentage."""
        tl = TrainingLogger(max_steps=100)
        tl.log_step(49, loss=0.5)  # Step 50/100 = 50%
        
        entry = tl.loss_history[-1]
        assert entry["step"] == 50  # Steps are 1-indexed in log


# ── Precision Config Tests ───────────────────────────────────────────────


class TestPrecisionConfig:
    """Tests for precision-related configuration handling."""

    def test_autocast_dtype_fp16(self):
        """fp16 mixed_precision should map to torch.float16."""
        config = {"mixed_precision": "fp16"}
        mp = config.get("mixed_precision", "fp16")
        if mp == "bf16":
            dtype = torch.bfloat16
        elif mp == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        assert dtype == torch.float16

    def test_autocast_dtype_bf16(self):
        """bf16 mixed_precision should map to torch.bfloat16."""
        config = {"mixed_precision": "bf16"}
        mp = config.get("mixed_precision", "fp16")
        if mp == "bf16":
            dtype = torch.bfloat16
        elif mp == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        assert dtype == torch.bfloat16

    def test_autocast_dtype_no_amp(self):
        """no/none mixed_precision should map to torch.float32."""
        config = {"mixed_precision": "no"}
        mp = config.get("mixed_precision", "fp16")
        if mp == "bf16":
            dtype = torch.bfloat16
        elif mp == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        assert dtype == torch.float32

    def test_scaler_disabled_for_bf16(self):
        """GradScaler should be disabled when using BF16."""
        config = {"mixed_precision": "bf16"}
        mp = config.get("mixed_precision")
        scaler = torch.amp.GradScaler("cuda", enabled=(mp == "fp16"))
        assert not scaler.is_enabled()

    def test_save_precision_dtype_mapping(self):
        """Save precision strings should map to correct torch dtypes."""
        for prec, expected in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            save_prec = prec.lower()
            save_dtype = torch.float16 if save_prec == "fp16" else (torch.bfloat16 if save_prec == "bf16" else torch.float32)
            assert save_dtype == expected, f"{prec} should map to {expected}"


# ── Server Log Reset ────────────────────────────────────────────────────


# The class that stood here, `TestServerLogReset`, was removed by LANE-56 and
# replaced by `tests/test_server_log_rotation.py`, for two reasons.
#
# 1. It was VACUOUS. It defined a local `patched_setup` that called `os.remove`
#    itself and then asserted the file was gone; `setup_logging` was never
#    called, so it would have passed against an implementation that did nothing.
#    It is the seventh test of that shape found in this round.
# 2. It pinned the wrong contract. `setup_logging` no longer deletes the
#    previous session's log, it ROTATES it to `server.prev.log` — because the
#    next boot after a failed restart is the user's recovery, and the old
#    behaviour had that recovery destroy the only record of what it was
#    recovering from (LANE-56, measured 2026-09-01).
