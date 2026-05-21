"""
Tests for EMAHandler — covers init, step, store_and_swap, restore, state_dict.
"""

import torch
import torch.nn as nn

from app.engine.strategies.ema import EMAHandler


def _make_model():
    """Simple linear model for EMA testing."""
    m = nn.Linear(4, 2, bias=False)
    nn.init.ones_(m.weight)
    return m


class TestEMAInit:
    def test_shadow_clones_trainable_params(self):
        model = _make_model()
        ema = EMAHandler(model, decay=0.99)
        assert len(ema.shadow) == 1  # weight only, no bias
        assert torch.allclose(ema.shadow["weight"], model.weight.data)

    def test_ignores_frozen_params(self):
        model = _make_model()
        model.weight.requires_grad = False
        ema = EMAHandler(model, decay=0.99)
        assert len(ema.shadow) == 0


class TestEMAStep:
    def test_step_updates_shadow(self):
        model = _make_model()
        ema = EMAHandler(model, decay=0.5)
        original_shadow = ema.shadow["weight"].clone()
        # Change model weights
        model.weight.data.fill_(0.0)
        ema.step()
        # shadow = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        assert torch.allclose(ema.shadow["weight"], torch.full_like(model.weight.data, 0.5))

    def test_multiple_steps_converge(self):
        model = _make_model()
        ema = EMAHandler(model, decay=0.9)
        model.weight.data.fill_(0.0)
        for _ in range(100):
            ema.step()
        # After many steps, shadow should converge toward model (0.0)
        assert ema.shadow["weight"].abs().max().item() < 0.01

    def test_step_increments_counter(self):
        model = _make_model()
        ema = EMAHandler(model, decay=0.99)
        ema.step()
        ema.step()
        assert ema._step_count == 2


class TestStoreAndSwap:
    def test_swap_loads_shadow_into_model(self):
        model = _make_model()
        ema = EMAHandler(model, decay=0.5)
        model.weight.data.fill_(0.0)
        ema.step()  # shadow now 0.5

        ema.store_and_swap()
        # Model should now have shadow weights (0.5)
        assert torch.allclose(model.weight.data, ema.shadow["weight"])

    def test_backup_is_created(self):
        model = _make_model()
        ema = EMAHandler(model, decay=0.99)
        ema.store_and_swap()
        assert len(ema.backup) > 0


class TestRestore:
    def test_restore_reverts_model(self):
        model = _make_model()
        ema = EMAHandler(model, decay=0.5)
        original_weights = model.weight.data.clone()
        model.weight.data.fill_(0.0)
        ema.step()

        ema.store_and_swap()  # swap to shadow
        ema.restore()  # revert
        # Model weights should be zero (the state before swap)
        assert torch.allclose(model.weight.data, torch.zeros_like(model.weight.data))

    def test_restore_clears_backup(self):
        model = _make_model()
        ema = EMAHandler(model, decay=0.99)
        ema.store_and_swap()
        ema.restore()
        assert ema.backup == []

    def test_restore_noop_without_backup(self):
        model = _make_model()
        ema = EMAHandler(model, decay=0.99)
        ema.restore()  # Should not raise


class TestStateDictRoundTrip:
    def test_state_dict_returns_shadow(self):
        model = _make_model()
        ema = EMAHandler(model, decay=0.99)
        sd = ema.state_dict()
        assert "weight" in sd

    def test_load_state_dict(self):
        model = _make_model()
        ema = EMAHandler(model, decay=0.99)
        new_shadow = {"weight": torch.zeros(2, 4)}
        ema.load_state_dict(new_shadow)
        assert torch.allclose(ema.shadow["weight"], torch.zeros(2, 4))

    def test_load_state_dict_rebinds_to_param_device(self):
        # Regression: checkpoints are saved with map_location="cpu", so on
        # resume the loaded shadow tensors must be moved to each parameter's
        # current device. Without this, EMA.step() mixes CPU shadow with
        # CUDA params and raises a device-mismatch RuntimeError.
        if not torch.cuda.is_available():
            import pytest
            pytest.skip("CUDA not available")
        model = _make_model().to("cuda")
        ema = EMAHandler(model, decay=0.99)
        cpu_shadow = {"weight": torch.zeros(2, 4, device="cpu")}
        ema.load_state_dict(cpu_shadow)
        assert ema.shadow["weight"].device.type == "cuda"
        # Real-world reproduction: step() must not raise device mismatch
        ema.step()
