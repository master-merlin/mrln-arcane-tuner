"""
Tests for Phase 8: Quantization Support.

Covers:
- QuantizationFactory API (availability, estimate, scheme listing)
- Strategy backend quantization (TorchAO, Quanto, BitsAndBytes) via mocking
- Error handling for unknown/unavailable schemes
- Config validation
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

from app.engine.factories.quantization import QuantizationFactory


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_model(in_features: int = 16, out_features: int = 8) -> nn.Module:
    """Create a small model for quantization testing."""
    model = nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(),
        nn.Linear(out_features, out_features),
    )
    model.requires_grad_(False)
    return model


# ── Factory API Tests ────────────────────────────────────────────────────


class TestQuantizationFactory:

    def test_get_backend_success(self):
        """Should return correct backend class."""
        backend = QuantizationFactory.get_backend("torchao")
        from app.engine.factories.quantizers.torchao import TorchAOBackend
        assert backend is TorchAOBackend

    def test_get_backend_failure(self):
        """Should raise ValueError for unknown backend."""
        with pytest.raises(ValueError, match="Unknown quantization backend"):
             QuantizationFactory.get_backend("fake_backend")

    def test_resolve_auto_none(self):
         """'none' should bypass auto resolution."""
         backend, scheme = QuantizationFactory._resolve_backend_and_scheme("auto", "none")
         assert backend == "auto"
         assert scheme == "none"

    @patch("app.engine.factories.quantizers.torchao.TorchAOBackend.is_available", return_value=True)
    def test_resolve_auto_torchao(self, mock_avail):
         """Auto should resolve 'int8' to 'torchao' if available."""
         backend, scheme = QuantizationFactory._resolve_backend_and_scheme("auto", "int8")
         assert backend == "torchao"
         assert scheme == "int8"


# ── VRAM Estimation ──────────────────────────────────────────────────────


class TestVRAMEstimation:

    def test_estimate_returns_dict(self):
        """Should return dict with before/after/savings keys."""
        model = _make_model()
        est = QuantizationFactory.estimate_vram(model, "int8", "torchao")
        assert "before_mb" in est
        assert "after_mb" in est
        assert "savings_pct" in est

    def test_estimate_none_zero_savings(self):
        """None/bf16 should show no change for bf16 model."""
        model = _make_model()
        model = model.to(torch.bfloat16)
        est = QuantizationFactory.estimate_vram(model, "bf16")
        assert est["savings_pct"] == 0

    @patch("app.engine.factories.quantization.QuantizationFactory._resolve_backend_and_scheme", return_value=("torchao", "int4"))
    def test_estimate_int4_75pct_savings(self, mock_resolve):
        """INT4 on a float32 model should show ~75% savings."""
        model = _make_model()  # default float32
        est = QuantizationFactory.estimate_vram(model, "int4")

        # 4.5 bits from 32 bits = ~85.9% savings
        assert est["savings_pct"] > 80.0

    @patch("app.engine.factories.quantization.QuantizationFactory._resolve_backend_and_scheme", return_value=("torchao", "fp8"))
    def test_estimate_fp8_50pct_for_fp16(self, mock_resolve):
        """FP8 on a float16 model should show ~46.9% savings."""
        model = _make_model().half()
        est = QuantizationFactory.estimate_vram(model, "fp8")
        # 8.5 bits from 16 bits = 46.9% savings
        assert est["savings_pct"] == 46.9

    def test_estimate_empty_model(self):
        """Empty model should return zeros."""
        model = nn.Module()
        est = QuantizationFactory.estimate_vram(model, "int8")

        assert est["before_mb"] == 0
        assert est["after_mb"] == 0


# ── Quantize API ─────────────────────────────────────────────────────────


class TestQuantizeAPI:

    def test_quantize_none_noop(self):
        """'none' should return the exact same module."""
        model = _make_model()
        result = QuantizationFactory.quantize(model, "none")
        assert result is model

    def test_quantize_bf16_noop(self):
        """'bf16' should return the exact same module."""
        model = _make_model()
        result = QuantizationFactory.quantize(model, "bf16")
        assert result is model

    def test_quantize_unknown_backend_raises(self):
        """Unknown backend should propagate ValueError."""
        model = _make_model()
        with pytest.raises(ValueError, match="Unknown quantization backend"):
            QuantizationFactory.quantize(model, "int8", "fake")

    # The factory picks its default device from `torch.cuda.is_available()`
    # (quantization.py, `QuantizationFactory.quantize`). That is pinned here
    # rather than read off the machine: the developer box has a GPU and the CI
    # runner does not, and the un-pinned `device="cuda"` expectation failed
    # there with `device="cpu"` (gate.yml run 33687356291). The dispatch tests
    # assert the device the test controls; the two below prove the default is
    # the CUDA probe and nothing else.
    _CUDA = "app.engine.factories.quantization.torch.cuda.is_available"

    @patch(_CUDA, return_value=True)
    @patch("app.engine.factories.quantizers.torchao.TorchAOBackend.quantize")
    @patch("app.engine.factories.quantizers.torchao.TorchAOBackend.is_available", return_value=True)
    def test_quantize_torchao_dispatches(self, _avail, mock_quant, _cuda):
        """FP8 scheme should call torchao backend."""
        model = _make_model()
        mock_quant.return_value = model
        QuantizationFactory.quantize(model, "fp8", "torchao")
        mock_quant.assert_called_once_with(model, "fp8", device="cuda")

    @patch(_CUDA, return_value=True)
    @patch("app.engine.factories.quantizers.bitsandbytes.BitsAndBytesBackend.quantize")
    @patch("app.engine.factories.quantizers.bitsandbytes.BitsAndBytesBackend.is_available", return_value=True)
    def test_quantize_nf4_dispatches(self, _avail, mock_nf4, _cuda):
        """NF4 scheme should call bitsandbytes backend."""
        model = _make_model()
        mock_nf4.return_value = model
        QuantizationFactory.quantize(model, "nf4", "bitsandbytes")
        mock_nf4.assert_called_once_with(model, "nf4", device="cuda")

    @patch(_CUDA, return_value=False)
    @patch("app.engine.factories.quantizers.torchao.TorchAOBackend.quantize")
    @patch("app.engine.factories.quantizers.torchao.TorchAOBackend.is_available", return_value=True)
    def test_default_device_is_cpu_without_cuda(self, _avail, mock_quant, _cuda):
        """Negative control: no CUDA -> the backend is handed ``device="cpu"``.

        This is the exact shape the runner produced; asserting it proves the
        two dispatch tests above pass because the probe is pinned, not because
        this machine happens to have a GPU.
        """
        model = _make_model()
        mock_quant.return_value = model
        QuantizationFactory.quantize(model, "fp8", "torchao")
        mock_quant.assert_called_once_with(model, "fp8", device="cpu")

    @patch(_CUDA, return_value=False)
    @patch("app.engine.factories.quantizers.torchao.TorchAOBackend.quantize")
    @patch("app.engine.factories.quantizers.torchao.TorchAOBackend.is_available", return_value=True)
    def test_an_explicit_device_outranks_the_probe(self, _avail, mock_quant, _cuda):
        """A caller-supplied device is forwarded verbatim; the probe is only the default."""
        model = _make_model()
        mock_quant.return_value = model
        QuantizationFactory.quantize(model, "fp8", "torchao", device="cuda:1")
        mock_quant.assert_called_once_with(model, "fp8", device="cuda:1")
        _cuda.assert_not_called()


# ── Config Schema ────────────────────────────────────────────────────────


class TestConfigSchema:
    """Tests for quantization field in BaseTrainingConfig."""

    def test_default_is_none(self):
        """Default quantization should be 'none'."""
        from app.engine.models.base import BaseTrainingConfig

        config = BaseTrainingConfig(
            lora_name="test",
            datasets=[{"dataset_name": "d1"}],
        )
        assert config.quantization == "none"

    def test_accepts_all_schemes(self):
        """All valid schemes should be accepted."""
        from app.engine.models.base import BaseTrainingConfig

        for scheme in ("none", "fp8", "nf4", "int4", "int5", "int6", "int7", "int8"):
            config = BaseTrainingConfig(
                lora_name="test",
                datasets=[{"dataset_name": "d1"}],
                quantization=scheme,
            )
            assert config.quantization == scheme

    def test_rejects_invalid_scheme(self):
        """Invalid scheme should fail validation."""
        from pydantic import ValidationError
        from app.engine.models.base import BaseTrainingConfig

        with pytest.raises(ValidationError):
            BaseTrainingConfig(
                lora_name="test",
                datasets=[{"dataset_name": "d1"}],
                quantization="banana",
            )
