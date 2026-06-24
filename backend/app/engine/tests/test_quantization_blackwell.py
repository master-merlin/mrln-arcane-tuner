"""Tests for Blackwell GPU detection used by the FP8 quantization cache path."""

from unittest.mock import patch

from app.engine.factories.quantization import _is_blackwell


def test_is_blackwell_true_for_workstation_sm120():
    """RTX PRO 6000 Blackwell / RTX 50-series = sm_120 = (12, 0)."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_capability", return_value=(12, 0)):
        assert _is_blackwell() is True


def test_is_blackwell_true_for_datacenter_sm100():
    """B100/B200/GB200 = sm_100 = (10, 0)."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_capability", return_value=(10, 0)):
        assert _is_blackwell() is True


def test_is_blackwell_false_for_hopper_sm90():
    """Hopper (H100) = sm_90 = (9, 0) — NOT Blackwell."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_capability", return_value=(9, 0)):
        assert _is_blackwell() is False


def test_is_blackwell_false_for_ada_sm89():
    """Ada (RTX 4090) = sm_89 = (8, 9) — NOT Blackwell."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_capability", return_value=(8, 9)):
        assert _is_blackwell() is False


def test_is_blackwell_false_when_no_cuda():
    with patch("torch.cuda.is_available", return_value=False):
        assert _is_blackwell() is False


def test_is_blackwell_false_on_capability_error():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_capability", side_effect=RuntimeError("boom")):
        assert _is_blackwell() is False
