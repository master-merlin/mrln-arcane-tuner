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


# ── torchao names, each guarded on its own ───────────────────────────────


def _fresh_copy(module):
    """A new, unregistered copy of ``module``, executed now: what its guarded imports find today."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"{module.__name__}_fresh_copy", module.__file__)
    copy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copy)
    return copy


def test_a_torchao_without_linear_activation_quantized_tensor_still_gives_every_other_name(monkeypatch):
    """torchao 0.18 dropped ``LinearActivationQuantizedTensor``. It shared one guarded import with
    ``Float8Tensor`` and ``Int4TilePackedTo4dTensor``, so all three went None, and with them the
    dense-weight backward of tile-packed int4 and dynamic fp8, without a word."""
    tq = pytest.importorskip("torchao.quantization")
    from app.engine.factories.quantizers import torchao as torchao_quantizer
    from app.engine.factories.quantizers import trainable_backward

    monkeypatch.delattr(tq, "LinearActivationQuantizedTensor", raising=False)  # as on 0.18
    tb = _fresh_copy(trainable_backward)
    for name in ("Float8Tensor", "Int4TilePackedTo4dTensor", "Int8Tensor"):
        assert getattr(tb, name, None) is getattr(tq, name), f"trainable_backward lost {name}"
    ta = _fresh_copy(torchao_quantizer)
    assert ta.torchao is not None, "the torchao quantizer went unavailable"
    for scheme in ("int8", "int8_dynamic"):
        assert ta.TorchAOBackend.weight_type(scheme) is tq.Int8Tensor, scheme


# ── The quantized cache checks the torch and torchao versions ────────────

import contextlib  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

from app.engine.factories import quantization as quantization_module  # noqa: E402

# What save_quantized wrote before it recorded the torchao version, captured from a real save and
# never regenerated: a save run today writes the new format.
LEGACY_METADATA = Path(__file__).parent / "fixtures" / "quantized_cache" / "legacy-2026-11-metadata.json"
A_TORCHAO_VERSION = "0.18.0"  # a real torchao release; the patch makes the test independent of the venv's


def _legacy_metadata() -> dict:
    return json.loads(LEGACY_METADATA.read_text())


def test_the_legacy_fixture_is_the_old_format():
    metadata = _legacy_metadata()
    assert metadata.get("torch_version"), "the legacy fixture lost its torch_version"
    assert "torchao_version" not in metadata, "the legacy fixture was regenerated in the new format"


def _entry_module() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(64, 64)).requires_grad_(False)  # 4160 params: the fixture's count


def _write_entry(path: str, metadata: dict | None) -> None:
    """A cache entry: ``metadata`` as its metadata.json, next to a matching state dict."""
    os.makedirs(path)
    if metadata is None:
        shutil.copyfile(LEGACY_METADATA, os.path.join(path, "metadata.json"))  # the fixture, unchanged
    else:
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f)
    torch.save(_entry_module().state_dict(), os.path.join(path, "model.pt"))


@contextlib.contextmanager
def _captured_logs():
    """Every structlog event of the quantization module; its proxy's cached bind is dropped so it
    binds to the capture's processor list (see test_definition_stats_calibration_fit.py)."""
    from structlog.testing import capture_logs

    quantization_module.logger.__dict__.pop("bind", None)
    try:
        with capture_logs() as logs:
            yield logs
    finally:
        quantization_module.logger.__dict__.pop("bind", None)


def _metadata_case(case: str) -> dict | None:
    legacy = _legacy_metadata()
    if case == "legacy":
        return None  # the fixture file itself
    synthetic = copy.deepcopy(legacy)
    if case == "versions_missing":
        synthetic.pop("torch_version")
    elif case == "torch_changed":
        synthetic.update(torch_version="2.0.0+cpu", torchao_version=A_TORCHAO_VERSION)
    elif case == "torchao_changed":
        synthetic.update(torchao_version="0.10.0")
    return synthetic


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("legacy", "torchao_version_missing"),
        ("versions_missing", "versions_missing"),
        ("torch_changed", "torch_version_changed"),
        ("torchao_changed", "torchao_version_changed"),
    ],
)
def test_an_entry_of_unverified_versions_is_a_miss_before_any_tensor_is_read(case, reason, tmp_path, monkeypatch):
    import safetensors.torch

    path = str(tmp_path / "entry")
    _write_entry(path, _metadata_case(case))
    # The versions this process runs: the fixture's own torch and a real torchao, so in each case
    # only the field the case changes can cause the miss.
    fixture_torch = _legacy_metadata()["torch_version"]
    monkeypatch.setattr(
        QuantizationFactory, "_current_versions",
        staticmethod(lambda: {"torch_version": fixture_torch, "torchao_version": A_TORCHAO_VERSION}),
        raising=False,
    )
    reads: list[str] = []
    real_torch_load, real_load_file = torch.load, safetensors.torch.load_file
    monkeypatch.setattr(torch, "load", lambda *a, **k: reads.append("torch.load") or real_torch_load(*a, **k))
    monkeypatch.setattr(
        safetensors.torch, "load_file", lambda *a, **k: reads.append("load_file") or real_load_file(*a, **k),
    )

    with _captured_logs() as logs:
        loaded = QuantizationFactory.load_quantized(_entry_module(), path, "int8")
    assert loaded is None, f"{case}: the entry was loaded"
    misses = [e for e in logs if e["event"] == "quantized_cache_invalidated"]
    assert [e.get("reason") for e in misses] == [reason], logs
    assert reads == [], f"{case}: a tensor was deserialized before the miss: {reads}"

    # The caller re-quantizes on a miss and saves: the entry then carries both versions and hits.
    monkeypatch.undo()
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed_torchao = version("torchao")
    except PackageNotFoundError:
        installed_torchao = None
    QuantizationFactory.save_quantized(_entry_module(), path, "int8", source_param_count=4160)
    written = json.loads(Path(path, "metadata.json").read_text())
    assert written["torch_version"] == torch.__version__
    assert "torchao_version" in written and written["torchao_version"] == installed_torchao
    assert QuantizationFactory.load_quantized(_entry_module(), path, "int8") is not None


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
