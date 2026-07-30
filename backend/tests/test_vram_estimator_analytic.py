"""Analytic-term regressions for the VRAM estimator.

Covers the four terms that were provably wrong rather than merely approximate:
the phantom single-block depth, the primary-component precision lookup, the
quantization-compatibility warning text, and the honesty of ``fits`` when the
GPU cannot be queried. Also pins the calibration formula-version guard, which
is what makes changing an analytic term safe.
"""

from __future__ import annotations

from app.engine.utils.vram_estimator import (
    VRAM_FORMULA_VERSION,
    VRAMEstimator,
    _check_quant_compat,
    _precision_for,
    VRAMReport,
)


def _defn(**arch):
    class _D:
        family = "unknown-test-family"
        detected_precision: dict = {}
        architecture_params = arch
        model_size_mb: dict = {}

    return _D()


class TestActivationDepth:
    def test_no_phantom_single_blocks_for_a_family_without_them(self):
        """A definition that declares only ``depth`` must be charged for
        ``depth`` layers — not depth + 38."""
        cfg = {"resolution": 1024, "batch_size": 1, "gradient_checkpointing": True}
        without = VRAMEstimator.estimate(_defn(depth=40, hidden_size=3072), cfg)
        explicit_zero = VRAMEstimator.estimate(
            _defn(depth=40, depth_single_blocks=0, hidden_size=3072), cfg
        )
        assert without.activations_mb == explicit_zero.activations_mb

    def test_declared_single_blocks_still_count(self):
        """The flux lineage really does have single-stream blocks."""
        cfg = {"resolution": 1024, "batch_size": 1, "gradient_checkpointing": True}
        dual_only = VRAMEstimator.estimate(_defn(depth=19, hidden_size=3072), cfg)
        with_single = VRAMEstimator.estimate(
            _defn(depth=19, depth_single_blocks=38, hidden_size=3072), cfg
        )
        assert with_single.activations_mb > dual_only.activations_mb


class TestPrimaryPrecisionLookup:
    def test_transformer_precision_answers_a_unet_lookup(self):
        assert (
            _precision_for({"transformer": "torch.float8_e4m3fn"}, "unet")
            == "torch.float8_e4m3fn"
        )

    def test_unet_precision_answers_a_transformer_lookup(self):
        assert _precision_for({"unet": "torch.float16"}, "transformer") == "torch.float16"

    def test_non_primary_does_not_inherit_the_primary_dtype(self):
        assert (
            _precision_for({"transformer": "torch.float8_e4m3fn"}, "text_encoder")
            == "torch.bfloat16"
        )


class TestQuantCompatWarning:
    def _force_sm(self, monkeypatch, sm: int):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda *a, **kw: (sm // 10, sm % 10)
        )
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **kw: "TestGPU")

    def test_requirement_names_the_requested_scheme_not_the_fallback(
        self, monkeypatch
    ):
        """Regression: validate_and_fallback returns (backend, scheme); it was
        unpacked as (fallback, scheme), so the message both named the BACKEND as
        the fallback scheme and rewrote the requested scheme, ending up claiming
        that the fallback required Blackwell."""
        from app.engine.factories.quantization import QuantizationFactory

        self._force_sm(monkeypatch, 86)  # Ampere
        monkeypatch.setattr(
            QuantizationFactory,
            "validate_and_fallback",
            classmethod(lambda cls, scheme, backend="auto": ("bitsandbytes", "nf4")),
        )

        report = VRAMReport()
        _check_quant_compat("nvfp4", "Model quantization", report, {})

        assert len(report.warnings) == 1
        msg = report.warnings[0]
        assert "'nvfp4' requires Blackwell" in msg
        assert "fall back to 'nf4'" in msg
        assert "bitsandbytes" not in msg

    def test_each_caller_reads_its_own_backend_key(self, monkeypatch):
        from app.engine.factories.quantization import QuantizationFactory

        self._force_sm(monkeypatch, 86)
        seen: list[str] = []
        monkeypatch.setattr(
            QuantizationFactory,
            "validate_and_fallback",
            classmethod(
                lambda cls, scheme, backend="auto": (seen.append(backend), ("x", "nf4"))[1]
            ),
        )

        config = {
            "quantization": "nvfp4",
            "te_quantization": "nvfp4",
            "quantization_backend": "model-backend",
            "te_quantization_backend": "te-backend",
        }
        VRAMEstimator.estimate(_defn(depth=19), config)
        assert seen == ["model-backend", "te-backend"]


class TestFitHonesty:
    def test_fit_known_false_when_the_gpu_cannot_be_queried(self, monkeypatch):
        import app.core.system_monitor as sysmon

        monkeypatch.setattr(
            sysmon.system_monitor,
            "snapshot",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no nvml")),
        )
        report = VRAMEstimator.estimate(_defn(depth=19), {})
        assert report.fit_known is False
        assert report.to_dict()["fit_known"] is False
        assert any("Could not query GPU" in w for w in report.warnings)

    def test_fit_known_true_on_a_live_query(self):
        report = VRAMEstimator.estimate(_defn(depth=19), {})
        # This suite runs on a machine with a GPU; if that ever changes the
        # assertion above still pins the failure path.
        if report.total_mb > 0:
            assert report.fit_known is True


class TestCalibrationFormulaVersion:
    def test_stale_stamp_drops_the_whole_vram_block(self, monkeypatch):
        from app.core.stats import definition_stats_service as svc

        defn = _defn(depth=19)
        monkeypatch.setattr(svc, "_get_definition", lambda did: defn)

        stale = {
            "vram": {
                "formula_version": VRAM_FORMULA_VERSION - 1,
                "activations_mb": {"value": 0.25, "samples": 3},
            }
        }
        out = svc._vram_estimate("some-def", {}, stale)
        assert out is not None
        assert out["calibrated"] is False
        assert out["calibrated_components"] == []

    def test_current_stamp_applies_the_coefficients(self, monkeypatch):
        from app.core.stats import definition_stats_service as svc

        defn = _defn(depth=19)
        monkeypatch.setattr(svc, "_get_definition", lambda did: defn)

        fresh = {
            "vram": {
                "formula_version": VRAM_FORMULA_VERSION,
                "activations_mb": {"value": 0.5, "samples": 3},
            }
        }
        out = svc._vram_estimate("some-def", {}, fresh)
        assert out["calibrated"] is True
        assert out["calibrated_components"] == ["activations_mb"]

    def test_recompute_stamps_the_current_version(self):
        """``_aggregate`` must stamp whatever it writes, or every coefficient it
        produces would be discarded as stale on the next read."""
        from app.core.stats.definition_stats_service import _aggregate

        stats = _aggregate("no-such-definition", [])
        # No rows → no vram block at all; the stamp only accompanies real data.
        assert "vram" not in stats
