"""The caching peak charges what a quantized text encoder keeps unquantized at its loaded precision.

``QuantizationFactory`` never quantizes a module the model declares float32: T5 and UMT5 declare
``wo``, one ``d_ff × d_model`` matrix per encoder layer, about 1.0 B parameters on an XXL encoder.
Their input embedding ``shared`` (``vocab_size × d_model``; ``embed_tokens`` is tied to it) is no
Linear, so no backend quantizes it either: about 1.05 B parameters on UMT5-XXL. Charging the whole
encoder at the quantized bytes per parameter undercounted UMT5-XXL by about 2 GB at int8. The share
is computed from the encoder's config in ``architecture_params``; an encoder of another class is
charged as before.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from app.engine.utils.vram_estimator import (
    _QUANT_BITS,
    VRAM_FORMULA_VERSION,
    VRAMEstimator,
    _get_te_params,
)

_FAMILIES = Path(__file__).resolve().parents[1] / "app/engine/models/families"
MIB = 1024 * 1024
XXL_WO = 24 * 4096 * 10240  # num_layers × d_model × d_ff of T5-XXL and UMT5-XXL
UMT5_XXL_EMBED = 256_384 * 4096  # vocab_size × d_model
T5_XXL_EMBED = 32_128 * 4096

#: UMT5-XXL at int8 through the app's text-encoder path, quantized disk cache warm: the
#: max_memory_allocated of one 4-caption batch's encoding after load and quantize, measured on
#: an RTX PRO 6000 on 2026-11-12 (recorded in t5_peak.txt).
MEASURED_UMT5_XXL_INT8_RESIDENCY_MB = 8248.7


def _definition(rel: str) -> SimpleNamespace:
    data = yaml.safe_load((_FAMILIES / rel).read_text(encoding="utf-8"))
    return SimpleNamespace(
        id=data["id"],
        family=data["family"],
        detected_precision=data.get("detected_precision") or {},
        architecture_params=data.get("architecture_params") or {},
        model_size_mb=data.get("model_size_mb") or {},
    )


def _caching_peak(defn, te_quantization: str) -> float:
    return VRAMEstimator.estimate(defn, {"te_quantization": te_quantization}, check_fit=False).caching_peak_mb


def test_an_int8_umt5_xxl_keeps_wo_and_its_embedding_at_bf16():
    """WAN 2.1's UMT5-XXL: no on-disk size, so the encoder is sized from the family's 5.7 B."""
    defn = _definition("wan21/definitions/wan21_t2v_14b.yaml")
    arch = defn.architecture_params
    assert arch["te.architectures"] == ["UMT5EncoderModel"]
    assert arch["te.num_layers"] * arch["te.d_model"] * arch["te.d_ff"] == XXL_WO
    assert arch["te.vocab_size"] * arch["te.d_model"] == UMT5_XXL_EMBED
    assert defn.detected_precision["text_encoder"] == "torch.bfloat16"

    unquantized = _caching_peak(defn, "none")
    total = _get_te_params(defn.family) * 1e9
    full_mb = total * 2 / MIB
    # The other rows (VAE + overhead) are what the unquantized peak holds beside the encoder.
    rest = unquantized - full_mb
    kept = XXL_WO + UMT5_XXL_EMBED
    want_te = (total - kept) * (_QUANT_BITS["int8"] / 8) / MIB + kept * 2 / MIB
    assert _caching_peak(defn, "int8") == pytest.approx(rest + want_te, abs=1e-6)
    # About 2 GB above the charge before, which priced wo and the embedding at int8 too.
    old = rest + total * (_QUANT_BITS["int8"] / 8) / MIB
    assert _caching_peak(defn, "int8") - old == pytest.approx(kept * 1 / MIB, abs=1e-6)


def test_an_int8_t5_xxl_keeps_wo_and_its_embedding_at_bf16():
    """FLUX.1's T5-XXL (the second encoder, ``te2``) beside CLIP: the on-disk size is the base."""
    defn = _definition("flux1/definitions/dev.yaml")
    arch = defn.architecture_params
    assert arch["te.architectures"] == ["CLIPTextModel"]
    assert arch["te2.architectures"] == ["T5EncoderModel"]
    assert arch["te2.num_layers"] * arch["te2.d_model"] * arch["te2.d_ff"] == XXL_WO
    assert arch["te2.vocab_size"] * arch["te2.d_model"] == T5_XXL_EMBED
    disk_mb = defn.model_size_mb["text_encoder"]

    rest = _caching_peak(defn, "none") - disk_mb
    kept_mb = (XXL_WO + T5_XXL_EMBED) * 2 / MIB
    want_te = (disk_mb - kept_mb) * ((_QUANT_BITS["int8"] / 8) / 2) + kept_mb
    assert _caching_peak(defn, "int8") == pytest.approx(rest + want_te, abs=1e-6)


def test_wan21_t2v_at_int8_covers_the_measured_residency():
    defn = _definition("wan21/definitions/wan21_t2v_14b.yaml")
    assert _caching_peak(defn, "int8") >= MEASURED_UMT5_XXL_INT8_RESIDENCY_MB


def test_an_encoder_class_not_in_the_table_is_charged_as_before():
    defn = _definition("wan21/definitions/wan21_t2v_14b.yaml")
    defn.architecture_params = {**defn.architecture_params, "te.architectures": ["SomeOtherEncoderModel"]}
    total = _get_te_params(defn.family) * 1e9
    rest = _caching_peak(defn, "none") - total * 2 / MIB
    assert _caching_peak(defn, "int8") == pytest.approx(rest + total * (_QUANT_BITS["int8"] / 8) / MIB, abs=1e-6)


def test_an_unquantized_encoder_is_charged_as_before():
    defn = _definition("wan21/definitions/wan21_t2v_14b.yaml")
    total = _get_te_params(defn.family) * 1e9
    other = _definition("wan21/definitions/wan21_t2v_14b.yaml")
    other.architecture_params = {k: v for k, v in other.architecture_params.items() if not k.startswith("te.")}
    for scheme in ("none", "bf16"):
        assert _caching_peak(defn, scheme) == _caching_peak(other, scheme)
    assert _caching_peak(defn, "none") > total * 2 / MIB


# ── The formula version ──────────────────────────────────────────────────


def test_the_formula_version_moved_with_the_kept_share():
    assert VRAM_FORMULA_VERSION == 5


@pytest.mark.parametrize(("stamp", "applied"), [(4, False), (5, True)])
def test_calibration_of_another_formula_version_is_dropped(stamp, applied, monkeypatch):
    import app.core.system_monitor as sysmon
    from app.core.stats import definition_stats_service as svc

    defn = _definition("wan21/definitions/wan21_t2v_14b.yaml")
    monkeypatch.setattr(svc, "_get_definition", lambda did: defn)
    monkeypatch.setattr(sysmon.system_monitor, "snapshot", lambda *a, **kw: None)  # no device query
    stats = {"vram": {"formula_version": stamp, "caching_peak_mb": {"value": 1.5, "samples": 3}}}
    out = svc._vram_estimate(defn.id, {"te_quantization": "int8"}, stats)
    assert out is not None
    assert out["calibrated"] is applied
    assert out["calibrated_components"] == (["caching_peak_mb"] if applied else [])
