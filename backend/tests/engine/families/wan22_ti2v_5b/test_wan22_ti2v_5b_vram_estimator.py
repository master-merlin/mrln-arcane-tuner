"""WAN 2.2 TI2V-5B VRAM estimator fallback entry.

The definition ships ``model_size_mb: {}`` (like wan21/wan22), so the
``_FAMILY_PARAMS`` fallback table IS the primary estimation path. Values are
meta-instantiated from the REAL diffusers classes with the recon'd checkpoint
config (transformer 30L/3072-hidden/48ch -> 5.000 B; the NEW high-compression
VAE z_dim=48/base_dim=160/decoder_base_dim=256 -> 0.671 B), so the lower bound
proves the entry (not the generic 2.0 B default) drives the estimate, and the
family must NOT get MoE-doubled (dual_expert is False).
"""

from __future__ import annotations

import math

from app.engine.models.registry import registry
from app.engine.utils.vram_estimator import (
    _FAMILY_PARAMS,
    VRAMEstimator,
    _get_primary_params,
    _get_te_params,
    _get_vae_params,
)


def _defn():
    registry.discover_families()
    registry.load_definitions("app/engine/models/definitions")
    d = registry.get_definition("wan2.2-ti2v-5b")
    assert d is not None
    return d


def test_family_registered_in_fallback_table():
    assert "wan22_ti2v_5b" in _FAMILY_PARAMS
    entry = _FAMILY_PARAMS["wan22_ti2v_5b"]
    assert any("text_encoder" in k for k in entry)
    assert "vae" in entry


def test_fallback_values_pinned():
    assert _get_primary_params("wan22_ti2v_5b", {}) == 5.0
    assert _get_te_params("wan22_ti2v_5b") == 5.7
    assert _get_vae_params("wan22_ti2v_5b") == 0.67


def test_estimate_is_realistic_and_not_moe_doubled():
    defn = _defn()
    d = VRAMEstimator.estimate(defn, {"quantization": "none"}).to_dict()
    # 5.0 B bf16 ~= 9.5 GB. A dual_expert double would push this near 19 GB.
    assert 8_000 < d["model_weights_mb"] < 12_000, d["model_weights_mb"]
    assert math.isfinite(d["peak_mb"]) and d["peak_mb"] > 0
