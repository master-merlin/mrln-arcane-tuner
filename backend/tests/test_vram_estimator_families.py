"""PR8a Task 5: hidream_o1 + ernie_image are first-class in the VRAM estimator.

Before this task both families fell through to weak generic fallbacks in
``_get_primary_params``/``_get_te_params``/``_get_vae_params``. HiDream is a
pixel-space *unified* transformer (no VAE, no external text encoder), so the
generic fallback invented a VAE + TE that do not exist. ERNIE-Image is a ~8B
transformer whose primary component must dwarf SDXL.
"""

from __future__ import annotations

import math

import pytest

from app.engine.models.registry import registry
from app.engine.utils.vram_estimator import _FAMILY_PARAMS, VRAMEstimator


@pytest.fixture(scope="module", autouse=True)
def _loaded_registry():
    registry.discover_families()
    registry.load_definitions("app/engine/models/definitions")
    return registry


def test_estimator_registers_new_families():
    assert "hidream_o1" in _FAMILY_PARAMS
    assert "ernie_image" in _FAMILY_PARAMS


def test_hidream_has_no_vae_or_external_te_contribution():
    defn = registry.get_definition("hidream_o1_image")
    assert defn is not None
    report = VRAMEstimator.estimate(defn, {"quantization": "none"})
    d = report.to_dict()

    # caching_peak = te_mb + vae_mb + overhead_mb. HiDream is pixel-space and
    # unified: both the VAE and the external text encoder must contribute ~0,
    # so the caching peak collapses to ~overhead.
    te_plus_vae_mb = d["caching_peak_mb"] - d["overhead_mb"]
    assert te_plus_vae_mb < 50, f"expected ~0 VAE+TE, got {te_plus_vae_mb} MB"

    # The unified transformer itself still costs real VRAM and is finite.
    assert d["model_weights_mb"] > 0
    assert d["peak_mb"] > 0
    assert math.isfinite(d["peak_mb"])


def test_ernie_estimate_is_reasonable():
    sdxl = registry.get_definition("sdxl_base_1.0")
    ernie = registry.get_definition("ernie-image-base-8b")
    assert ernie is not None

    ernie_report = VRAMEstimator.estimate(ernie, {"quantization": "none"})
    ernie_d = ernie_report.to_dict()

    # ~8B params in bf16 is ~16 GB of primary weights — clearly not a fallback
    # constant (2 GB) and within a sane range.
    assert ernie_d["model_weights_mb"] > 10_000, ernie_d["model_weights_mb"]
    assert ernie_d["model_weights_mb"] < 40_000, ernie_d["model_weights_mb"]
    assert math.isfinite(ernie_d["peak_mb"])

    if sdxl is not None:
        sdxl_d = VRAMEstimator.estimate(sdxl, {"quantization": "none"}).to_dict()
        assert ernie_d["model_weights_mb"] > sdxl_d["model_weights_mb"]
