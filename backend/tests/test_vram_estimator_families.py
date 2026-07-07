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
from app.engine.utils.vram_estimator import (
    _FAMILY_PARAMS,
    VRAMEstimator,
    _get_primary_params,
    _get_te_params,
    _get_vae_params,
)


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


# ── Audit P1b (FAM-7): 6 families missing from the fallback table ──────────
#
# ideogram4 / krea2 / wan21 / wan22 ship ``model_size_mb: {}`` and ltx2 ships
# all-zero sizes, so for THOSE definitions the "fallback" table is the PRIMARY
# estimation path today (no definition carries ``total_params`` either).
# Without entries they all fell to the generic 2.0 B default.

_P1B_FAMILIES = (
    "ideogram4",
    "krea2",
    "longcat_image",
    "ltx2",
    "microsoft_lens",
    "ovis_image",
    "prx",
    "wan21",
    "wan22",
)

# family → (definition id, min expected primary-weights MB). All six primaries
# are far above the generic 2.0 B fallback (~3.8 GB bf16), so the lower bound
# also proves the entry (not the default) produced the estimate.
_P1B_DEFINITIONS = {
    "ideogram4": ("ideogram4-fp8", 15_000),  # 9.3 B bf16 ≈ 17.7 GB
    "krea2": ("krea2-raw", 20_000),  # 12.8 B bf16 ≈ 24.4 GB
    "longcat_image": ("longcat-image-base", 20_000),  # 11.9 B bf16 ≈ 22.7 GB
    "ltx2": ("ltx2-3-base", 30_000),  # 18.9 B bf16 ≈ 36.0 GB
    "ovis_image": ("ovis-image-base", 12_000),  # 7.4 B bf16 ≈ 14.1 GB
    "prx": ("prx-sft", 2_000),  # 1.2 B bf16 ≈ 2.3 GB (upper bound pinned below)
    "wan21": ("wan2.1-t2v-14b", 20_000),  # 14.3 B bf16 ≈ 27.3 GB
    "wan22": ("wan2.2-t2v-a14b", 20_000),  # ≥ one 14.3 B expert (MoE may double)
}


def test_estimator_registers_p1b_families():
    for family in _P1B_FAMILIES:
        assert family in _FAMILY_PARAMS, f"{family} missing from _FAMILY_PARAMS"


def test_p1b_family_entries_are_sane_and_schema_consistent():
    for family in _P1B_FAMILIES:
        entry = _FAMILY_PARAMS[family]
        # Same schema as existing entries: a primary + text_encoder + vae key.
        assert _get_primary_params(family, {}) > 1.0, family
        assert any("text_encoder" in k for k in entry), family
        assert "vae" in entry, family
        assert _get_te_params(family) > 0.0, family
        assert _get_vae_params(family) > 0.0, family


def test_p1b_fallback_estimates_are_realistic():
    """The families whose definitions ship empty/zero model_size_mb must get
    realistic primary-weight estimates from the table (fallback path IS the
    live path for them)."""
    for family, (def_id, min_mb) in _P1B_DEFINITIONS.items():
        defn = registry.get_definition(def_id)
        assert defn is not None, f"definition {def_id} not found"
        d = VRAMEstimator.estimate(defn, {"quantization": "none"}).to_dict()
        assert d["model_weights_mb"] > min_mb, (family, d["model_weights_mb"])
        assert d["model_weights_mb"] < 80_000, (family, d["model_weights_mb"])
        assert math.isfinite(d["peak_mb"]) and d["peak_mb"] > 0, family


def test_microsoft_lens_fallback_matches_on_disk_sizes():
    """lens ships real model_size_mb (7600/40000/335) — its table entry is a
    true fallback, calibrated to those on-disk bf16 sizes (size_mb / 2)."""
    assert _get_primary_params("microsoft_lens", {}) == pytest.approx(3.8)
    assert _get_te_params("microsoft_lens") == pytest.approx(20.0)
    assert _get_vae_params("microsoft_lens") == pytest.approx(0.17)


def test_prx_entry_beats_generic_default_from_above():
    """PRX's 1.2 B transformer is SMALLER than the generic 2.0 B fallback, so
    the usual lower bound can't prove the entry is live — pin exact table
    values AND an upper bound below the 2.0 B default's ~3.8 GB estimate."""
    assert _get_primary_params("prx", {}) == pytest.approx(1.2)
    assert _get_te_params("prx") == pytest.approx(2.6)
    assert _get_vae_params("prx") == pytest.approx(0.08)

    defn = registry.get_definition("prx-sft")
    assert defn is not None
    d = VRAMEstimator.estimate(defn, {"quantization": "none"}).to_dict()
    # 1.2 B bf16 ≈ 2.3 GB — a 2.0 B generic fallback would exceed 3.5 GB.
    assert d["model_weights_mb"] < 3_500, d["model_weights_mb"]


def test_prx_pixel_has_te_but_no_vae_contribution():
    """prx_pixel is pixel-space (NO VAE) but keeps an EXTERNAL ~1.7B TE —
    unlike hidream_o1 the caching peak must contain a real TE term while
    the VAE term stays exactly zero (the generic 0.08 fallback would
    invent a VAE that does not exist)."""
    # Table values pinned from meta-instantiation of the checkpoint configs
    # (PRXTransformer2DModel pixel variant 7.0B; Qwen3VLTextModel 1.72B).
    assert _get_primary_params("prx_pixel", {}) == pytest.approx(7.0)
    assert _get_te_params("prx_pixel") == pytest.approx(1.7)
    assert _get_vae_params("prx_pixel") == 0.0

    defn = registry.get_definition("prx-pixel-t2i")
    assert defn is not None
    report = VRAMEstimator.estimate(defn, {"quantization": "none"})
    d = report.to_dict()

    # caching_peak = te_mb + vae_mb + overhead. The TE contributes ~3.3 GB
    # (1.7B bf16); the VAE contributes 0.
    te_plus_vae_mb = d["caching_peak_mb"] - d["overhead_mb"]
    assert 2_000 < te_plus_vae_mb < 5_000, (
        f"expected ~3.3 GB TE and zero VAE, got {te_plus_vae_mb} MB"
    )

    # 7.0 B bf16 ≈ 13.4 GB of primary weights — clearly not the 2.0 B
    # generic fallback (~3.8 GB).
    assert d["model_weights_mb"] > 10_000, d["model_weights_mb"]
    assert d["model_weights_mb"] < 40_000, d["model_weights_mb"]
    assert math.isfinite(d["peak_mb"]) and d["peak_mb"] > 0


def test_wan21_te_is_umt5_xxl_not_generic_default():
    """Without a wan21 entry _get_te_params returned the generic 0.35 —
    UMT5-XXL is ~5.7 B, a 16× underestimate of the caching peak."""
    assert _get_te_params("wan21") == pytest.approx(5.7)
    assert _get_te_params("wan22") == pytest.approx(5.7)
