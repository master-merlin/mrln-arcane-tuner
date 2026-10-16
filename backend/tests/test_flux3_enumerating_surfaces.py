"""LANE-144 task 6: the surfaces that ENUMERATE families know ``flux3``.

Registry block topology (derived from the YAML), the driver's runtime topology,
and the VRAM estimator's fallback table. Each assertion reads the observable
output; the mutant for the registry branch is "the ``flux3`` branch removed".
"""

from __future__ import annotations

import math

import pytest
import torch

from app.engine.models.registry import _derive_block_topology, registry
from app.engine.utils.vram_estimator import (
    _FAMILY_PARAMS,
    VRAMEstimator,
    _get_primary_params,
    _get_te_params,
    _get_vae_params,
)

DEF_ID = "flux3-image-klein-base"


@pytest.fixture(scope="module", autouse=True)
def _loaded_registry():
    registry.discover_families()
    registry.load_definitions("app/engine/models/definitions")
    return registry


def test_registry_derives_flux3_block_topology():
    defn = registry.get_definition(DEF_ID)
    assert defn is not None
    # Topology is derived at enrichment time from the definition's own architecture_params.
    groups = {g["name"]: g for g in _derive_block_topology(defn.family, defn.architecture_params)}
    # The 28 joint single-stream blocks are the swappable stack the vendored model exposes
    # as ``single_transformer_blocks`` (the per-mode ModuleDict is not an attribute path).
    assert groups["single_blocks"]["attr_path"] == "single_transformer_blocks"
    assert groups["single_blocks"]["count"] == 28
    assert groups["single_blocks"]["approx_vram_mb"] > 0


def test_derive_block_topology_flux3_direct():
    topo = _derive_block_topology("flux3", {"transformer.num_single_layers": 28})
    assert [g["count"] for g in topo] == [28]
    assert _derive_block_topology("flux3", {}) == []


def test_driver_topology_matches_the_real_model():
    from app.engine.models.families.flux3.driver import Flux3Driver
    from app.engine.models.families.flux3.vendor.transformer_flux3 import Flux3Transformer2DModel

    model = Flux3Transformer2DModel(
        num_layers=1, num_single_layers=3, attention_head_dim=16, num_attention_heads=2,
        joint_attention_dim=32,
    )
    driver = Flux3Driver(registry.get_definition(DEF_ID), torch.device("cpu"))
    assert driver.get_block_topology() == []  # no model assigned yet
    driver.assign_components({"unet": model, "vae": object()})
    topo = driver.get_block_topology()
    assert [(g["name"], g["attr_path"], g["count"]) for g in topo] == [
        ("single_blocks", "single_transformer_blocks", 3)
    ]
    assert topo[0]["approx_vram_mb"] > 0


def test_flux3_vram_profile_is_not_marked_placeholder():
    # The UAT (LANE-144) settled the profile; a leftover placeholder marker would tell the
    # next reader the numbers are still a guess.
    import inspect

    from app.engine.utils import vram_estimator

    src = inspect.getsource(vram_estimator)
    start = src.index('    "flux3": {')
    block = src[start : src.index("    },", start)]
    assert "placeholder" not in block.lower(), block


def test_vram_estimator_has_a_flux3_profile():
    assert "flux3" in _FAMILY_PARAMS
    entry = _FAMILY_PARAMS["flux3"]
    assert any("text_encoder" in k for k in entry) and "vae" in entry
    # ~5.5 B transformer / ~4.4 B Qwen3-VL-4B TE: not the generic 2.0 B default.
    assert 4.0 < _get_primary_params("flux3", {}) < 8.0
    assert 3.0 < _get_te_params("flux3") < 6.0
    assert _get_vae_params("flux3") > 0.0

    defn = registry.get_definition(DEF_ID)
    d = VRAMEstimator.estimate(defn, {"quantization": "none"}).to_dict()
    assert 8_000 < d["model_weights_mb"] < 40_000, d["model_weights_mb"]
    assert math.isfinite(d["peak_mb"]) and d["peak_mb"] > 0
