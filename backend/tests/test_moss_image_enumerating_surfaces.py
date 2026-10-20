"""LANE-165 task 6: the surfaces that ENUMERATE families know ``moss_image``.

Registry block topology (derived from the YAML), the driver's runtime topology, and the VRAM
estimator's fallback table. Each assertion reads the observable output; the mutant for the
registry branch is "the ``moss_image`` branch removed from ``_derive_block_topology``".
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

DEF_IDS = ("moss-image-base", "moss-image-sft")


@pytest.fixture(scope="module", autouse=True)
def _loaded_registry():
    registry.discover_families()
    registry.load_definitions("app/engine/models/definitions")
    return registry


@pytest.mark.parametrize("def_id", DEF_IDS)
def test_registry_derives_moss_image_block_topology(def_id):
    defn = registry.get_definition(def_id)
    assert defn is not None
    groups = {g["name"]: g for g in _derive_block_topology(defn.family, defn.architecture_params)}
    # The 42 uniform DiT blocks live in the vendored model's ``layers`` ModuleList.
    assert groups["layers"]["attr_path"] == "layers"
    assert groups["layers"]["count"] == 42
    assert groups["layers"]["approx_vram_mb"] > 0


def test_derive_block_topology_moss_image_direct():
    topo = _derive_block_topology("moss_image", {"transformer.num_layers": 42})
    assert [g["count"] for g in topo] == [42]
    assert _derive_block_topology("moss_image", {}) == []


def test_driver_topology_matches_the_real_model():
    from app.engine.models.families.moss_image.driver import MossImageDriver

    layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(3)])
    model = torch.nn.Module()
    model.layers = layers
    driver = MossImageDriver(registry.get_definition("moss-image-base"), torch.device("cpu"))
    assert driver.get_block_topology() == []  # no model assigned yet
    driver.assign_components({"unet": model, "vae": object()})
    topo = driver.get_block_topology()
    assert [(g["name"], g["attr_path"], g["count"]) for g in topo] == [("layers", "layers", 3)]
    assert topo[0]["approx_vram_mb"] > 0


def test_vram_estimator_has_a_moss_image_profile():
    assert "moss_image" in _FAMILY_PARAMS
    entry = _FAMILY_PARAMS["moss_image"]
    assert any("text_encoder" in k for k in entry) and "vae" in entry
    # ~5.08 B transformer / ~8.39 B Qwen3.5 TE (research §1.2/§1.5): not the generic 2.0 B default.
    assert 4.5 < _get_primary_params("moss_image", {}) < 6.0
    assert 7.0 < _get_te_params("moss_image") < 10.0
    assert _get_vae_params("moss_image") > 0.0

    for def_id in DEF_IDS:
        defn = registry.get_definition(def_id)
        d = VRAMEstimator.estimate(defn, {"quantization": "none"}).to_dict()
        assert 8_000 < d["model_weights_mb"] < 60_000, d["model_weights_mb"]
        assert math.isfinite(d["peak_mb"]) and d["peak_mb"] > 0
