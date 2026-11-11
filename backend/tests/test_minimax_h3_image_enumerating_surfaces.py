"""The surfaces that ENUMERATE families know ``minimax_h3_image``.

Registry block topology (shipped in the YAML, and derived by ``_derive_block_topology`` for a
definition that ships none), the curated LoRA targets against the native key map, and the VRAM
estimator's fallback table against the on-disk sizes of ``MiniMaxAI/MiniMax-H3-Image`` at
``fbde899f``. The mutant for the registry branch is "the ``minimax_h3_image`` branch removed from
``_derive_block_topology``".
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

DEF_ID = "minimax-h3-image-t2i"
FAMILY = "minimax_h3_image"
MIB = 1024 * 1024

# On-disk bytes at fbde899f: the two safetensors index files' metadata.total_size, and the VAE's
# single safetensors file (no index), from the repo's file listing.
TRANSFORMER_BYTES = 66_289_356_032  # transformer_diffusers/diffusion_pytorch_model.safetensors.index.json
TEXT_ENCODER_BYTES = 66_714_780_128  # text_encoder/model.safetensors.index.json
VAE_BYTES = 2_336_973_400  # vae_diffusers/diffusion_pytorch_model.safetensors
# Parameter counts of the vendored classes built on the meta device from the checkpoint configs.
TRANSFORMER_PARAMS = 33_125_224_384
VAE_PARAMS = 584_228_408  # float32 on disk: 4.0001 bytes per parameter


@pytest.fixture(scope="module", autouse=True)
def _loaded_registry():
    registry.discover_families()
    registry.load_definitions("app/engine/models/definitions")
    return registry


def _definition():
    defn = registry.get_definition(DEF_ID)
    assert defn is not None, f"{DEF_ID} is not in the registry"
    return defn


def test_the_family_and_its_one_definition_are_registered():
    assert FAMILY in registry._families
    assert [d.id for d in registry._definitions.values() if d.family == FAMILY] == [DEF_ID]
    assert registry.is_definition_available(DEF_ID)


def test_the_shipped_topology_is_the_vendored_models_two_stacks():
    groups = {g["attr_path"]: g for g in _definition().block_topology}
    assert set(groups) == {"transformer_blocks", "token_refiner.refiner_blocks"}
    assert groups["transformer_blocks"]["count"] == 50
    assert groups["token_refiner.refiner_blocks"]["count"] == 2
    # bf16 bytes of one block of each stack (645,571,840 / 385,362,688 parameters).
    assert groups["transformer_blocks"]["approx_vram_mb"] == 1231
    assert groups["token_refiner.refiner_blocks"]["approx_vram_mb"] == 735


def test_the_registry_derives_the_same_topology_when_the_yaml_ships_none():
    shipped = [(g["name"], g["attr_path"], g["count"]) for g in _definition().block_topology]
    derived = _derive_block_topology(FAMILY, _definition().architecture_params)
    assert [(g["name"], g["attr_path"], g["count"]) for g in derived] == shipped
    assert all(g["approx_vram_mb"] > 0 for g in derived)


def test_derive_block_topology_minimax_h3_image_direct():
    topo = _derive_block_topology(
        FAMILY, {"transformer.num_layers": 50, "transformer.token_refiner_num_layers": 2}
    )
    assert [(g["attr_path"], g["count"]) for g in topo] == [
        ("transformer_blocks", 50),
        ("token_refiner.refiner_blocks", 2),
    ]
    assert _derive_block_topology(FAMILY, {}) == []


def test_every_shipped_target_reaches_a_native_module_in_both_stacks():
    """The curated list names attention q/k/v/out and the FFN; each one is a module of every main
    and refiner block, and the LoRA file's key map covers it (no adaln, no heads)."""
    from app.engine.models.families.minimax_h3_image.lora_keys import native_module_for

    targets = _definition().lora_targetable_modules
    assert targets == ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2"]
    for stack in ("transformer_blocks", "token_refiner.refiner_blocks"):
        for target in targets:
            native, _ = native_module_for(f"{stack}.0.{target}")
            assert native.endswith((".attn.qkv_proj", ".attn.out_proj", ".mlp.fc1", ".mlp.fc2")), native


def test_the_definition_ships_the_on_disk_sizes():
    sizes = _definition().model_size_mb
    assert sizes["transformer"] == pytest.approx(TRANSFORMER_BYTES / MIB, abs=0.01)
    assert sizes["text_encoder"] == pytest.approx(TEXT_ENCODER_BYTES / MIB, abs=0.01)
    assert sizes["vae"] == pytest.approx(VAE_BYTES / MIB, abs=0.01)


def test_the_estimator_fallback_agrees_with_the_checkpoint():
    assert FAMILY in _FAMILY_PARAMS
    assert _get_primary_params(FAMILY, {}) == pytest.approx(TRANSFORMER_PARAMS / 1e9, abs=0.005)
    # The same Qwen3-VL as minimax_h3: bf16, bytes / 2.
    assert _get_te_params(FAMILY) == pytest.approx(TEXT_ENCODER_BYTES / 2 / 1e9, abs=0.005)
    assert _get_vae_params(FAMILY) == pytest.approx(VAE_PARAMS / 1e9, abs=0.005)
    # The video family's fallback table is untouched.
    assert _FAMILY_PARAMS["minimax_h3"] == {"transformer": 33.12, "text_encoder": 33.36, "vae": 5.21}


def test_the_estimate_is_the_right_order_of_magnitude():
    d = VRAMEstimator.estimate(_definition(), {"quantization": "none"}).to_dict()
    # ~61.7 GiB of bf16 transformer on disk; the generic 2.0 B fallback read ~3.8 GB.
    assert 50_000 < d["model_weights_mb"] < 90_000, d["model_weights_mb"]
    # caching = text encoder + VAE + overhead: the ~62 GiB encoder dominates.
    assert d["caching_peak_mb"] > 50_000, d["caching_peak_mb"]
    assert math.isfinite(d["peak_mb"]) and d["peak_mb"] > 0


def test_the_driver_reports_the_shipped_topology():
    from app.engine.models.families.minimax_h3_image.driver import MiniMaxH3ImageDriver

    driver = MiniMaxH3ImageDriver(_definition(), torch.device("cpu"))
    assert driver.get_block_topology() == _definition().block_topology
    assert driver.get_lora_targets() == _definition().lora_targetable_modules
