"""Lumina2 LoRA portability: diffusers-canonical keys + pinned key count.

Key format decision (see ``saver.py`` module docstring for the full,
evidence-cited decision — stock ComfyUI's ``comfy/lora.py`` DOES carry an
``isinstance(model, comfy.model_base.Lumina2)`` branch, but its key_map is
built from ``comfy/utils.py::z_image_to_diffusers``, whose LEFT-hand
candidate keys use the ORIGINAL Alpha-VLLM native checkpoint naming
(``attention.*``, ``feed_forward.w1/w2/w3``, ``attention_norm1/2``) — NOT
diffusers' ``Lumina2Transformer2DModel`` naming (``attn.*``,
``feed_forward.linear_1/2/3``, ``norm1``/``norm2``), verified by live
introspection. None of ComfyUI's four key_map variants match a diffusers-
native Lumina2 LoRA, so this saver uses diffusers-canonical keys
(``transformer.`` + real module names) instead, matching
``Lumina2LoraLoaderMixin`` (``_lora_loadable_modules = ["transformer"]``).

Pinned key math (tiny 1-layer + 1-context_refiner + 1-noise_refiner model):
- Every block (all three groups share the same submodule surface): 4
  attention Linear modules (to_q/to_k/to_v/to_out.0) + 3 feed-forward Linear
  modules (linear_1/linear_2/linear_3) = 7 modules/block.
- 3 blocks (1 each of layers/context_refiner/noise_refiner) -> 21 modules
  -> 21 * 2 (lora_A + lora_B) = 42 keys.
Full checkpoint (26 layers + 2 context_refiner + 2 noise_refiner = 30
blocks): 30 * 7 = 210 modules -> 420 keys.
"""

from __future__ import annotations

import pathlib
import tempfile
from unittest.mock import MagicMock

import torch


_TINY_CFG = dict(
    sample_size=8,
    patch_size=2,
    in_channels=4,
    out_channels=None,
    hidden_size=16,
    num_layers=1,
    num_refiner_layers=1,
    num_attention_heads=2,
    num_kv_heads=1,
    multiple_of=8,
    norm_eps=1e-5,
    axes_dim_rope=(4, 4, 4),
    axes_lens=(8, 8, 8),
    cap_feat_dim=12,
)

_NUM_LAYERS = 26
_NUM_REFINER_LAYERS = 2  # context_refiner AND noise_refiner each have this many

_MODULES_PER_BLOCK = 7  # 4 attention + 3 feed_forward

_EXPECTED_FULL_MODEL_KEYS = (
    (_NUM_LAYERS + 2 * _NUM_REFINER_LAYERS) * _MODULES_PER_BLOCK
) * 2  # == 420

_EXPECTED_TINY_KEYS = (3 * _MODULES_PER_BLOCK) * 2  # == 42


def _make_driver():
    from app.engine.models.families.lumina2.driver import Lumina2Driver

    definition = MagicMock()
    definition.family = "lumina2"
    definition.id = "lumina2-test"
    definition.lora_targetable_modules = []
    definition.architecture_params = {}
    return Lumina2Driver(definition, torch.device("cpu"))


def _build_peft_model(driver):
    from peft import LoraConfig, get_peft_model
    from diffusers.models.transformers.transformer_lumina2 import (
        Lumina2Transformer2DModel,
    )

    base = Lumina2Transformer2DModel(**_TINY_CFG)
    lora_cfg = LoraConfig(
        r=4, lora_alpha=4,
        target_modules=driver.get_lora_targets(),
        exclude_modules=driver.get_lora_exclude_modules(),
    )
    return get_peft_model(base, lora_cfg)


def _save_lora(tmp_dir: str):
    from safetensors.torch import load_file

    drv = _make_driver()
    peft_model = _build_peft_model(drv)
    saver = drv.get_saver()
    path = pathlib.Path(tmp_dir) / "lumina2_lora.safetensors"
    saver.save(components={"unet": peft_model, "config": {}}, path=path)
    assert path.exists(), "Saver did not produce a safetensors file"
    return path, load_file(str(path))


def test_key_count_pinned_42_tiny_420_full():
    """Tiny 1+1+1-block model: exactly 42 keys; full checkpoint (26+2+2
    blocks) expectation is 420 keys."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)

    assert len(sd) == _EXPECTED_TINY_KEYS, (
        f"expected {_EXPECTED_TINY_KEYS} keys, got {len(sd)}"
    )
    assert _EXPECTED_FULL_MODEL_KEYS == 420

    assert "transformer.layers.0.attn.to_q.lora_A.weight" in sd
    assert "transformer.context_refiner.0.feed_forward.linear_1.lora_B.weight" in sd
    assert "transformer.noise_refiner.0.attn.to_out.0.lora_A.weight" in sd


def test_saver_key_format_is_transformer_prefixed():
    """All keys are transformer.{module}.lora_A/B.weight."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)

    assert sd, "Saved state dict is empty"
    for k in sd:
        assert k.startswith("transformer."), f"bad prefix: {k!r}"
        assert not k.startswith("diffusion_model."), f"unexpected prefix: {k!r}"
        assert k.endswith(".weight"), f"bad suffix: {k!r}"
        assert ".lora_A." in k or ".lora_B." in k, f"not a LoRA key: {k!r}"
        assert ".default." not in k, f"PEFT adapter name leaked: {k!r}"

    lora_a = [k for k in sd if ".lora_A." in k]
    lora_b = [k for k in sd if ".lora_B." in k]
    assert len(lora_a) == len(lora_b), "lora_A/lora_B counts must match"


def test_saver_covers_every_module_class_across_all_three_groups():
    """Spot-check every module class across layers/context_refiner/noise_refiner."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)

    must_have = [
        "transformer.layers.0.attn.to_q.lora_A.weight",
        "transformer.layers.0.attn.to_k.lora_B.weight",
        "transformer.layers.0.attn.to_v.lora_A.weight",
        "transformer.layers.0.attn.to_out.0.lora_B.weight",
        "transformer.layers.0.feed_forward.linear_1.lora_A.weight",
        "transformer.layers.0.feed_forward.linear_2.lora_B.weight",
        "transformer.layers.0.feed_forward.linear_3.lora_A.weight",
        "transformer.context_refiner.0.attn.to_q.lora_A.weight",
        "transformer.context_refiner.0.feed_forward.linear_3.lora_B.weight",
        "transformer.noise_refiner.0.attn.to_v.lora_A.weight",
        "transformer.noise_refiner.0.feed_forward.linear_2.lora_B.weight",
    ]
    missing = [k for k in must_have if k not in sd]
    assert not missing, f"missing expected keys: {missing}"

    # NOT the native/ComfyUI-only naming (attention.*, feed_forward.w1-3) —
    # would silently zero-effect if it ever leaked in.
    for k in sd:
        assert ".attention." not in k, f"leaked native-naming key: {k!r}"
        assert ".w1." not in k and ".w2." not in k and ".w3." not in k, (
            f"leaked native feed_forward naming: {k!r}"
        )


def test_saver_architecture_metadata():
    """modelspec.architecture must be 'lumina2'."""
    from safetensors import safe_open

    with tempfile.TemporaryDirectory() as td:
        path, _ = _save_lora(td)
        with safe_open(str(path), framework="pt") as f:
            metadata = f.metadata()

    assert metadata is not None
    assert metadata.get("modelspec.architecture") == "lumina2", (
        f"wrong architecture metadata: {metadata.get('modelspec.architecture')!r}"
    )


def test_lora_round_trips_onto_fresh_model():
    """Saved keys load back onto a fresh identically-wrapped model with zero
    missing LoRA keys and zero unexpected keys."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)

    fresh = _build_peft_model(_make_driver())

    def _remap_to_peft(key: str) -> str:
        module_path = key[len("transformer."):]
        module_path = module_path.replace(".lora_A.weight", ".lora_A.default.weight")
        module_path = module_path.replace(".lora_B.weight", ".lora_B.default.weight")
        return f"base_model.model.{module_path}"

    remapped = {_remap_to_peft(k): v for k, v in sd.items()}
    missing, unexpected = fresh.load_state_dict(remapped, strict=False)

    lora_missing = [k for k in missing if "lora" in k.lower()]
    assert not lora_missing, f"LoRA keys missing on reload: {lora_missing[:5]}"
    assert not unexpected, f"Unexpected keys on reload: {unexpected[:5]}"


def test_lora_loads_via_diffusers_lumina2_pipeline_mixin_convention():
    """The saved key format matches what ``Lumina2LoraLoaderMixin`` expects:
    ``_lora_loadable_modules = ["transformer"]`` (venv/Lib/site-packages/
    diffusers/loaders/lora_pipeline.py line 3856) — i.e. every key must be
    prefixed with the loadable-module name ``transformer``."""
    from diffusers.loaders.lora_pipeline import Lumina2LoraLoaderMixin

    assert Lumina2LoraLoaderMixin._lora_loadable_modules == ["transformer"]

    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)
    for k in sd:
        assert k.split(".", 1)[0] == "transformer"


def test_definition_ships_curated_target_list_matching_driver():
    """lumina-image-2.0's YAML ships the SAME curated list
    driver.get_lora_targets() returns (no enrichment-drift risk)."""
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._definitions_loaded = False
    ModelRegistry.initialize()

    drv = _make_driver()
    expected = set(drv.get_lora_targets())

    defn = ModelRegistry._definitions["lumina-image-2.0"]
    shipped = set(defn.lora_targetable_modules or [])
    assert shipped, "YAML must ship a non-empty LoRA target list"
    assert shipped == expected, (
        f"shipped list diverges from driver.get_lora_targets(): "
        f"+{shipped - expected} -{expected - shipped}"
    )
