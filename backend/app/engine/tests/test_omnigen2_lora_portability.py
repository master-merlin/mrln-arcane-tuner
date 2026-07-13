"""OmniGen2 LoRA portability: upstream-loadable keys + pinned key count.

Key format decision (see ``saver.py`` module docstring for the full,
evidence-cited decision): ``transformer.{module}.lora_A/B.weight`` — the
exact format the upstream ``omnigen2`` package's
``OmniGen2LoraLoaderMixin`` produces (``pack_weights(...,
transformer_name="transformer")``) and consumes
(``load_lora_into_transformer``). Stock ComfyUI's Omnigen2 LoRA branch
registers ONLY bare module-path keys (no prefix) — documented gap, not
papered over.

Pinned key math (tiny 1+1+1+1-block model):
- Every block (all FOUR groups share the same submodule surface): 4
  attention Linears (to_q/to_k/to_v/to_out.0) + 3 feed-forward Linears
  (linear_1/2/3) = 7 modules/block.
- 4 blocks (1 each of layers/noise_refiner/ref_image_refiner/
  context_refiner) -> 28 modules -> 56 keys.
Full checkpoint (32 layers + 2+2+2 refiners = 38 blocks): 38 * 7 = 266
modules -> 532 keys.
"""

from __future__ import annotations

import pathlib
import tempfile
from unittest.mock import MagicMock

import torch


_TINY_CFG = dict(
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
    axes_dim_rope=(4, 2, 2),
    axes_lens=(8, 8, 8),
    text_feat_dim=12,
    timestep_scale=1000.0,
)

_NUM_LAYERS = 32
_NUM_REFINER_LAYERS = 2  # noise_refiner, ref_image_refiner AND context_refiner

_MODULES_PER_BLOCK = 7  # 4 attention + 3 feed_forward

_EXPECTED_FULL_MODEL_KEYS = (
    (_NUM_LAYERS + 3 * _NUM_REFINER_LAYERS) * _MODULES_PER_BLOCK
) * 2  # == 532

_EXPECTED_TINY_KEYS = (4 * _MODULES_PER_BLOCK) * 2  # == 56


def _make_driver():
    from app.engine.models.families.omnigen2.driver import OmniGen2Driver

    definition = MagicMock()
    definition.family = "omnigen2"
    definition.id = "omnigen2-test"
    definition.lora_targetable_modules = []
    definition.architecture_params = {}
    return OmniGen2Driver(definition, torch.device("cpu"))


def _build_peft_model(driver):
    from peft import LoraConfig, get_peft_model

    from app.engine.models.families.omnigen2.vendor.models.transformers.transformer_omnigen2 import (
        OmniGen2Transformer2DModel,
    )

    base = OmniGen2Transformer2DModel(**_TINY_CFG)
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
    path = pathlib.Path(tmp_dir) / "omnigen2_lora.safetensors"
    saver.save(components={"unet": peft_model, "config": {}}, path=path)
    assert path.exists(), "Saver did not produce a safetensors file"
    return path, load_file(str(path))


def test_key_count_pinned_56_tiny_532_full():
    """Tiny 1+1+1+1-block model: exactly 56 keys; full checkpoint (32+2+2+2
    blocks) expectation is 532 keys."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)

    assert len(sd) == _EXPECTED_TINY_KEYS, (
        f"expected {_EXPECTED_TINY_KEYS} keys, got {len(sd)}"
    )
    assert _EXPECTED_FULL_MODEL_KEYS == 532

    assert "transformer.layers.0.attn.to_q.lora_A.weight" in sd
    assert "transformer.ref_image_refiner.0.attn.to_v.lora_A.weight" in sd
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


def test_saver_covers_every_module_class_across_all_four_groups():
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
        "transformer.noise_refiner.0.attn.to_q.lora_A.weight",
        "transformer.noise_refiner.0.feed_forward.linear_2.lora_B.weight",
        "transformer.ref_image_refiner.0.attn.to_out.0.lora_A.weight",
        "transformer.ref_image_refiner.0.feed_forward.linear_3.lora_B.weight",
        "transformer.context_refiner.0.attn.to_k.lora_A.weight",
        "transformer.context_refiner.0.feed_forward.linear_1.lora_B.weight",
    ]
    missing = [k for k in must_have if k not in sd]
    assert not missing, f"missing expected keys: {missing}"


def test_saver_architecture_metadata():
    from safetensors import safe_open

    with tempfile.TemporaryDirectory() as td:
        path, _ = _save_lora(td)
        with safe_open(str(path), framework="pt") as f:
            metadata = f.metadata()

    assert metadata is not None
    assert metadata.get("modelspec.architecture") == "omnigen2"


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


def test_lora_matches_upstream_mixin_prefix_convention():
    """Every key is prefixed with the loadable-module name ``transformer``
    — what the upstream OmniGen2LoraLoaderMixin (``_lora_loadable_modules =
    ["transformer"]``, lora_pipeline.py L59-60 at the pinned REVISION)
    packs on save and strips on load. The upstream target-module recipe
    (train.py L262: to_q/to_k/to_v/to_out.0) is a SUBSET of ours — its
    PEFT-based loader accepts any subset of transformer.* adapters."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)
    for k in sd:
        assert k.split(".", 1)[0] == "transformer"


def test_definition_ships_curated_target_list_matching_driver():
    """omnigen2's YAML ships the SAME curated list driver.get_lora_targets()
    returns (no enrichment-drift risk)."""
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._definitions_loaded = False
    ModelRegistry.initialize()

    drv = _make_driver()
    expected = set(drv.get_lora_targets())

    defn = ModelRegistry._definitions["omnigen2"]
    shipped = set(defn.lora_targetable_modules or [])
    assert shipped, "YAML must ship a non-empty LoRA target list"
    assert shipped == expected, (
        f"shipped list diverges from driver.get_lora_targets(): "
        f"+{shipped - expected} -{expected - shipped}"
    )
