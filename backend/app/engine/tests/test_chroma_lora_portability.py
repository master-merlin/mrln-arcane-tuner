"""Chroma LoRA portability: canonical ComfyUI-mapped keys + pinned key count.

Chroma's blocks reuse FLUX's own ``FluxAttention``/``FluxAttnProcessor``
verbatim (``transformer_chroma.py`` line 33), so its module surface is
byte-identical to flux1's — this saver mirrors ``Flux1Saver`` (``transformer.``
prefix, diffusers module names, NO top-level ``proj_out`` exclusion, unlike
ovis_image's curated/excluded surface).

ComfyUI route (see ``chroma/saver.py`` module docstring for the full,
evidence-cited decision): ComfyUI's ``comfy/model_base.py`` line 2134
declares ``class Chroma(Flux):`` — its Chroma model class SUBCLASSES
``comfy.model_base.Flux``, so the ``isinstance(model, comfy.model_base.
Flux)`` branch in ``comfy/lora.py::model_lora_keys_unet`` fires for Chroma
and runs ``comfy.utils.flux_to_diffusers`` (the Chroma detection block in
``comfy/model_detection.py`` sets the ``hidden_size``/``depth``/
``depth_single_blocks`` keys it reads), registering ``key_map`` entries
keyed ``transformer.<diffusers_module>`` — exactly the format this saver
emits. So the shipped file auto-applies in stock ComfyUI's native Chroma
loader through the SAME Flux route as flux1/ovis_image, AND loads via
``ChromaPipeline.load_lora_weights()`` (diffusers' ``FluxLoraLoaderMixin``).
This module pins the format + the round-trip guarantee.

Pinned key math (tiny 1-double + 1-single-block model, NO exclusion):
- double block:  12 attention + feed-forward modules
- single block:   5 attention + proj_mlp/proj_out modules
- top-level proj_out: 1 module (the model's own final projection — matched
  by the "proj_out" suffix pattern, same collision FLUX.1 itself has and
  does not exclude; see driver.py docstring for the flux1-parity rationale)
- total: 12 + 5 + 1 = 18 modules -> 18 * 2 (lora_A + lora_B) = 36 keys.
Full checkpoint (19 double + 38 single): 19*12 + 38*5 + 1 = 419 modules ->
838 keys.
"""

from __future__ import annotations

import pathlib
import tempfile
from unittest.mock import MagicMock

import torch


_TINY_CFG = dict(
    patch_size=1,
    in_channels=64,
    out_channels=64,
    num_layers=1,
    num_single_layers=1,
    attention_head_dim=8,
    num_attention_heads=2,
    joint_attention_dim=16,
    axes_dims_rope=(2, 4, 2),
    approximator_num_channels=8,
    approximator_hidden_dim=16,
    approximator_layers=1,
)

_NUM_LAYERS = 19
_NUM_SINGLE_LAYERS = 38

_DOUBLE_BLOCK_MODULES = 12
_SINGLE_BLOCK_MODULES = 5
_TOP_LEVEL_PROJ_OUT = 1

_EXPECTED_FULL_MODEL_KEYS = (
    _NUM_LAYERS * _DOUBLE_BLOCK_MODULES
    + _NUM_SINGLE_LAYERS * _SINGLE_BLOCK_MODULES
    + _TOP_LEVEL_PROJ_OUT
) * 2  # == 838

_EXPECTED_TINY_KEYS = (
    _DOUBLE_BLOCK_MODULES + _SINGLE_BLOCK_MODULES + _TOP_LEVEL_PROJ_OUT
) * 2  # == 36


def _make_driver():
    from app.engine.models.families.chroma.driver import ChromaDriver

    definition = MagicMock()
    definition.family = "chroma"
    definition.id = "chroma-test"
    definition.lora_targetable_modules = []
    definition.architecture_params = {}
    return ChromaDriver(definition, torch.device("cpu"))


def _build_peft_model(driver):
    from peft import LoraConfig, get_peft_model
    from diffusers.models.transformers.transformer_chroma import (
        ChromaTransformer2DModel,
    )

    base = ChromaTransformer2DModel(**_TINY_CFG)
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
    path = pathlib.Path(tmp_dir) / "chroma_lora.safetensors"
    saver.save(components={"unet": peft_model, "config": {}}, path=path)
    assert path.exists(), "Saver did not produce a safetensors file"
    return path, load_file(str(path))


def test_key_count_pinned_36_tiny_838_full():
    """Tiny 1+1-block model: exactly 36 keys; the full-checkpoint expectation
    (19+38 blocks) is 838 keys — INCLUDING the top-level proj_out (flux1
    parity, no exclusion)."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)

    assert len(sd) == _EXPECTED_TINY_KEYS, (
        f"expected {_EXPECTED_TINY_KEYS} keys, got {len(sd)}"
    )
    assert _EXPECTED_FULL_MODEL_KEYS == 838

    # Top-level proj_out MUST be present (unlike ovis_image's exclusion).
    assert "transformer.proj_out.lora_A.weight" in sd, (
        "top-level proj_out must be LoRA-targeted (flux1 parity)"
    )
    assert "transformer.single_transformer_blocks.0.proj_out.lora_A.weight" in sd


def test_saver_key_format_is_transformer_prefixed():
    """All keys are transformer.{module}.lora_A/B.weight."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)

    assert sd, "Saved state dict is empty"
    for k in sd:
        assert k.startswith("transformer."), f"bad prefix: {k!r}"
        assert not k.startswith("diffusion_model."), (
            f"diffusion_model. prefix + diffusers module names matches NOTHING "
            f"in ComfyUI's Flux key_map (that prefix is paired only with "
            f"BFL-native double_blocks/single_blocks names — the historical "
            f"flux1/ovis zero-effect-LoRA bug): {k!r}"
        )
        assert k.endswith(".weight"), f"bad suffix: {k!r}"
        assert ".lora_A." in k or ".lora_B." in k, f"not a LoRA key: {k!r}"
        assert ".default." not in k, f"PEFT adapter name leaked: {k!r}"

    lora_a = [k for k in sd if ".lora_A." in k]
    lora_b = [k for k in sd if ".lora_B." in k]
    assert len(lora_a) == len(lora_b), "lora_A/lora_B counts must match"


def test_saver_covers_every_module_class():
    """Spot-check every module class the driver's target list produces."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)

    must_have = [
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer.transformer_blocks.0.attn.to_k.lora_B.weight",
        "transformer.transformer_blocks.0.attn.to_v.lora_A.weight",
        "transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight",
        "transformer.transformer_blocks.0.attn.add_q_proj.lora_A.weight",
        "transformer.transformer_blocks.0.attn.add_k_proj.lora_B.weight",
        "transformer.transformer_blocks.0.attn.add_v_proj.lora_A.weight",
        "transformer.transformer_blocks.0.attn.to_add_out.lora_B.weight",
        "transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight",
        "transformer.transformer_blocks.0.ff.net.2.lora_B.weight",
        "transformer.transformer_blocks.0.ff_context.net.0.proj.lora_A.weight",
        "transformer.transformer_blocks.0.ff_context.net.2.lora_B.weight",
        "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer.single_transformer_blocks.0.attn.to_k.lora_B.weight",
        "transformer.single_transformer_blocks.0.attn.to_v.lora_A.weight",
        "transformer.single_transformer_blocks.0.proj_mlp.lora_B.weight",
        "transformer.single_transformer_blocks.0.proj_out.lora_A.weight",
        "transformer.proj_out.lora_B.weight",
    ]
    missing = [k for k in must_have if k not in sd]
    assert not missing, f"missing expected keys: {missing}"


def test_saver_architecture_metadata():
    """modelspec.architecture must be 'chroma'."""
    from safetensors import safe_open

    with tempfile.TemporaryDirectory() as td:
        path, _ = _save_lora(td)
        with safe_open(str(path), framework="pt") as f:
            metadata = f.metadata()

    assert metadata is not None
    assert metadata.get("modelspec.architecture") == "chroma", (
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


def test_definitions_do_not_ship_pattern_list_as_curated_full_expansion():
    """chroma1-base/chroma1-hd YAMLs ship the SHORT pattern list (like
    flux1's dev.yaml), NOT a fully-expanded per-layer list (like
    ovis_image's base.yaml) — both are valid PEFT target_modules inputs
    (PEFT does suffix matching), and the short form is what the driver's
    own get_lora_targets() returns, so there is no enrichment-drift risk
    (dreamlite 2026-07-08 precedent) for either representation here since
    they're identical."""
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._definitions_loaded = False
    ModelRegistry.initialize()

    drv = _make_driver()
    expected = set(drv.get_lora_targets())

    for def_id in ("chroma1-base", "chroma1-hd"):
        defn = ModelRegistry._definitions[def_id]
        shipped = set(defn.lora_targetable_modules or [])
        assert shipped, f"{def_id}: YAML must ship a non-empty LoRA target list"
        assert shipped == expected, (
            f"{def_id}: shipped list diverges from driver.get_lora_targets(): "
            f"+{shipped - expected} -{expected - shipped}"
        )
