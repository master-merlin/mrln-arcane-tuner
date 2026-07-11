"""Ovis-Image LoRA portability: canonical keys + pinned key count.

Ovis is a Flux-architecture MMDiT. ComfyUI detects and loads it as a
``comfy.model_base.Flux`` model (``comfy/model_detection.py``: the
``double_blocks.*.img_mlp.gate_proj`` + ``txt_norm`` branch, ``# Ovis
model``), so its LoRA key mapping is governed by the Flux handler in
``comfy/lora.py::model_lora_keys_unet`` — ``comfy.utils.flux_to_diffusers``
registers ``key_map`` entries keyed ``transformer.<diffusers_module>`` (and
bare ``<diffusers_module>`` for DiffSynth), NEVER ``diffusion_model.<module>``
with diffusers module names (that prefix in ComfyUI is paired with
BFL-native ``double_blocks.*``/``single_blocks.*`` names via the generic
block, which our diffusers ``transformer_blocks.*`` names do not match).

Therefore OUR saver's canonical portable keys are
``transformer.{module}.lora_A/B.weight`` — the diffusers/PEFT/SimpleTuner
convention that ComfyUI's ``flux_to_diffusers`` ``transformer.`` route maps
onto every one of our 207 curated modules. (Historic note: the saver
previously emitted ``diffusion_model.{module}.*``, which matched NOTHING in
ComfyUI's Flux key_map and silently applied a zero-effect LoRA — the
UAT bug this pins against.)

Pinned key math (checkpoint config: num_layers=6, num_single_layers=27):
- double block:  8 attention + 4 feed-forward projections = 12 modules
- single block:  3 attention + proj_mlp + proj_out         =  5 modules
- total modules: 6*12 + 27*5 = 207 → 207 * 2 (lora_A + lora_B) = **414 keys**
The tiny 1+1-layer model used here pins the per-block counts (12 + 5 →
34 keys) and the full-model expectation is derived from them.
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
)

# Checkpoint block counts (verified transformer/config.json)
_NUM_LAYERS = 6
_NUM_SINGLE_LAYERS = 27

# Per-block LoRA module counts (asserted empirically below)
_DOUBLE_BLOCK_MODULES = 12
_SINGLE_BLOCK_MODULES = 5

# THE pinned count for the real checkpoint: 414 keys.
_EXPECTED_FULL_MODEL_KEYS = (
    _NUM_LAYERS * _DOUBLE_BLOCK_MODULES + _NUM_SINGLE_LAYERS * _SINGLE_BLOCK_MODULES
) * 2  # == 414

# Tiny model (1 double + 1 single): (12 + 5) * 2 = 34 keys.
_EXPECTED_TINY_KEYS = (_DOUBLE_BLOCK_MODULES + _SINGLE_BLOCK_MODULES) * 2


def _make_driver():
    from app.engine.models.families.ovis_image.driver import OvisImageDriver

    definition = MagicMock()
    definition.family = "ovis_image"
    definition.id = "ovis-image-test"
    definition.lora_targetable_modules = []
    definition.architecture_params = {}
    return OvisImageDriver(definition, torch.device("cpu"))


def _build_peft_model(driver):
    """Tiny OvisImageTransformer2DModel wrapped with the driver's LoRA spec."""
    from peft import LoraConfig, get_peft_model
    from diffusers.models.transformers.transformer_ovis_image import (
        OvisImageTransformer2DModel,
    )

    base = OvisImageTransformer2DModel(**_TINY_CFG)
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=4,
        target_modules=driver.get_lora_targets(),
        exclude_modules=driver.get_lora_exclude_modules(),
    )
    return get_peft_model(base, lora_cfg)


def _save_lora(tmp_dir: str):
    from safetensors.torch import load_file

    drv = _make_driver()
    peft_model = _build_peft_model(drv)
    saver = drv.get_saver()
    path = pathlib.Path(tmp_dir) / "ovis_lora.safetensors"
    saver.save(components={"unet": peft_model, "config": {}}, path=path)
    assert path.exists(), "Saver did not produce a safetensors file"
    return path, load_file(str(path))


def test_key_count_pinned_34_tiny_414_full():
    """Tiny 1+1-block model: exactly 34 keys; per-block counts pin the
    full-checkpoint expectation at 414 keys (207 modules × A/B)."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)

    assert len(sd) == _EXPECTED_TINY_KEYS, (
        f"expected {_EXPECTED_TINY_KEYS} keys for the 1+1-block tiny model, "
        f"got {len(sd)}"
    )

    # Per-block module counts measured from the actual saved keys
    double_modules = {
        k.rsplit(".lora_", 1)[0]
        for k in sd
        if ".transformer_blocks." in k or k.startswith(
            "diffusion_model.transformer_blocks."
        )
    }
    single_modules = {
        k.rsplit(".lora_", 1)[0]
        for k in sd
        if "single_transformer_blocks." in k
    }
    # "transformer_blocks." also substring-matches single_transformer_blocks;
    # subtract to get true double-block modules.
    double_modules -= single_modules
    assert len(double_modules) == _DOUBLE_BLOCK_MODULES, (
        f"double block must contribute {_DOUBLE_BLOCK_MODULES} modules, "
        f"got {len(double_modules)}: {sorted(double_modules)}"
    )
    assert len(single_modules) == _SINGLE_BLOCK_MODULES, (
        f"single block must contribute {_SINGLE_BLOCK_MODULES} modules, "
        f"got {len(single_modules)}: {sorted(single_modules)}"
    )

    # The full-model pin: 6*12 + 27*5 = 207 modules → 414 keys.
    assert _EXPECTED_FULL_MODEL_KEYS == 414


def _real_config_model():
    """Meta-instantiate the REAL checkpoint transformer config (no weights)."""
    from diffusers.models.transformers.transformer_ovis_image import (
        OvisImageTransformer2DModel,
    )

    with torch.device("meta"):
        return OvisImageTransformer2DModel(
            patch_size=1,
            in_channels=64,
            out_channels=64,
            num_layers=_NUM_LAYERS,
            num_single_layers=_NUM_SINGLE_LAYERS,
            attention_head_dim=128,
            num_attention_heads=24,
            joint_attention_dim=2048,
            axes_dims_rope=(16, 56, 56),
        )


def test_definition_ships_curated_lora_target_list():
    """ovis-image-base MUST ship the curated 207-module list in its YAML.

    dreamlite precedent (2026-07-08 GPU-UAT crash): a YAML with NO
    ``lora_targetable_modules`` gets the field auto-filled at first real model
    load by ``registry.enrich_definition`` with the introspector's EXHAUSTIVE
    Linear catalog (time embedders, input/output projections, text blocks...),
    and the driver prefers a non-empty definition list over its curated
    pattern defaults — silently breaking the 414-key pinned surface.

    The shipped list is the fully-expanded resolution of the driver's pattern
    targets + top-level ``proj_out`` exclusion against the real config:
    6×12 double + 27×5 single = 207 modules. The top-level ``proj_out`` is
    OMITTED from the list (the driver's exclude regex becomes a no-op on the
    full-path list), so PEFT wraps exactly the tested surface either way.
    """
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._definitions_loaded = False
    ModelRegistry.initialize()

    patterns = _make_driver().get_lora_targets()  # pattern defaults
    real = _real_config_model()
    expected = {
        n
        for n, m in real.named_modules()
        if isinstance(m, torch.nn.Linear)
        and any(n == p or n.endswith("." + p) for p in patterns)
    }
    assert "proj_out" in expected  # sanity: suffix DOES hit the top level...
    expected.discard("proj_out")  # ...and the curated list must omit it
    assert len(expected) == (
        _NUM_LAYERS * _DOUBLE_BLOCK_MODULES
        + _NUM_SINGLE_LAYERS * _SINGLE_BLOCK_MODULES
    )  # == 207

    defn = ModelRegistry._definitions["ovis-image-base"]
    shipped = set(defn.lora_targetable_modules or [])
    assert shipped, "ovis-image-base: YAML must ship the curated LoRA target list"
    assert shipped == expected, (
        f"ovis-image-base: shipped list diverges from the curated/tested surface "
        f"(+{len(shipped - expected)} extra, -{len(expected - shipped)} missing). "
        f"Extras include e.g. {sorted(shipped - expected)[:3]}"
    )


def test_saver_key_format_is_transformer_prefixed():
    """All keys are transformer.{module}.lora_A/B.weight; the model's
    top-level proj_out (final projection) is NOT among them."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)

    assert sd, "Saved state dict is empty"
    for k in sd:
        assert k.startswith("transformer."), f"bad prefix: {k!r}"
        assert not k.startswith("diffusion_model."), (
            f"legacy diffusion_model. prefix is a ComfyUI zero-match: {k!r}"
        )
        assert k.endswith(".weight"), f"bad suffix: {k!r}"
        assert ".lora_A." in k or ".lora_B." in k, f"not a LoRA key: {k!r}"
        assert ".default." not in k, f"PEFT adapter name leaked: {k!r}"

    # Final projection excluded (regex-string exclude_modules)
    assert "transformer.proj_out.lora_A.weight" not in sd, (
        "top-level proj_out must be excluded from the LoRA"
    )
    # But single-block proj_out is present
    assert "transformer.single_transformer_blocks.0.proj_out.lora_A.weight" in sd

    lora_a = [k for k in sd if ".lora_A." in k]
    lora_b = [k for k in sd if ".lora_B." in k]
    assert len(lora_a) == len(lora_b), "lora_A/lora_B counts must match"


def test_saver_keys_match_comfyui_flux_transformer_route():
    """The saved keys must be exactly what ComfyUI's Flux LoRA handler maps.

    ComfyUI loads Ovis as ``comfy.model_base.Flux`` and builds its LoRA
    ``key_map`` from ``comfy.utils.flux_to_diffusers``, registering entries
    keyed ``transformer.<diffusers_module>`` (``comfy/lora.py``
    ``model_lora_keys_unet``: ``key_map["transformer.{}".format(k[:-len(
    ".weight")])] = to``). ``comfy/weight_adapter/lora.py::LoRAAdapter.load``
    then accepts our PEFT ``lora_A``/``lora_B`` suffix via its
    ``diffusers2_lora`` branch (``"{x}.lora_B.weight"``). So for each saved
    module ``transformer.<m>.lora_B.weight``, ComfyUI's adapter key ``x`` is
    ``transformer.<m>`` — which is a live key_map entry -> the LoRA applies.

    This spot-checks the exact diffusers module names ``flux_to_diffusers``
    emits for double blocks (``transformer_blocks.N.attn.to_q``,
    ``...ff.net.0.proj``, ``...ff_context.net.2``) and single blocks
    (``single_transformer_blocks.N.attn.to_q``, ``...proj_mlp``,
    ``...proj_out``) — every module class in our curated target list.
    """
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)

    # Spot-check one representative of every module class flux_to_diffusers
    # maps under the transformer. route (tiny model => block index 0).
    must_have = [
        # double-block attention (img_attn.qkv + txt_attn.qkv slices + proj)
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer.transformer_blocks.0.attn.to_k.lora_B.weight",
        "transformer.transformer_blocks.0.attn.to_v.lora_A.weight",
        "transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight",
        "transformer.transformer_blocks.0.attn.add_q_proj.lora_A.weight",
        "transformer.transformer_blocks.0.attn.add_k_proj.lora_B.weight",
        "transformer.transformer_blocks.0.attn.add_v_proj.lora_A.weight",
        "transformer.transformer_blocks.0.attn.to_add_out.lora_B.weight",
        # double-block feed-forward (img_mlp + txt_mlp)
        "transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight",
        "transformer.transformer_blocks.0.ff.net.2.lora_B.weight",
        "transformer.transformer_blocks.0.ff_context.net.0.proj.lora_A.weight",
        "transformer.transformer_blocks.0.ff_context.net.2.lora_B.weight",
        # single-block (linear1 qkv+mlp slices + linear2)
        "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer.single_transformer_blocks.0.attn.to_k.lora_B.weight",
        "transformer.single_transformer_blocks.0.attn.to_v.lora_A.weight",
        "transformer.single_transformer_blocks.0.proj_mlp.lora_B.weight",
        "transformer.single_transformer_blocks.0.proj_out.lora_A.weight",
    ]
    missing = [k for k in must_have if k not in sd]
    assert not missing, (
        f"keys ComfyUI's flux_to_diffusers transformer. route expects are "
        f"absent from the saved LoRA: {missing}"
    )


def test_saver_architecture_metadata():
    """modelspec.architecture must be 'ovis_image'."""
    from safetensors import safe_open

    with tempfile.TemporaryDirectory() as td:
        path, _ = _save_lora(td)
        with safe_open(str(path), framework="pt") as f:
            metadata = f.metadata()

    assert metadata is not None
    assert metadata.get("modelspec.architecture") == "ovis_image", (
        f"wrong architecture metadata: {metadata.get('modelspec.architecture')!r}"
    )


def test_lora_round_trips_onto_fresh_model():
    """Saved keys load back onto a fresh identically-wrapped model with zero
    missing LoRA keys and zero unexpected keys (ai-toolkit → PEFT remap)."""
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
