"""Nucleus-Image LoRA portability: diffusers-canonical keys + pinned key
count + router-gate absence.

Key format decision (see ``saver.py`` module docstring for the full,
evidence-cited decision): ``NucleusMoEImagePipeline`` has NO LoRA loader
mixin at all (verified: no ``NucleusMoEImageLoraLoaderMixin`` anywhere in
``diffusers/loaders/lora_pipeline.py`` 0.39.0), but
``NucleusMoEImageTransformer2DModel`` inherits ``PeftAdapterMixin`` whose
``load_lora_adapter(..., prefix="transformer")`` default strips exactly the
``transformer.`` key prefix this saver emits — so the file is directly
loadable via ``transformer.load_lora_adapter(path)`` today, with zero
saver-side changes needed if a pipeline-level mixin is added later. ComfyUI
has ZERO Nucleus support (verified against live ``comfy/lora.py``/
``comfy/supported_models.py`` — no "nucleus" string anywhere), a strictly
larger gap than lumina2's non-matching stub branch.

Pinned key math (tiny 4-block model: 3 dense + 1 MoE, ``dense_moe_strategy=
"leave_first_three_blocks_dense"``):
- Every block: 6 attention Linear modules (to_q/to_k/to_v/to_out.0/
  add_k_proj/add_v_proj) — matches ALL 4 blocks.
- 3 dense blocks: 2 FFN Linear modules each (img_mlp.net.0.proj/net.2).
- 1 MoE block: 2 FFN Linear modules (shared_expert.net.0.proj/net.2) — the
  64 routed experts (raw nn.Parameter, not nn.Linear) and the router gate
  are BOTH excluded from LoRA (controller-pinned scope).
- Total modules: 4*6 + 3*2 + 1*2 = 24 + 6 + 2 = 32 -> 32*2 (lora_A+lora_B)
  = 64 keys.
"""

from __future__ import annotations

import pathlib
import tempfile
from unittest.mock import MagicMock

import torch


_TINY_CFG = dict(
    patch_size=2,
    in_channels=16,
    out_channels=4,
    num_layers=4,
    attention_head_dim=8,
    num_attention_heads=2,
    num_key_value_heads=1,
    joint_attention_dim=12,
    axes_dims_rope=(2, 2, 4),
    mlp_ratio=4.0,
    moe_enabled=True,
    dense_moe_strategy="leave_first_three_blocks_dense",
    num_experts=4,
    moe_intermediate_dim=8,
    capacity_factors=[0.0, 0.0, 0.0, 2.0],
    use_sigmoid=False,
    route_scale=1.0,
    use_grouped_mm=False,
)

_NUM_LAYERS = 4
_NUM_DENSE = 3
_NUM_MOE = 1
_ATTN_MODULES_PER_BLOCK = 6  # to_q/to_k/to_v/to_out.0/add_k_proj/add_v_proj
_FFN_MODULES_PER_BLOCK = 2  # net.0.proj + net.2 (dense img_mlp OR MoE shared_expert)

_EXPECTED_TINY_MODULES = (
    _NUM_LAYERS * _ATTN_MODULES_PER_BLOCK
    + _NUM_DENSE * _FFN_MODULES_PER_BLOCK
    + _NUM_MOE * _FFN_MODULES_PER_BLOCK
)  # == 32
_EXPECTED_TINY_KEYS = _EXPECTED_TINY_MODULES * 2  # == 64


def _make_driver():
    from app.engine.models.families.nucleus_image.driver import NucleusImageDriver

    definition = MagicMock()
    definition.family = "nucleus_image"
    definition.id = "nucleus-image-test"
    definition.lora_targetable_modules = []
    definition.architecture_params = {}
    return NucleusImageDriver(definition, torch.device("cpu"))


def _build_peft_model(driver):
    from peft import LoraConfig, get_peft_model
    from diffusers.models.transformers.transformer_nucleusmoe_image import (
        NucleusMoEImageTransformer2DModel,
    )

    torch.manual_seed(0)
    base = NucleusMoEImageTransformer2DModel(**_TINY_CFG)
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
    path = pathlib.Path(tmp_dir) / "nucleus_image_lora.safetensors"
    saver.save(components={"unet": peft_model, "config": {}}, path=path)
    assert path.exists(), "Saver did not produce a safetensors file"
    return path, load_file(str(path))


def test_key_count_pinned_64_tiny_model():
    """Tiny 4-block model (3 dense + 1 MoE): exactly 64 keys."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)

    assert len(sd) == _EXPECTED_TINY_KEYS, (
        f"expected {_EXPECTED_TINY_KEYS} keys, got {len(sd)}: {sorted(sd)}"
    )

    assert "transformer.transformer_blocks.0.attn.to_q.lora_A.weight" in sd
    assert "transformer.transformer_blocks.0.img_mlp.net.0.proj.lora_B.weight" in sd
    assert "transformer.transformer_blocks.3.img_mlp.shared_expert.net.2.lora_A.weight" in sd


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


def test_saver_no_router_gate_no_routed_experts_leak():
    """PINS task-brief decision #1: no saved key ever touches the MoE
    router gate or the routed experts (structurally impossible either way,
    but assert it explicitly so a future refactor can't silently regress
    it)."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)

    for k in sd:
        assert ".img_mlp.gate." not in k, f"leaked router gate key: {k!r}"
        assert not k.endswith(".gate.lora_A.weight"), f"leaked router gate key: {k!r}"
        assert not k.endswith(".gate.lora_B.weight"), f"leaked router gate key: {k!r}"
        assert "experts.gate_up_proj" not in k, f"leaked routed-expert key: {k!r}"
        assert "experts.down_proj" not in k, f"leaked routed-expert key: {k!r}"


def test_saver_covers_every_module_class_dense_and_moe():
    """Spot-check every module class across dense (idx 0-2) and MoE (idx 3)
    blocks."""
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)

    must_have = [
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer.transformer_blocks.0.attn.to_k.lora_B.weight",
        "transformer.transformer_blocks.0.attn.to_v.lora_A.weight",
        "transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight",
        "transformer.transformer_blocks.0.attn.add_k_proj.lora_A.weight",
        "transformer.transformer_blocks.0.attn.add_v_proj.lora_B.weight",
        "transformer.transformer_blocks.2.img_mlp.net.0.proj.lora_A.weight",
        "transformer.transformer_blocks.2.img_mlp.net.2.lora_B.weight",
        "transformer.transformer_blocks.3.attn.to_q.lora_A.weight",
        "transformer.transformer_blocks.3.img_mlp.shared_expert.net.0.proj.lora_A.weight",
        "transformer.transformer_blocks.3.img_mlp.shared_expert.net.2.lora_B.weight",
    ]
    missing = [k for k in must_have if k not in sd]
    assert not missing, f"missing expected keys: {missing}"

    # Block 3 (MoE) must NOT have a bare img_mlp.net.* key (that pattern
    # only exists on dense blocks whose img_mlp IS a FeedForward directly).
    for k in sd:
        if "transformer_blocks.3.img_mlp.net." in k:
            raise AssertionError(f"MoE block leaked a dense-only key shape: {k!r}")


def test_saver_architecture_metadata():
    """modelspec.architecture must be 'nucleus_image'."""
    from safetensors import safe_open

    with tempfile.TemporaryDirectory() as td:
        path, _ = _save_lora(td)
        with safe_open(str(path), framework="pt") as f:
            metadata = f.metadata()

    assert metadata is not None
    assert metadata.get("modelspec.architecture") == "nucleus_image", (
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


def test_lora_loads_via_transformer_peft_adapter_mixin_default_prefix():
    """The saved key format matches ``PeftAdapterMixin.load_lora_adapter``'s
    DEFAULT ``prefix="transformer"`` argument (``diffusers/loaders/peft.py``
    line 80-81) — i.e. every key must be prefixed with ``transformer``, the
    literal string the default strips. Also confirms NO pipeline-level
    mixin exists yet for this family (the honest gap documented in
    ``saver.py``)."""
    from diffusers.loaders.peft import PeftAdapterMixin
    from diffusers.models.transformers.transformer_nucleusmoe_image import (
        NucleusMoEImageTransformer2DModel,
    )
    import inspect

    assert issubclass(NucleusMoEImageTransformer2DModel, PeftAdapterMixin)
    sig = inspect.signature(PeftAdapterMixin.load_lora_adapter)
    assert sig.parameters["prefix"].default == "transformer"

    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)
    for k in sd:
        assert k.split(".", 1)[0] == "transformer"

    # No pipeline-level mixin exists for this family yet (honest gap).
    import diffusers.loaders.lora_pipeline as lora_pipeline_mod

    assert not hasattr(lora_pipeline_mod, "NucleusMoEImageLoraLoaderMixin")


def test_definition_ships_curated_target_list_matching_driver():
    """nucleus-image's YAML ships the SAME curated list
    driver.get_lora_targets() returns (no enrichment-drift risk)."""
    from app.engine.models.registry import ModelRegistry

    ModelRegistry._discovered = False
    ModelRegistry._families = {}
    ModelRegistry._definitions = {}
    ModelRegistry._definitions_loaded = False
    ModelRegistry.initialize()

    drv = _make_driver()
    expected = set(drv.get_lora_targets())

    defn = ModelRegistry._definitions["nucleus-image"]
    shipped = set(defn.lora_targetable_modules or [])
    assert shipped, "YAML must ship a non-empty LoRA target list"
    assert shipped == expected, (
        f"shipped list diverges from driver.get_lora_targets(): "
        f"+{shipped - expected} -{expected - shipped}"
    )
