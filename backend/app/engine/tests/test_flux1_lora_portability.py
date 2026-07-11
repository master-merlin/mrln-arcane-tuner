"""FLUX.1 LoRA portability: saved keys must load in stock ComfyUI.

FLUX.1 trains against diffusers ``FluxTransformer2DModel`` with diffusers
module names (``transformer_blocks.N.attn.to_q``, ``single_transformer_
blocks.N.proj_mlp``, ...). ComfyUI's Flux model is BFL-native
(``double_blocks.*``/``single_blocks.*``), and its LoRA ``key_map``
(``comfy/lora.py::model_lora_keys_unet``) maps diffusers-named LoRAs ONLY
under the ``transformer.<module>`` / bare ``<module>`` keys registered by
``comfy.utils.flux_to_diffusers``. The ``diffusion_model.`` prefix is paired
exclusively with BFL-native names via the generic block (built from the
model's own state dict).

Historic bug (same as ovis_image, which ComfyUI loads through this very Flux
path): our saver emitted ``diffusion_model.<diffusers_module>`` — a hybrid
matching NOTHING in the Flux key_map → every key logged "lora key not
loaded" and the LoRA silently applied with zero effect. Format of record is
now ``transformer.{module}.lora_A/B.weight``.

All three shipped definitions (dev, schnell, kontext_dev) share the same
suffix-pattern target list (verified in their YAMLs), so the tiny
1 double + 1 single block model here covers the whole family's key surface.
"""

from __future__ import annotations

import pathlib
import tempfile
from unittest.mock import MagicMock

import torch


_TINY_CFG = dict(
    patch_size=1,
    in_channels=4,
    num_layers=1,
    num_single_layers=1,
    attention_head_dim=8,
    num_attention_heads=2,
    joint_attention_dim=16,
    pooled_projection_dim=8,
    axes_dims_rope=(2, 2, 4),
)


def _make_driver():
    from app.engine.models.families.flux1.driver import Flux1Driver

    definition = MagicMock()
    definition.family = "flux1"
    definition.id = "flux1-test"
    definition.lora_targetable_modules = []
    definition.architecture_params = {}
    return Flux1Driver(definition, torch.device("cpu"))


def _build_peft_model(driver):
    from peft import LoraConfig, get_peft_model
    from diffusers import FluxTransformer2DModel

    base = FluxTransformer2DModel(**_TINY_CFG)
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=4,
        target_modules=driver.get_lora_targets(),
    )
    return get_peft_model(base, lora_cfg)


def _save_lora(tmp_dir: str):
    from safetensors.torch import load_file

    drv = _make_driver()
    peft_model = _build_peft_model(drv)
    saver = drv.get_saver()
    path = pathlib.Path(tmp_dir) / "flux1_lora.safetensors"
    saver.save(components={"unet": peft_model, "config": {}}, path=path)
    assert path.exists(), "Saver did not produce a safetensors file"
    return path, load_file(str(path))


def test_saver_key_format_is_transformer_prefixed():
    """All keys are transformer.{module}.lora_A/B.weight — the only prefix
    ComfyUI's Flux LoRA route maps for diffusers module names."""
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

    lora_a = [k for k in sd if ".lora_A." in k]
    lora_b = [k for k in sd if ".lora_B." in k]
    assert len(lora_a) == len(lora_b), "lora_A/lora_B counts must match"


def test_saver_keys_match_comfyui_flux_transformer_route():
    """Spot-check every module class against the exact diffusers names
    ``comfy.utils.flux_to_diffusers`` registers under ``transformer.``:

    - double block: ``attn.to_q/to_k/to_v`` -> ``img_attn.qkv`` slices,
      ``attn.add_*_proj`` -> ``txt_attn.qkv`` slices, ``attn.to_out.0`` /
      ``attn.to_add_out`` -> ``img/txt_attn.proj``, ``ff.net.0.proj`` /
      ``ff.net.2`` -> ``img_mlp.0/2``, ``ff_context.*`` -> ``txt_mlp.0/2``
    - single block: ``attn.to_q/to_k/to_v`` + ``proj_mlp`` -> ``linear1``
      slices, ``proj_out`` -> ``linear2``
    - top level: ``proj_out`` -> ``final_layer.linear`` (MAP_BASIC; the flux1
      driver ships no exclude, so the top-level projection IS trained and
      must stay mappable)
    """
    with tempfile.TemporaryDirectory() as td:
        _, sd = _save_lora(td)

    must_have = [
        # double-block attention
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer.transformer_blocks.0.attn.to_k.lora_B.weight",
        "transformer.transformer_blocks.0.attn.to_v.lora_A.weight",
        "transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight",
        "transformer.transformer_blocks.0.attn.add_q_proj.lora_A.weight",
        "transformer.transformer_blocks.0.attn.add_k_proj.lora_B.weight",
        "transformer.transformer_blocks.0.attn.add_v_proj.lora_A.weight",
        "transformer.transformer_blocks.0.attn.to_add_out.lora_B.weight",
        # double-block feed-forward
        "transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight",
        "transformer.transformer_blocks.0.ff.net.2.lora_B.weight",
        "transformer.transformer_blocks.0.ff_context.net.0.proj.lora_A.weight",
        "transformer.transformer_blocks.0.ff_context.net.2.lora_B.weight",
        # single-block
        "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight",
        "transformer.single_transformer_blocks.0.attn.to_k.lora_B.weight",
        "transformer.single_transformer_blocks.0.attn.to_v.lora_A.weight",
        "transformer.single_transformer_blocks.0.proj_mlp.lora_B.weight",
        "transformer.single_transformer_blocks.0.proj_out.lora_A.weight",
        # top-level final projection (flux_to_diffusers MAP_BASIC)
        "transformer.proj_out.lora_A.weight",
    ]
    missing = [k for k in must_have if k not in sd]
    assert not missing, (
        f"keys ComfyUI's flux_to_diffusers transformer. route expects are "
        f"absent from the saved LoRA: {missing}"
    )


def test_saver_architecture_metadata():
    """modelspec.architecture must be 'flux1'."""
    from safetensors import safe_open

    with tempfile.TemporaryDirectory() as td:
        path, _ = _save_lora(td)
        with safe_open(str(path), framework="pt") as f:
            metadata = f.metadata()

    assert metadata is not None
    assert metadata.get("modelspec.architecture") == "flux1"


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
