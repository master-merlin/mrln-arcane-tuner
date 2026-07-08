"""Kandinsky 5.0 LoRA saver + upstream-mixin portability (tiny REAL model).

Pins:

1. Saved key format ``diffusion_model.visual_transformer_blocks.{i}...lora_A/B.weight``
   with the canonical count (10 targets x 2 tensors per visual block; the full
   Lite checkpoint pins at 640 tensors).
2. Round-trip back onto a fresh PEFT-wrapped tiny model (zero missing /
   unexpected LoRA keys).
3. **Bidirectional mapping to the upstream ``KandinskyLoraLoaderMixin``
   format** — kandinsky5 is our only new family with a real upstream mixin:
   ``to_diffusers_lora`` output must load through the exact seam the mixin
   uses (``Kandinsky5Transformer3DModel.load_lora_adapter``,
   ``prefix="transformer"``), and ``from_diffusers_lora`` must invert
   losslessly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from app.engine.models.families.kandinsky5.driver import (
    Kandinsky5Driver,
    tiny_transformer_config,
)
from app.engine.models.families.kandinsky5.saver import (
    from_diffusers_lora,
    to_diffusers_lora,
)

# Full Lite checkpoint: 32 blocks x 10 targets x lora_A+lora_B.
LITE_PINNED_KEY_COUNT = 640
# Tiny 1-block model: 10 targets x 2.
TINY_PINNED_KEY_COUNT = 20


def _definition() -> MagicMock:
    d = MagicMock()
    d.family = "kandinsky5"
    d.id = "k5-test"
    d.lora_targetable_modules = []
    d.architecture_params = {
        "mode": "t2v",
        "transformer.num_visual_blocks": 1,
        "transformer.visual_cond": False,
    }
    return d


def _driver() -> Kandinsky5Driver:
    return Kandinsky5Driver(_definition(), torch.device("cpu"))


def _build_peft_model():
    from diffusers import Kandinsky5Transformer3DModel
    from peft import LoraConfig, get_peft_model

    base = Kandinsky5Transformer3DModel(**tiny_transformer_config())
    cfg = LoraConfig(r=4, lora_alpha=4, target_modules=_driver().get_lora_targets())
    return get_peft_model(base, cfg)


def _save_tiny_lora(tmp_path):
    model = _build_peft_model()
    saver = _driver().get_saver()
    path = tmp_path / "k5_lora.safetensors"
    saver.save(components={"unet": model, "config": {}}, path=path, metadata=None)
    assert path.exists(), "Saver did not produce a safetensors file"
    return load_file(str(path)), path


def test_saved_keys_format_and_pinned_count(tmp_path):
    sd, _ = _save_tiny_lora(tmp_path)

    assert len(sd) == TINY_PINNED_KEY_COUNT
    for key in sd:
        assert key.startswith("diffusion_model.visual_transformer_blocks."), key
        assert key.endswith(".weight"), key
        assert "lora_A" in key or "lora_B" in key, key
    # Custom Kandinsky attention naming — NOT to_q/to_k/to_v.
    assert any(".self_attention.to_query." in k for k in sd)
    assert any(".cross_attention.out_layer." in k for k in sd)
    assert any(".feed_forward.in_layer." in k for k in sd)
    assert not any("to_q." in k for k in sd)
    # Frozen stacks must never leak into the file.
    assert not any("text_transformer_blocks" in k for k in sd)
    assert not any("time_embeddings" in k for k in sd)


def test_lite_key_count_pins_at_640():
    """32 blocks x 10 targets x 2 tensors — the full Lite canonical count."""
    d = _definition()
    d.architecture_params["transformer.num_visual_blocks"] = 32
    drv = Kandinsky5Driver(d, torch.device("cpu"))
    assert len(drv.get_lora_targets()) * 2 == LITE_PINNED_KEY_COUNT


def test_saver_architecture_metadata(tmp_path):
    _, path = _save_tiny_lora(tmp_path)
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata()
    assert metadata["modelspec.architecture"] == "kandinsky5"


def test_lora_round_trips_onto_fresh_model(tmp_path):
    sd, _ = _save_tiny_lora(tmp_path)
    fresh = _build_peft_model()

    def _remap_to_peft(key: str) -> str:
        module_path = key[len("diffusion_model.") :]
        module_path = module_path.replace(".lora_A.weight", ".lora_A.default.weight")
        module_path = module_path.replace(".lora_B.weight", ".lora_B.default.weight")
        return f"base_model.model.{module_path}"

    remapped = {_remap_to_peft(k): v for k, v in sd.items()}
    missing, unexpected = fresh.load_state_dict(remapped, strict=False)
    lora_missing = [k for k in missing if "lora" in k.lower()]
    assert not lora_missing, f"LoRA keys missing on reload: {lora_missing[:5]}"
    assert not unexpected, f"Unexpected keys on reload: {unexpected[:5]}"


# ── Upstream KandinskyLoraLoaderMixin mapping ──────────────────────────────


def test_to_diffusers_lora_produces_mixin_native_keys(tmp_path):
    sd, _ = _save_tiny_lora(tmp_path)
    native = to_diffusers_lora(sd)
    assert len(native) == len(sd)
    for key in native:
        assert key.startswith("transformer.visual_transformer_blocks."), key
    # Value identity (pure key remap, no tensor mutation).
    ours = next(iter(sd))
    theirs = "transformer." + ours[len("diffusion_model.") :]
    assert torch.equal(sd[ours], native[theirs])


def test_mapping_is_bidirectional_and_lossless(tmp_path):
    sd, _ = _save_tiny_lora(tmp_path)
    back = from_diffusers_lora(to_diffusers_lora(sd))
    assert set(back) == set(sd)
    assert all(torch.equal(back[k], sd[k]) for k in sd)


def test_converted_file_loads_through_the_upstream_mixin_seam(tmp_path):
    """PROOF of convertibility: `to_diffusers_lora` output loads via
    ``Kandinsky5Transformer3DModel.load_lora_adapter`` — the exact call
    ``KandinskyLoraLoaderMixin.load_lora_into_transformer`` makes (with the
    mixin's default ``prefix="transformer"``) — with every A/B pair applied.
    """
    from diffusers import Kandinsky5Transformer3DModel

    sd, _ = _save_tiny_lora(tmp_path)
    native = to_diffusers_lora(sd)

    target = Kandinsky5Transformer3DModel(**tiny_transformer_config())
    target.load_lora_adapter(dict(native), adapter_name="default")

    # The adapter must be materialized on every visual-block target module...
    lora_modules = [
        name
        for name, module in target.named_modules()
        if hasattr(module, "lora_A") and getattr(module, "lora_A", None)
    ]
    assert len(lora_modules) == TINY_PINNED_KEY_COUNT // 2
    assert all("visual_transformer_blocks" in n for n in lora_modules)

    # ...with our exact tensors (spot-check every lora_A/B weight).
    loaded_sd = {
        k: v for k, v in target.state_dict().items() if "lora_" in k
    }
    for our_key, tensor in native.items():
        module_path = our_key[len("transformer.") :]
        peft_key = module_path.replace(".lora_A.weight", ".lora_A.default.weight")
        peft_key = peft_key.replace(".lora_B.weight", ".lora_B.default.weight")
        assert peft_key in loaded_sd, f"{peft_key} not found after mixin load"
        assert torch.equal(loaded_sd[peft_key], tensor)


def test_upstream_mixin_exists_with_expected_contract():
    """Guard: the upstream mixin still targets the transformer with the
    SD3-style prefix our mapping assumes."""
    from diffusers.loaders.lora_pipeline import KandinskyLoraLoaderMixin

    assert KandinskyLoraLoaderMixin._lora_loadable_modules == ["transformer"]
    assert KandinskyLoraLoaderMixin.transformer_name == "transformer"
