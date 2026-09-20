"""Finding F-1, arm A — the native ↔ diffusers key plan over the committed
key-NAME fixture (plan row 1.6; no weights, always runs).

The risk this pins (research §3.9): a LoRA saved under diffusers' module
names (`transformer_blocks.N.attn.to_q`) interoperates with nothing — ComfyUI,
ai-toolkit and diffusion-pipe all speak the checkpoint's own names
(`blocks.N.attn.qkv_proj`). ``lora_keys.py`` is the ONE place that knows how
the two conventions relate; row 2.8 extends the same map for the saver. The
fixture ``fixtures/h3_index_keys.json`` holds the 535 native + 638 converted
key names captured once from the on-disk indexes (sha256s inside), so the
plan is exercised against the real key sets without the 465 GB snapshot.

Arm B (the live-index read, ``.agent/workdir/minimax-pr1/f1_key_plan.py``)
asserts this fixture still equals the indexes on disk and writes
``.agent/output/h3-gates/F-1.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.engine.models.families.minimax_h3.lora_keys import (
    DROPPED_NATIVE_KEYS,
    map_native_key,
    native_to_diffusers,
    unmapped_keys,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "h3_index_keys.json"


@pytest.fixture(scope="module")
def index_keys() -> dict:
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert len(data["native"]) == 535, "fixture drifted: native key count"
    assert len(data["converted"]) == 638, "fixture drifted: converted key count"
    return data


def test_f1_every_native_key_maps(index_keys):
    plan = native_to_diffusers(index_keys["native"])
    unmapped = unmapped_keys(plan)
    assert not unmapped, f"{len(unmapped)} unmapped native keys: {unmapped[:5]}"
    assert plan.keys() == set(index_keys["native"])
    assert all(m.targets for m in plan.values() if m.transform != "drop")


def test_f1_every_converted_key_is_reached(index_keys):
    plan = native_to_diffusers(index_keys["native"])
    reached: dict[str, str] = {}
    for m in plan.values():
        for target in m.targets:
            assert target not in reached, f"{target} reached twice: {reached[target]} and {m.native}"
            reached[target] = m.native
    converted = set(index_keys["converted"])
    missing = sorted(converted - reached.keys())
    invented = sorted(reached.keys() - converted)
    assert not missing, f"{len(missing)} converted keys never reached: {missing[:5]}"
    assert not invented, f"{len(invented)} targets not in the converted index: {invented[:5]}"


def test_f1_only_rope_inv_freq_dropped(index_keys):
    plan = native_to_diffusers(index_keys["native"])
    dropped = sorted(k for k, m in plan.items() if m.transform == "drop")
    assert dropped == ["rope.inv_freq"], f"dropped set drifted: {dropped}"
    assert DROPPED_NATIVE_KEYS == ("rope.inv_freq",)
    assert map_native_key("rope.inv_freq").targets == ()


def test_f1_transforms_are_the_converters(index_keys):
    """The map records WHAT the converter does to each tensor, not just the
    name: the fused QKV splits into contiguous thirds q/k/v, `mlp.fc1` keeps
    its fused shape with the two halves swapped, everything else copies."""
    qkv = map_native_key("blocks.7.attn.qkv_proj.weight")
    assert qkv.transform == "qkv_split"
    assert qkv.targets == (
        "transformer_blocks.7.attn.to_q.weight",
        "transformer_blocks.7.attn.to_k.weight",
        "transformer_blocks.7.attn.to_v.weight",
    )
    fc1 = map_native_key("token_refiner.blocks.1.mlp.fc1.weight")
    assert fc1.transform == "swiglu_swap"
    assert fc1.targets == ("token_refiner.refiner_blocks.1.ff.net.0.proj.weight",)
    assert map_native_key("condition_proj.weight").targets == ("context_embedder.weight",)
    assert map_native_key("final_layer.adaln_proj.linear.bias").targets == ("norm_out.linear.bias",)
    assert map_native_key("time_embedder.proj_out.weight").targets == ("time_embedder.linear_2.weight",)
    plan = native_to_diffusers(index_keys["native"])
    kinds = {m.transform for m in plan.values()}
    assert kinds == {"copy", "qkv_split", "swiglu_swap", "drop"}
    assert sum(m.transform == "qkv_split" for m in plan.values()) == 52  # 50 blocks + 2 refiner blocks


def test_f1_unknown_key_raises():
    with pytest.raises(KeyError, match="blocks.0.attn.mystery.weight"):
        map_native_key("blocks.0.attn.mystery.weight")


# ── Row 2.8: the artifact layout (PEFT adapters -> the ORIGINAL checkpoint
# names). Proven against the INSTALLED consumer: diffusers 0.40's
# `_convert_non_diffusers_minimax_h3_lora_to_diffusers` must reach every
# targeted module and reproduce every delta with no `.alpha` key present.


def _lora_model(build_tiny_transformer, *, r: int = 2, alpha: float = 2.0):
    import torch
    from peft import LoraConfig, get_peft_model

    from app.engine.models.registry import ModelRegistry

    # The YAML is the single source of truth for the targets (Task 4).
    ModelRegistry._definitions_loaded = False
    ModelRegistry._definitions = {}
    ModelRegistry.initialize()
    targets = list(ModelRegistry._definitions["minimax-h3-t2va"].lora_targetable_modules)
    model = get_peft_model(build_tiny_transformer(), LoraConfig(r=r, lora_alpha=alpha, target_modules=targets))
    g = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".lora_B." in name:  # PEFT zero-inits B: randomise so every delta is non-zero
                p.copy_(torch.randn(p.shape, generator=g))
    return model.eval()


def _adapters_and_artifact(build_tiny_transformer, **kw):
    from app.engine.models.families.minimax_h3.lora_keys import adapters_from_peft_model, build_artifact

    adapters = adapters_from_peft_model(_lora_model(build_tiny_transformer, **kw))
    return adapters, build_artifact(adapters)


def _delta(adapter):
    return adapter.scaling * (adapter.lora_B.float() @ adapter.lora_A.float())


def test_key_map_round_trips_every_targeted_module(build_tiny_transformer):
    import torch
    from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_minimax_h3_lora_to_diffusers

    adapters, artifact = _adapters_and_artifact(build_tiny_transformer)
    assert adapters, "no PEFT adapters found on the tiny model"
    assert any(k.startswith("diffusion_model.blocks.") for k in artifact), "no key starts with diffusion_model.blocks."
    assert all(
        k.startswith(("diffusion_model.blocks.", "diffusion_model.token_refiner.blocks.")) for k in artifact
    ), sorted(artifact)[:4]
    # The installed consumer maps the artifact back onto EVERY diffusers
    # module PEFT trained - and no other.
    back = _convert_non_diffusers_minimax_h3_lora_to_diffusers(dict(artifact))
    expected = {f"transformer.{m}.lora_{ab}.weight" for m in adapters for ab in ("A", "B")}
    assert set(back) == expected, (
        f"missing={sorted(expected - set(back))[:4]} extra={sorted(set(back) - expected)[:4]}"
    )
    for module, adapter in adapters.items():
        got = back[f"transformer.{module}.lora_B.weight"].float() @ back[f"transformer.{module}.lora_A.weight"].float()
        assert torch.allclose(got, _delta(adapter), atol=1e-5), f"delta mismatch after the round trip: {module}"


def test_fused_qkv_delta_equals_split_deltas(build_tiny_transformer):
    import torch

    adapters, artifact = _adapters_and_artifact(build_tiny_transformer)
    q, k, v = (adapters[f"transformer_blocks.0.attn.to_{p}"] for p in "qkv")
    a_fused = artifact["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight"].float()
    b_fused = artifact["diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight"].float()
    r, hidden = q.lora_A.shape
    inner = q.lora_B.shape[0]
    assert a_fused.shape == (3 * r, hidden), f"A_fused {tuple(a_fused.shape)} != (3r, hidden) {(3 * r, hidden)}"
    assert b_fused.shape == (3 * inner, 3 * r), f"B_fused {tuple(b_fused.shape)} != (3*inner, 3r)"
    delta = b_fused @ a_fused
    for i, adapter in enumerate((q, k, v)):
        assert torch.allclose(delta[i * inner : (i + 1) * inner], _delta(adapter), atol=1e-5), (
            f"delta mismatch for projection {'qkv'[i]}: the fusion is not exact"
        )
    # Block-diagonal: the q rows touch only the q rank columns, etc.
    for i in range(3):
        for j in range(3):
            block = b_fused[i * inner : (i + 1) * inner, j * r : (j + 1) * r]
            assert bool(block.abs().sum() > 0) is (i == j), (
                f"B_fused block ({i},{j}) is {'zero' if i == j else 'non-zero'}"
            )


def test_fc1_halves_swapped_back(build_tiny_transformer):
    import torch

    adapters, artifact = _adapters_and_artifact(build_tiny_transformer)
    fc1 = adapters["transformer_blocks.0.ff.net.0.proj"]
    b_native = artifact["diffusion_model.blocks.0.mlp.fc1.lora_B.weight"].float()
    a_native = artifact["diffusion_model.blocks.0.mlp.fc1.lora_A.weight"].float()
    half = fc1.lora_B.shape[0] // 2
    b_peft = fc1.scaling * fc1.lora_B.float()
    # diffusers SwiGLU rows are [value; gate]; the checkpoint fc1 is [gate; value].
    assert torch.allclose(b_native[:half], b_peft[half:]) and torch.allclose(b_native[half:], b_peft[:half]), (
        "fc1 lora_B halves were not swapped back to [gate; value]"
    )
    assert torch.equal(a_native, fc1.lora_A.float()), "fc1 lora_A must be untouched (the swap permutes OUTPUT rows)"


def test_alpha_folded_no_alpha_key(build_tiny_transformer):
    import torch

    adapters, artifact = _adapters_and_artifact(build_tiny_transformer, r=2, alpha=4.0)
    assert not any(k.endswith(".alpha") for k in artifact), "an .alpha key was written"
    out = adapters["transformer_blocks.0.attn.to_out.0"]
    assert out.scaling == 2.0, f"PEFT scaling alpha/r should be 2.0, got {out.scaling}"
    a = artifact["diffusion_model.blocks.0.attn.out_proj.lora_A.weight"].float()
    b = artifact["diffusion_model.blocks.0.attn.out_proj.lora_B.weight"].float()
    assert torch.equal(a, out.lora_A.float()), "lora_A must carry no scaling"
    assert torch.allclose(b, 2.0 * out.lora_B.float()), "alpha/r was not folded into lora_B"
    assert torch.allclose(b @ a, _delta(out), atol=1e-6), "a loader applying scale 1 does not reproduce the delta"
