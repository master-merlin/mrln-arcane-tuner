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
