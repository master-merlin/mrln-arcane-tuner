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


# ── Row 2.9: LoRA interop PROOF, both directions ──────────────────────────────
#
# Outbound: an artifact written by the row-2.8 saver — adapter weights
# NON-ZERO and `alpha/r != 1` (r=4, alpha=8) — is loaded through diffusers'
# OWN path (the private converter, then `load_lora_adapter`, the fallback
# DISABLED) onto a fresh copy of the base, and the upstream model's output on a
# fixed packed batch equals our PEFT-wrapped model's within 1e-5. The
# production fallback (our own inverse in `lora_keys.py`) is tested
# SEPARATELY with the private symbol absent. Inbound: kohya and bare ComfyUI
# layouts land on the model through `lora_keys.py` and reproduce the delta.

_PRIVATE_CONVERTER = "_convert_non_diffusers_minimax_h3_lora_to_diffusers"


def _fixed_batch():
    from app.engine.models.families.minimax_h3.packing import H3Geometry, build_layout, pack_audio, patchify_video
    import torch

    geometry = H3Geometry(num_text=8, latent_frames=2, latent_height=6, latent_width=4, audio_latents=3)
    layout = build_layout(geometry)
    g = torch.Generator().manual_seed(0)
    video = torch.randn(1, 24, 2, 6, 4, generator=g)
    audio = torch.randn(1, 2, 32, 3, generator=g)
    text = torch.randn(1, 8, 16, generator=g)
    return layout, patchify_video(video, layout.patch_size), pack_audio(audio), text, torch.tensor([0.7, 0.4])


def _forward(model):
    import torch

    from app.engine.models.families.minimax_h3.packing import packed_forward

    layout, video_rows, audio_rows, text, timesteps = _fixed_batch()
    with torch.no_grad():
        v, a = packed_forward(model, layout, video_rows, audio_rows, text, timesteps)
    return v.float(), a.float()


def _outbound_artifact(tmp_path, build_tiny_transformer, *, r: int = 4, alpha: float = 8.0):
    """Save the tiny PEFT model through the REAL driver saver; returns
    (path, ours_video, ours_audio). The fixture guard refuses a LoRA whose
    delta is zero — PEFT's zero-init `lora_B` would make any mapping pass."""
    from app.engine.models.families.minimax_h3.lora_keys import adapters_from_peft_model
    from app.engine.models.registry import ModelRegistry
    from app.engine.models.families.minimax_h3.driver import MiniMaxH3Driver
    import torch

    model = _lora_model(build_tiny_transformer, r=r, alpha=alpha)
    for module, adapter in adapters_from_peft_model(model).items():
        assert float(_delta(adapter).abs().sum()) > 0, f"adapter delta is zero; the proof proves nothing ({module})"
        assert adapter.scaling != 1.0, f"alpha/r == 1 on {module}: a missing fold would be invisible"
    definition = ModelRegistry._definitions["minimax-h3-t2va"]
    saver = MiniMaxH3Driver(definition, torch.device("cpu")).get_saver()
    path = tmp_path / "h3_outbound.safetensors"
    saver.save({"unet": model, "config": {"save_precision": "fp32"}}, path, metadata={})
    ours_v, ours_a = _forward(model)
    return path, ours_v, ours_a


def test_outbound_artifact_loads_through_diffusers_converter_without_fallback(tmp_path, build_tiny_transformer):
    from app.engine.models.families.minimax_h3.lora_keys import load_via_diffusers

    path, _, _ = _outbound_artifact(tmp_path, build_tiny_transformer)
    fresh = build_tiny_transformer().eval()
    result = load_via_diffusers(fresh, path, allow_fallback=False)
    assert result.fallback_used is False, "fallback_used is True"
    assert getattr(fresh, "peft_config", None), "no adapter was injected by diffusers' path"
    adapted = {n.removesuffix(".lora_A") for n, m in fresh.named_modules() if n.endswith(".lora_A")}
    assert len(adapted) == 3 * 6, f"{len(adapted)} adapted modules, expected 3 blocks x 6 targets: {sorted(adapted)[:4]}"


def test_outbound_forward_matches_ours(tmp_path, build_tiny_transformer):
    import torch

    from app.engine.models.families.minimax_h3.lora_keys import load_via_diffusers

    path, ours_v, ours_a = _outbound_artifact(tmp_path, build_tiny_transformer)
    fresh = build_tiny_transformer().eval()
    base_v, base_a = _forward(fresh)
    assert not torch.allclose(base_v, ours_v, atol=1e-5), "the LoRA has no effect on the output: the proof is vacuous"
    result = load_via_diffusers(fresh, path, allow_fallback=False)
    assert result.fallback_used is False
    theirs_v, theirs_a = _forward(fresh)
    assert torch.allclose(theirs_v, ours_v, atol=1e-5), (
        f"video output mismatch through diffusers' path: max |diff| {float((theirs_v - ours_v).abs().max()):.3e}"
    )
    assert torch.allclose(theirs_a, ours_a, atol=1e-5), (
        f"audio output mismatch through diffusers' path: max |diff| {float((theirs_a - ours_a).abs().max()):.3e}"
    )


def test_production_fallback_runs_when_private_symbol_absent(tmp_path, build_tiny_transformer, monkeypatch):
    import torch
    from diffusers.loaders import lora_conversion_utils

    from app.engine.models.families.minimax_h3.lora_keys import load_via_diffusers

    path, ours_v, ours_a = _outbound_artifact(tmp_path, build_tiny_transformer)

    def _gone(*args, **kwargs):
        raise ImportError("private converter removed in this diffusers")

    monkeypatch.setattr(lora_conversion_utils, _PRIVATE_CONVERTER, _gone)
    fresh = build_tiny_transformer().eval()
    with pytest.raises(ImportError):
        load_via_diffusers(fresh, path, allow_fallback=False)
    fresh = build_tiny_transformer().eval()
    result = load_via_diffusers(fresh, path)
    assert result.fallback_used is True, "the lora_keys inverse did not run"
    theirs_v, theirs_a = _forward(fresh)
    assert torch.allclose(theirs_v, ours_v, atol=1e-5) and torch.allclose(theirs_a, ours_a, atol=1e-5), (
        "the fallback path does not reproduce our output"
    )


def _injected_delta(model, module: str):
    """scaling · B @ A of the adapter diffusers injected on `module`."""
    layer = model.get_submodule(module)
    a = layer.lora_A["default"].weight.float()
    b = layer.lora_B["default"].weight.float()
    return float(layer.scaling["default"]) * (b @ a)


def _kohya_lora(*, r: int = 4, alpha: float = 2.0):
    """A synthetic musubi/kohya-layout LoRA (flattened `lora_unet_` names,
    `lora_down`/`lora_up`, a per-module `.alpha`) for block 0 + refiner 0."""
    import torch

    g = torch.Generator().manual_seed(21)
    hidden, inner, ffn = 16, 32, 32  # tiny arch: 2 heads x 16 -> attention inner 32; ffn_dim 32
    sd = {}
    for mod, in_features, out_rows in (
        ("lora_unet_blocks_0_attn_qkv_proj", hidden, 3 * inner),
        ("lora_unet_blocks_0_mlp_fc1", hidden, 2 * ffn),
        ("lora_unet_blocks_0_attn_out_proj", inner, hidden),
        ("lora_unet_token_refiner_blocks_0_mlp_fc2", ffn, hidden),
    ):
        sd[f"{mod}.lora_down.weight"] = torch.randn(r, in_features, generator=g)
        sd[f"{mod}.lora_up.weight"] = torch.randn(out_rows, r, generator=g)
        sd[f"{mod}.alpha"] = torch.tensor(alpha)
    return sd


def test_inbound_kohya_layout_loads(build_tiny_transformer):
    import torch

    from app.engine.models.families.minimax_h3.lora_keys import load_via_lora_keys

    r, alpha = 4, 2.0
    sd = _kohya_lora(r=r, alpha=alpha)
    model = build_tiny_transformer().eval()
    result = load_via_lora_keys(model, sd)
    assert result.modules == 4, f"{result.modules} native modules mapped (qkv_proj counts as one)"
    scale = alpha / r
    inner = 32
    down, up = sd["lora_unet_blocks_0_attn_qkv_proj.lora_down.weight"], sd["lora_unet_blocks_0_attn_qkv_proj.lora_up.weight"]
    for i, proj in enumerate(("to_q", "to_k", "to_v")):
        expected = scale * (up[i * inner : (i + 1) * inner] @ down)
        got = _injected_delta(model, f"transformer_blocks.0.attn.{proj}")
        assert torch.allclose(got, expected, atol=1e-5), f"kohya qkv_proj third {i} did not land on {proj}"
    down, up = sd["lora_unet_blocks_0_mlp_fc1.lora_down.weight"], sd["lora_unet_blocks_0_mlp_fc1.lora_up.weight"]
    half = up.shape[0] // 2
    expected = scale * (torch.cat([up[half:], up[:half]], dim=0) @ down)  # [gate; value] -> [value; gate]
    assert torch.allclose(_injected_delta(model, "transformer_blocks.0.ff.net.0.proj"), expected, atol=1e-5), (
        "kohya fc1 halves were not swapped into diffusers' [value; gate] order"
    )
    down, up = sd["lora_unet_blocks_0_attn_out_proj.lora_down.weight"], sd["lora_unet_blocks_0_attn_out_proj.lora_up.weight"]
    assert torch.allclose(_injected_delta(model, "transformer_blocks.0.attn.to_out.0"), scale * (up @ down), atol=1e-5), (
        "kohya alpha was not applied on out_proj"
    )
    down, up = (
        sd["lora_unet_token_refiner_blocks_0_mlp_fc2.lora_down.weight"],
        sd["lora_unet_token_refiner_blocks_0_mlp_fc2.lora_up.weight"],
    )
    assert torch.allclose(
        _injected_delta(model, "token_refiner.refiner_blocks.0.ff.net.2"), scale * (up @ down), atol=1e-5
    ), "the refiner block did not map"


def test_inbound_comfy_layout_loads(build_tiny_transformer):
    """Bare ComfyUI layout: native names, `lora_A`/`lora_B`, no alpha (scale 1),
    with and without the `diffusion_model.` prefix."""
    import torch

    from app.engine.models.families.minimax_h3.lora_keys import load_via_lora_keys

    g = torch.Generator().manual_seed(5)
    r, hidden, inner = 3, 16, 32
    a_qkv, b_qkv = torch.randn(r, hidden, generator=g), torch.randn(3 * inner, r, generator=g)
    a_out, b_out = torch.randn(r, inner, generator=g), torch.randn(hidden, r, generator=g)  # out_proj: inner -> hidden
    for prefix in ("", "diffusion_model."):
        sd = {
            f"{prefix}blocks.1.attn.qkv_proj.lora_A.weight": a_qkv,
            f"{prefix}blocks.1.attn.qkv_proj.lora_B.weight": b_qkv,
            f"{prefix}token_refiner.blocks.0.attn.out_proj.lora_A.weight": a_out,
            f"{prefix}token_refiner.blocks.0.attn.out_proj.lora_B.weight": b_out,
        }
        model = build_tiny_transformer().eval()
        result = load_via_lora_keys(model, sd)
        assert result.modules == 2, f"prefix {prefix!r}: {result.modules} modules mapped"
        for i, proj in enumerate(("to_q", "to_k", "to_v")):
            expected = b_qkv[i * inner : (i + 1) * inner] @ a_qkv
            assert torch.allclose(_injected_delta(model, f"transformer_blocks.1.attn.{proj}"), expected, atol=1e-5), (
                f"prefix {prefix!r}: comfy qkv third {i} did not land on {proj}"
            )
        assert torch.allclose(
            _injected_delta(model, "token_refiner.refiner_blocks.0.attn.to_out.0"), b_out @ a_out, atol=1e-5
        ), f"prefix {prefix!r}: the refiner out_proj did not map"
