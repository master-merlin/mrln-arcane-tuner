"""MiniMax-H3-Image's use of the shared training adapter, on CPU with a tiny model.

The adapter (ostris' ``minimax_h3_image_training_adapter_v1``) is a LoRA in the model's native
layout: ``diffusion_model.blocks.N.attn.qkv_proj`` (q/k/v fused, one rank-16 ``lora_A``),
``attn.out_proj``, ``mlp.fc1`` (SwiGLU rows ``[gate; value]``), ``mlp.fc2``, the same for
``token_refiner.blocks.0-1``, no AdaLN. ``lora_keys.training_adapter_from_native`` maps it onto
the diffusers modules PEFT trains. What this file holds it to:

* **the key map**: every native module of the real checkpoint's index reaches modules the
  diffusers index holds (0 unmatched), and a file for the tiny model attaches with 0 unmatched
  and 0 unused;
* **an independent converter**: diffusers' own ``_convert_non_diffusers_minimax_h3_lora_to_diffusers``
  reads the same file, and its per-module deltas equal ours and the attached layers' to 1e-6;
* **the job**: ``setup()`` fetches and checks the file before any weight loads (a wrong sha256
  never reaches ``load_model``); in a training step the adapter is active, gets no gradient,
  is not in the optimizer and does not change; the saved LoRA has exactly the user's keys;
  and ``none`` trains bit for bit as a definition without the block.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.engine.families.test_minimax_h3_image_trainer import (
    LORA_TARGETS,
    _latents,
    _text,
    _with_model,
    bare_trainer,
    stand_in_definition,
)
from tests.engine.families.test_minimax_h3_image_vendor import _tiny_transformer

_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "minimax_h3_image_hf_configs" / "transformer_index_keys.json"
_NATIVE_LINEAR = re.compile(r"^(?:token_refiner\.)?blocks\.\d+\.(?:attn\.qkv_proj|attn\.out_proj|mlp\.fc1|mlp\.fc2)\.weight$")
R = 4


def _native_file(model: torch.nn.Module, *, seed: int = 11) -> dict[str, torch.Tensor]:
    """A native-layout adapter for ``model``, shaped as ostris' file is: one dense rank-R pair
    per native module, q/k/v fused with one shared ``lora_A``, no AdaLN."""
    g = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    for stack, native_stack in (("transformer_blocks", "blocks"), ("token_refiner.refiner_blocks", "token_refiner.blocks")):
        blocks = model.get_submodule(stack)
        for n, block in enumerate(blocks):
            q = block.attn.to_q
            shapes = {
                "attn.qkv_proj": (q.in_features, 3 * q.out_features),
                "attn.out_proj": (block.attn.to_out[0].in_features, block.attn.to_out[0].out_features),
                "mlp.fc1": (block.ff.net[0].proj.in_features, block.ff.net[0].proj.out_features),
                "mlp.fc2": (block.ff.net[2].in_features, block.ff.net[2].out_features),
            }
            for native, (fan_in, fan_out) in shapes.items():
                key = f"diffusion_model.{native_stack}.{n}.{native}"
                tensors[f"{key}.lora_A.weight"] = 0.2 * torch.randn(R, fan_in, generator=g)
                tensors[f"{key}.lora_B.weight"] = 0.2 * torch.randn(fan_out, R, generator=g)
    return tensors


def _write(tmp_path: Path, tensors: dict[str, torch.Tensor]) -> tuple[Path, str]:
    from safetensors.torch import save_file

    path = tmp_path / "minimax_h3_image_training_adapter_v1.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _block(sha256: str, **changes: Any) -> dict[str, Any]:
    block = {
        "step": True,
        "guidance": True,
        "guidance_scale": 3.5,
        "training_adapter": {
            "repo": "ostris/minimax_h3_training_adapter",
            "revision": "e4e119caa391053286145e48ebbaf0e0c00293d3",
            "file": "minimax_h3_image_training_adapter_v1.safetensors",
            "sha256": sha256,
        },
        "default_handling": "adapter",
    }
    block.update(changes)
    return block


def _resolve_to(monkeypatch, path: Path) -> list[str]:
    from app.engine.utils.model_utils import ModelPathResolver

    seen: list[str] = []

    def resolve(uri, base_dir=None, *, local_files_only=False):
        seen.append(uri)
        return str(path)

    monkeypatch.setattr(ModelPathResolver, "resolve", staticmethod(resolve))
    return seen


# ── The key map ──────────────────────────────────────────────────────────────


def test_each_native_module_feeds_its_training_modules():
    from app.engine.models.families.minimax_h3_image.lora_keys import training_modules_for

    assert training_modules_for("blocks.7.attn.qkv_proj") == (
        "transformer_blocks.7.attn.to_q", "transformer_blocks.7.attn.to_k", "transformer_blocks.7.attn.to_v",
    )
    assert training_modules_for("blocks.0.attn.out_proj") == ("transformer_blocks.0.attn.to_out.0",)
    assert training_modules_for("blocks.0.mlp.fc1") == ("transformer_blocks.0.ff.net.0.proj",)
    assert training_modules_for("blocks.0.mlp.fc2") == ("transformer_blocks.0.ff.net.2",)
    assert training_modules_for("blocks.3.adaln_proj.linear") == ("transformer_blocks.3.adaln_proj.linear",)
    assert training_modules_for("token_refiner.blocks.1.attn.qkv_proj")[0] == "token_refiner.refiner_blocks.1.attn.to_q"
    for module in ("token_refiner.blocks.0.adaln_proj.linear", "pdd_proj_out.0", "blocks.0.attn.mystery"):
        with pytest.raises(KeyError, match=re.escape(module)):
            training_modules_for(module)


def test_every_native_module_of_the_checkpoint_reaches_the_diffusers_index():
    """The full-size adapter's modules (50 blocks + 2 refiner blocks, no AdaLN): 0 unmatched."""
    from app.engine.models.families.minimax_h3_image.lora_keys import training_modules_for

    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    native = sorted(k.removesuffix(".weight") for k in data["native"] if _NATIVE_LINEAR.match(k))
    assert len(native) == 50 * 4 + 2 * 4  # 416 tensors in ostris' file = 208 modules x (A, B)
    diffusers = set(data["diffusers"])
    reached = [m for module in native for m in training_modules_for(module)]
    unmatched = [m for m in reached if f"{m}.weight" not in diffusers]
    assert not unmatched, unmatched[:4]
    assert len(reached) == len(set(reached)) == 52 * 6


def test_a_key_outside_the_layout_is_refused():
    from app.engine.models.families.minimax_h3_image.lora_keys import training_adapter_from_native

    with pytest.raises(KeyError, match="diffusion_model."):
        training_adapter_from_native({"blocks.0.attn.out_proj.lora_A.weight": torch.zeros(1, 1)})
    with pytest.raises(ValueError, match="incomplete"):
        training_adapter_from_native({"diffusion_model.blocks.0.attn.out_proj.lora_A.weight": torch.zeros(1, 1)})


def test_the_file_converts_with_0_unmatched_and_0_unused(tmp_path):
    """The file on disk through the shared loader and the family's key map onto the tiny model."""
    from peft import LoraConfig, get_peft_model

    from app.engine.core import training_adapter
    from app.engine.models.families.minimax_h3_image import lora_keys

    path, _ = _write(tmp_path, _native_file(_tiny_transformer()))
    key_map = training_adapter.family_key_map(stand_in_definition())
    assert key_map is lora_keys.training_adapter_from_native
    pairs = training_adapter.load_pairs(str(path), key_map)
    lora_linears = {
        name
        for name, module in _tiny_transformer().named_modules()
        if isinstance(module, torch.nn.Linear) and re.search(r"(attn\.to_[qkv]|attn\.to_out\.0|ff\.net\.0\.proj|ff\.net\.2)$", name)
    }
    model = get_peft_model(_tiny_transformer(), LoraConfig(r=2, lora_alpha=2, target_modules=list(LORA_TARGETS)))
    assert set(pairs) == lora_linears  # 0 unused modules of the model, 0 modules it lacks
    assert training_adapter.attach(model, pairs) == len(pairs)


def test_an_alpha_is_folded_into_lora_b():
    from app.engine.models.families.minimax_h3_image.lora_keys import training_adapter_from_native

    a, b = torch.ones(2, 3), torch.ones(5, 2)
    pairs = training_adapter_from_native({
        "diffusion_model.blocks.0.mlp.fc2.lora_A.weight": a,
        "diffusion_model.blocks.0.mlp.fc2.lora_B.weight": b,
        "diffusion_model.blocks.0.mlp.fc2.alpha": torch.tensor(1.0),
    })
    torch.testing.assert_close(pairs["transformer_blocks.0.ff.net.2"][1], b * 0.5)


def test_the_upstream_converter_gives_the_same_deltas_and_so_do_the_attached_layers():
    from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_minimax_h3_lora_to_diffusers
    from peft import LoraConfig, get_peft_model

    from app.engine.core import training_adapter
    from app.engine.models.families.minimax_h3_image.lora_keys import training_adapter_from_native

    native = _native_file(_tiny_transformer())
    ours = training_adapter_from_native(dict(native))
    upstream = _convert_non_diffusers_minimax_h3_lora_to_diffusers(dict(native))
    assert set(upstream) == {f"transformer.{m}.lora_{ab}.weight" for m in ours for ab in ("A", "B")}
    assert len(ours) == 2 * 6 + 2 * 6
    for module, (a, b) in ours.items():
        delta = upstream[f"transformer.{module}.lora_B.weight"].float() @ upstream[f"transformer.{module}.lora_A.weight"].float()
        assert float(delta.abs().sum()) > 0, module
        torch.testing.assert_close(b @ a, delta, atol=1e-6, rtol=0, msg=lambda m, module=module: f"{module}: {m}")

    model = get_peft_model(_tiny_transformer(), LoraConfig(r=2, lora_alpha=2, target_modules=list(LORA_TARGETS)))
    assert training_adapter.attach(model, ours) == len(ours)  # 0 unmatched, 0 unused
    for module, (a, b) in ours.items():
        layer = model.get_submodule(f"base_model.model.{module}")
        scale = float(layer.scaling[training_adapter.TRAINING_ADAPTER])
        attached = scale * layer.lora_B[training_adapter.TRAINING_ADAPTER].weight @ layer.lora_A[training_adapter.TRAINING_ADAPTER].weight
        torch.testing.assert_close(attached.detach(), b @ a, atol=1e-6, rtol=0)


def test_the_qkv_split_is_the_fused_delta_in_thirds():
    from app.engine.models.families.minimax_h3_image.lora_keys import training_adapter_from_native

    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(R, 8, generator=g), torch.randn(3 * 6, R, generator=g)
    pairs = training_adapter_from_native({
        "diffusion_model.blocks.2.attn.qkv_proj.lora_A.weight": a,
        "diffusion_model.blocks.2.attn.qkv_proj.lora_B.weight": b,
    })
    fused = b @ a
    for i, part in enumerate("qkv"):
        pa, pb = pairs[f"transformer_blocks.2.attn.to_{part}"]
        torch.testing.assert_close(pb @ pa, fused[i * 6 : (i + 1) * 6], atol=1e-6, rtol=0)


# ── The job ──────────────────────────────────────────────────────────────────


def _trainer(definition, **config):
    trainer = bare_trainer(definition, timestep_sampling="fixed_sigmas", **config)
    asyncio.run(trainer.setup())
    return trainer


def _peft_trainer(definition, **config):
    trainer = _trainer(definition, **config)
    _with_model(trainer, _tiny_transformer())
    trainer._apply_peft()
    return trainer


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    path, sha = _write(tmp_path, _native_file(_tiny_transformer()))
    seen = _resolve_to(monkeypatch, path)
    return {"path": path, "sha": sha, "seen": seen}


def test_setup_fetches_and_checks_the_pinned_file(adapter):
    trainer = _trainer(stand_in_definition(distillation=_block(adapter["sha"])))
    assert trainer.distillation.handling == "adapter"
    assert trainer._training_adapter_path == str(adapter["path"])
    assert adapter["seen"] == [
        "huggingface:ostris/minimax_h3_training_adapter@e4e119caa391053286145e48ebbaf0e0c00293d3:"
        "minimax_h3_image_training_adapter_v1.safetensors"
    ]


def _setup_then_load(trainer) -> list[str]:
    """``run_trainer.run_async_trainer``'s first two awaits, with a spy for the weight load."""
    loads: list[str] = []

    async def load_model():
        loads.append("load_model")

    trainer.load_model = load_model

    async def run():
        await trainer.setup()
        await trainer.load_model()

    asyncio.run(run())
    return loads


@pytest.mark.parametrize(
    ("block", "config", "error", "match"),
    [
        (lambda sha: _block("0" * 64), {}, ValueError, "does not match the pinned"),
        (lambda sha: _block(sha, training_adapter=None, default_handling="contrastive_guidance"),
         {"distillation_handling": "adapter"}, ValueError, "distillation_handling='adapter' is not supported"),
        (lambda sha: _block(sha), {"distillation_handling": "both", "distillation_guidance_scale": 0.5},
         ValueError, "distillation_guidance_scale"),
    ],
    ids=["wrong-sha256", "handling-not-allowed", "guidance-below-1"],
)
def test_a_bad_adapter_or_handling_refuses_the_job_before_any_weight_loads(adapter, block, config, error, match):
    trainer = bare_trainer(stand_in_definition(distillation=block(adapter["sha"])), **config)
    loads: list[str] = []
    with pytest.raises(error, match=match):
        loads = _setup_then_load(trainer)
    assert loads == []
    assert trainer.components == {}


def test_a_missing_file_refuses_the_job_before_any_weight_loads(tmp_path, monkeypatch, adapter):
    _resolve_to(monkeypatch, tmp_path / "gone.safetensors")
    trainer = bare_trainer(stand_in_definition(distillation=_block(adapter["sha"])))
    with pytest.raises(FileNotFoundError):
        _setup_then_load(trainer)


def test_a_good_file_lets_the_job_go_on_to_load(adapter):
    trainer = bare_trainer(stand_in_definition(distillation=_block(adapter["sha"])))
    assert _setup_then_load(trainer) == ["load_model"]


def test_in_a_step_the_adapter_is_active_frozen_and_out_of_the_optimizer(adapter):
    from app.engine.core import training_adapter

    definition = stand_in_definition(distillation=_block(adapter["sha"]))
    trainer = _peft_trainer(definition, optimizer_type="AdamW", learning_rate=1e-2, targeted_layers=[r".*"])
    plain = _peft_trainer(definition, distillation_handling="none")
    model, plain_model = trainer._get_primary_model(), plain._get_primary_model()
    assert training_adapter.has_training_adapter(model) and not training_adapter.has_training_adapter(plain_model)
    # The same user LoRA in both (the base weights already are: one seed), so the only
    # difference is the adapter.
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, p in plain_model.named_parameters():
            if ".lora_" in name:
                p.copy_(params[name])
            else:  # a module the adapter wraps carries its weights under ``base_layer``
                head, tail = name.rsplit(".", 1)
                assert torch.equal(p, params.get(name, params.get(f"{head}.base_layer.{tail}"))), name

    # Targeted training matched every LoRA parameter; the adapter stays frozen anyway.
    trainer._configure_targeted_training()
    trainer._refreeze_training_adapter()
    trainer._configure_optimization(10)
    adapter_params = training_adapter.adapter_parameters(model)
    assert adapter_params and not any(p.requires_grad for _, p in adapter_params)
    in_optimizer = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert not any(id(p) in in_optimizer for _, p in adapter_params)
    assert all(".default." in n for n, p in model.named_parameters() if id(p) in in_optimizer)

    torch.manual_seed(0)
    step = trainer.forward_loss_step(_latents(2), _text(), {"captions": ["a", "b"]}, 1)
    torch.manual_seed(0)
    without = plain.forward_loss_step(_latents(2), _text(), {"captions": ["a", "b"]}, 1)
    assert torch.equal(step.target, without.target), "the two steps must see the same draw"
    assert not torch.allclose(step.pred, without.pred), "the adapter was not active in the step"

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    step.loss.backward()
    assert all(p.grad is None for _, p in adapter_params)
    trainer.optimizer.step()
    changed = {n for n, p in model.named_parameters() if not torch.equal(p.detach(), before[n])}
    assert changed and all(".default." in n for n in changed), sorted(n for n in changed if ".default." not in n)[:4]


def test_the_saved_lora_has_exactly_the_users_keys(adapter, tmp_path):
    from safetensors import safe_open

    from tests.engine.families.test_minimax_h3_image_trainer import _driver

    definition = stand_in_definition(distillation=_block(adapter["sha"]))
    with_adapter = _peft_trainer(definition)._get_primary_model()
    without = _peft_trainer(definition, distillation_handling="none")._get_primary_model()

    def keys(model, name):
        path = tmp_path / name
        _driver().get_saver().save({"unet": model, "config": {"save_precision": "fp32"}}, path, metadata={})
        with safe_open(str(path), framework="pt") as fh:
            return set(fh.keys())

    saved = keys(with_adapter, "with.safetensors")
    assert saved == keys(without, "without.safetensors")
    # The user's targets are attention only; the adapter's fc1/fc2 would show if it leaked.
    assert not any(".mlp." in k for k in saved)


def test_none_trains_bit_for_bit_as_a_definition_without_the_block(adapter):
    with_block = _peft_trainer(stand_in_definition(distillation=_block(adapter["sha"])), distillation_handling="none")
    without_block = _peft_trainer(stand_in_definition())
    assert with_block._training_adapter_path is None and adapter["seen"] == []
    without_block._get_primary_model().load_state_dict(with_block._get_primary_model().state_dict())
    results = []
    for trainer in (with_block, without_block):
        torch.manual_seed(0)
        results.append(trainer.forward_loss_step(_latents(2), _text(), {"captions": ["a", "b"]}, 1))
    assert torch.equal(results[0].loss, results[1].loss)
    assert torch.equal(results[0].pred, results[1].pred)
