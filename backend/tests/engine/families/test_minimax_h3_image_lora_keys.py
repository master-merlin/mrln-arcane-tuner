"""The MiniMax-H3-Image LoRA file: native-layout keys, an independent interop proof, and
the trigger metadata.

The file the saver writes is the model's native layout, the one ComfyUI's repack loads:
``diffusion_model.blocks.N.attn.qkv_proj`` (q/k/v fused), ``attn.out_proj``, ``mlp.fc1``
(SwiGLU halves in the checkpoint's ``[gate; value]`` order), ``mlp.fc2``, with PEFT's
scaling folded into ``lora_B`` and no ``.alpha`` key. The tensor work is the shared core
(``minimax_h3_shared.lora_artifact``); the module map is this family's own.

What this file holds it to:

* **0 unmatched against the real checkpoint.** ``fixtures/minimax_h3_image_hf_configs/
  transformer_index_keys.json`` holds the key NAMES of both safetensors indexes of
  ``MiniMaxAI/MiniMax-H3-Image`` at ``fbde899f``: the native ``transformer/`` folder and the
  diffusers ``transformer_diffusers/`` folder (sha256s of the source files inside). Every
  LoRA-able Linear of every block in the diffusers index maps onto a module the native index
  holds, and every native block Linear is reached.
* **Interop through an independent converter.** A tiny PEFT model with non-zero random A and B
  is saved through the driver's own saver, and the file goes through diffusers' upstream
  ``_convert_non_diffusers_minimax_h3_lora_to_diffusers`` (installed diffusers 0.41.0). For
  every adapted module, the converted ``B @ A`` equals the PEFT layer's own
  ``scaling * B @ A`` to 1e-6 in fp32. No inverse of ours is used anywhere.
* **The trigger word** lands in both ``ss_training_comment`` and ``modelspec.trigger_phrase``.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest
import torch

from tests.engine.families.test_minimax_h3_image_vendor import _tiny_transformer

_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "minimax_h3_image_hf_configs" / "transformer_index_keys.json"
_FAMILY_DIR = Path(__file__).resolve().parents[3] / "app" / "engine" / "models" / "families" / "minimax_h3_image"

# The LoRA-able Linears of a diffusers block (the refiner blocks carry no AdaLN projection).
_BLOCK_LINEARS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2", "adaln_proj.linear")
# The default targets of the family (attention q/k/v/out and the FFN), plus the AdaLN projection.
_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2", "adaln_proj.linear"]
_BLOCK_KEY = re.compile(
    r"^(?P<stack>transformer_blocks|token_refiner\.refiner_blocks)\.\d+\.(?P<rest>.+)\.weight$"
)
_NATIVE_BLOCK_LINEAR = re.compile(
    r"^(?:token_refiner\.)?blocks\.\d+\.(?:attn\.qkv_proj|attn\.out_proj|mlp\.fc1|mlp\.fc2|adaln_proj\.linear)\.weight$"
)


@pytest.fixture(scope="module")
def index_keys() -> dict:
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert data["repo"] == "MiniMaxAI/MiniMax-H3-Image" and data["revision"] == "fbde899f"
    assert len(data["native"]) == 531, "fixture drifted: native key count"
    assert len(data["diffusers"]) == 644, "fixture drifted: diffusers key count"
    return data


def _diffusers_block_linears(keys: list[str]) -> list[str]:
    modules = []
    for key in keys:
        match = _BLOCK_KEY.match(key)
        if match and match.group("rest") in _BLOCK_LINEARS:
            modules.append(key.removesuffix(".weight"))
    return sorted(modules)


# ── The key map against the real checkpoint ─────────────────────────────────


def test_every_block_linear_maps_onto_a_native_module(index_keys):
    from app.engine.models.families.minimax_h3_image.lora_keys import native_module_for

    native = set(index_keys["native"])
    modules = _diffusers_block_linears(index_keys["diffusers"])
    # 50 blocks x 7 Linears + 2 refiner blocks x 6 (no AdaLN on a refiner block).
    assert len(modules) == 50 * 7 + 2 * 6
    unmatched = []
    reached: set[str] = set()
    for module in modules:
        target, _ = native_module_for(module)
        if f"{target}.weight" not in native:
            unmatched.append((module, target))
        reached.add(f"{target}.weight")
    assert not unmatched, f"{len(unmatched)} modules map to no native key: {unmatched[:4]}"
    native_linears = {k for k in native if _NATIVE_BLOCK_LINEAR.match(k)}
    assert reached == native_linears, (
        f"never reached={sorted(native_linears - reached)[:4]} invented={sorted(reached - native_linears)[:4]}"
    )


def test_the_transform_of_each_module():
    from app.engine.models.families.minimax_h3_image.lora_keys import native_module_for

    assert native_module_for("transformer_blocks.7.attn.to_q") == ("blocks.7.attn.qkv_proj", "q")
    assert native_module_for("transformer_blocks.7.attn.to_k") == ("blocks.7.attn.qkv_proj", "k")
    assert native_module_for("transformer_blocks.7.attn.to_v") == ("blocks.7.attn.qkv_proj", "v")
    assert native_module_for("transformer_blocks.49.attn.to_out.0") == ("blocks.49.attn.out_proj", "rename")
    assert native_module_for("transformer_blocks.0.ff.net.0.proj") == ("blocks.0.mlp.fc1", "fc1")
    assert native_module_for("transformer_blocks.0.ff.net.2") == ("blocks.0.mlp.fc2", "rename")
    assert native_module_for("transformer_blocks.3.adaln_proj.linear") == ("blocks.3.adaln_proj.linear", "rename")
    assert native_module_for("token_refiner.refiner_blocks.1.attn.to_v") == ("token_refiner.blocks.1.attn.qkv_proj", "v")
    assert native_module_for("token_refiner.refiner_blocks.0.ff.net.2") == ("token_refiner.blocks.0.mlp.fc2", "rename")


@pytest.mark.parametrize(
    "module",
    [
        "pdd_proj_out.0",  # the per-step heads live outside the native transformer file
        "context_embedder",
        "transformer_blocks.0.attn.mystery",
        "token_refiner.refiner_blocks.0.adaln_proj.linear",  # a refiner block has no AdaLN
    ],
)
def test_a_module_no_rule_covers_raises(module):
    from app.engine.models.families.minimax_h3_image.lora_keys import native_module_for

    with pytest.raises(KeyError, match=re.escape(module)):
        native_module_for(module)


def test_a_full_size_artifact_names_only_native_modules(index_keys):
    """Every adapter of a full-size target list (all 52 blocks) through the real core:
    each artifact key names a module the native index holds, with 0 unmatched."""
    from app.engine.models.families.minimax_h3_image.lora_keys import Adapter, build_artifact

    g = torch.Generator().manual_seed(0)
    adapters = {
        module: Adapter(lora_A=torch.randn(1, 4, generator=g), lora_B=torch.randn(4, 1, generator=g), scaling=1.0)
        for module in _diffusers_block_linears(index_keys["diffusers"])
    }
    artifact = build_artifact(adapters)
    native = set(index_keys["native"])
    modules = {re.sub(r"\.lora_[AB]\.weight$", "", k) for k in artifact}
    assert all(k.startswith("diffusion_model.") for k in artifact)
    assert not any(k.endswith(".alpha") for k in artifact)
    unmatched = sorted(m for m in modules if f"{m.removeprefix('diffusion_model.')}.weight" not in native)
    assert not unmatched, f"{len(unmatched)} artifact modules are not in the native index: {unmatched[:4]}"
    # q/k/v fuse into one qkv_proj per block: 50 x 5 + 2 x 4 native modules.
    assert len(modules) == 50 * 5 + 2 * 4


# ── Interop: diffusers' own converter reads the file back ───────────────────


def _peft_model(*, r: int = 4, alpha: float = 8.0):
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(_tiny_transformer(), LoraConfig(r=r, lora_alpha=alpha, target_modules=list(_TARGETS)))
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            if ".lora_A." in name or ".lora_B." in name:  # PEFT zero-inits B: a zero delta proves nothing
                p.copy_(0.1 * torch.randn(p.shape, generator=g, dtype=torch.float32))
    return model.eval()


def _peft_deltas(model) -> dict[str, torch.Tensor]:
    """``scaling * B @ A`` read straight off each PEFT layer, keyed by diffusers module path."""
    deltas = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and isinstance(getattr(module, "scaling", None), dict) and "default" in module.scaling:
            a = module.lora_A["default"].weight.detach().float()
            b = module.lora_B["default"].weight.detach().float()
            deltas[name.removeprefix("base_model.model.")] = float(module.scaling["default"]) * (b @ a)
    return deltas


def _saved(tmp_path, model, config: dict | None = None):
    from safetensors import safe_open

    from tests.engine.families.test_minimax_h3_image_trainer import _driver

    path = tmp_path / "h3_image_lora.safetensors"
    _driver().get_saver().save({"unet": model, "config": {"save_precision": "fp32", **(config or {})}}, path, metadata={})
    with safe_open(str(path), framework="pt") as fh:
        tensors = {k: fh.get_tensor(k) for k in fh.keys()}
        header = dict(fh.metadata() or {})
    return tensors, header


def test_the_upstream_converter_reproduces_every_delta(tmp_path):
    from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_minimax_h3_lora_to_diffusers

    model = _peft_model()
    deltas = _peft_deltas(model)
    # 2 blocks x 7 targets + 2 refiner blocks x 6.
    assert len(deltas) == 2 * 7 + 2 * 6
    for module, delta in deltas.items():
        assert float(delta.abs().sum()) > 0, f"zero delta on {module}: the proof would be vacuous"
    tensors, _ = _saved(tmp_path, model)
    assert all(k.startswith(("diffusion_model.blocks.", "diffusion_model.token_refiner.blocks.")) for k in tensors)
    assert not any(k.endswith(".alpha") for k in tensors)

    converted = _convert_non_diffusers_minimax_h3_lora_to_diffusers(dict(tensors))
    expected = {f"transformer.{m}.lora_{ab}.weight" for m in deltas for ab in ("A", "B")}
    assert set(converted) == expected, (
        f"missing={sorted(expected - set(converted))[:4]} extra={sorted(set(converted) - expected)[:4]}"
    )
    for module, delta in deltas.items():
        a = converted[f"transformer.{module}.lora_A.weight"].float()
        b = converted[f"transformer.{module}.lora_B.weight"].float()
        torch.testing.assert_close(b @ a, delta, atol=1e-6, rtol=0, msg=lambda m, module=module: f"{module}: {m}")


def test_a_wrong_scale_would_be_caught(tmp_path):
    """The proof's sensitivity: with alpha/r = 2 the folded file's delta is twice the unscaled one."""
    model = _peft_model(r=4, alpha=8.0)
    deltas = _peft_deltas(model)
    tensors, _ = _saved(tmp_path, model)
    module = "transformer_blocks.0.attn.to_out.0"
    layer = model.get_submodule(f"base_model.model.{module}")
    assert float(layer.scaling["default"]) == 2.0
    a = tensors["diffusion_model.blocks.0.attn.out_proj.lora_A.weight"]
    b = tensors["diffusion_model.blocks.0.attn.out_proj.lora_B.weight"]
    unscaled = layer.lora_B["default"].weight.detach().float() @ layer.lora_A["default"].weight.detach().float()
    assert not torch.allclose(unscaled, deltas[module], atol=1e-6)
    torch.testing.assert_close(b @ a, deltas[module], atol=1e-6, rtol=0)


# ── The saver ────────────────────────────────────────────────────────────────


def test_the_trigger_word_lands_in_both_fields(tmp_path):
    from app.engine.utils.lora_metadata import TRIGGER_COMMENT_KEY, TRIGGER_PHRASE_KEY

    _, header = _saved(tmp_path, _peft_model(), {"global_triggerword": "  fizgig  "})
    assert header[TRIGGER_COMMENT_KEY] == "fizgig"
    assert header[TRIGGER_PHRASE_KEY] == "fizgig"
    assert header["modelspec.architecture"] == "minimax-h3-image"
    assert header["minimax_h3_image.layout_version"] == "1"


def test_no_trigger_word_writes_neither_field(tmp_path):
    from app.engine.utils.lora_metadata import TRIGGER_COMMENT_KEY, TRIGGER_PHRASE_KEY

    _, header = _saved(tmp_path, _peft_model())
    assert TRIGGER_COMMENT_KEY not in header and TRIGGER_PHRASE_KEY not in header


def test_the_save_precision_is_honoured(tmp_path):
    tensors, _ = _saved(tmp_path, _peft_model(), {"save_precision": "bf16"})
    assert {t.dtype for t in tensors.values()} == {torch.bfloat16}


def test_a_model_without_adapters_is_refused(tmp_path):
    from tests.engine.families.test_minimax_h3_image_trainer import _driver

    saver = _driver().get_saver()
    with pytest.raises(ValueError, match="not a PEFT model"):
        saver.save({"unet": _tiny_transformer(), "config": {}}, tmp_path / "x.safetensors")
    with pytest.raises(ValueError, match="no 'unet' / 'transformer'"):
        saver.save({"config": {}}, tmp_path / "x.safetensors")
    assert not (tmp_path / "x.safetensors").exists()


# ── The family's own map over the shared core ───────────────────────────────


def test_the_family_uses_the_shared_core_and_nothing_of_the_video_family():
    from app.engine.models.families.minimax_h3_image import lora_keys
    from app.engine.models.families.minimax_h3_shared import lora_artifact

    assert lora_keys.Adapter is lora_artifact.Adapter
    assert lora_keys.ARTIFACT_PREFIX == lora_artifact.ARTIFACT_PREFIX == "diffusion_model."
    for name in ("lora_keys.py", "saver.py"):
        src = (_FAMILY_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"families\.minimax_h3(\.| import)", src), f"{name} imports the video family"
        assert "get_peft_model_state_dict" not in src and "torch.zeros" not in src, f"{name} copies the shared core"
    assert "minimax_h3_shared" in inspect.getsource(lora_keys)
