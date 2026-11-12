"""The weights row charges what a quantized transformer keeps unquantized at its loaded precision.

``QuantizationFactory`` never quantizes a module the model declares in ``_keep_in_fp32_modules``,
and diffusers loads those modules in float32 whatever ``torch_dtype`` the loader asks for. On
Kandinsky 5 Pro the declared modules (the per-block modulation above all) hold 2.38 B of its
19.31 B parameters: charging the whole transformer at int8 left about 6.8 GB of a training run
unreported. The share is counted on a meta-device instance of the definition's own transformer
class, built from the ``transformer.*`` config in ``architecture_params``; it reaches the block-swap
credit too, so a swapped block takes its kept modules off the card at their loaded precision.
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from app.engine.utils.vram_estimator import _QUANT_BITS, VRAMEstimator

_FAMILIES = Path(__file__).resolve().parents[1] / "app/engine/models/families"
_DEFS = sorted(_FAMILIES.glob("*/definitions/*.yaml"))
MIB = 1024 * 1024
FP32 = 4  # bytes: diffusers loads every declared module in float32
K5_PRO = "kandinsky5/definitions/k5_i2v_pro_sft_5s.yaml"


def _definition(path: Path) -> SimpleNamespace:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SimpleNamespace(
        id=data["id"],
        family=data["family"],
        detected_precision=data.get("detected_precision") or {},
        architecture_params=data.get("architecture_params") or {},
        model_size_mb=data.get("model_size_mb") or {},
        block_topology=data.get("block_topology") or [],
    )


def _transformer_class(defn: SimpleNamespace) -> type | None:
    """The definition's transformer class: diffusers first, then the family's vendored package."""
    name = defn.architecture_params.get("transformer._class_name")
    if not name:
        return None
    import diffusers

    cls = getattr(diffusers, name, None)
    if cls is None:
        try:
            cls = getattr(importlib.import_module(f"app.engine.models.families.{defn.family}.vendor"), name, None)
        except ImportError:
            cls = None
    return cls


def _meta_instance(defn: SimpleNamespace, cls: type) -> torch.nn.Module:
    accepted = set(inspect.signature(cls.__init__).parameters)
    arch = defn.architecture_params
    config = {k[len("transformer."):]: v for k, v in arch.items() if k.startswith("transformer.")}
    with torch.device("meta"):
        return cls.from_config({k: v for k, v in config.items() if k in accepted})


def _kept(model: torch.nn.Module, prefix: str = "") -> int:
    """Parameters diffusers loads in float32: a declared name is a whole segment of the parameter's path.

    This is diffusers' own loading rule (``load_model_dict_into_meta``), written out here
    independently of the rule the estimator and the quantization factory share.
    """
    declared = list(type(model)._keep_in_fp32_modules or [])
    return sum(
        p.numel()
        for name, p in model.named_parameters()
        if name.startswith(prefix) and any(m in name.split(".") for m in declared)
    )


def _weights(defn: SimpleNamespace, **config) -> float:
    return VRAMEstimator.estimate(defn, config, check_fit=False).model_weights_mb


# ── Kandinsky 5 Pro, the reported case ─────────────────────────────────────


def test_kandinsky5_pro_at_int8_covers_its_parameter_inventory():
    defn = _definition(_FAMILIES / K5_PRO)
    cls = _transformer_class(defn)
    assert cls is not None and cls._keep_in_fp32_modules
    model = _meta_instance(defn, cls)
    total = sum(p.numel() for p in model.parameters())
    kept = _kept(model)
    assert kept == 2_381_539_328
    int8 = _QUANT_BITS["int8"] / 8
    inventory_mb = ((total - kept) * int8 + kept * FP32) / MIB  # 25,230.4 MiB
    # The stated on-disk size (36,833 MB) is the inventory rounded to whole MB, so the
    # estimate may sit that rounding (half a MB at int8) below the inventory, never more.
    assert _weights(defn, quantization="int8") >= inventory_mb - 1.0


def test_kandinsky5_pro_charges_the_kept_share_once_at_float32():
    defn = _definition(_FAMILIES / K5_PRO)
    model = _meta_instance(defn, _transformer_class(defn))
    kept = _kept(model)
    disk_mb = defn.model_size_mb["transformer"]
    want = (disk_mb - kept * 2 / MIB) * 0.5 + kept * FP32 / MIB
    assert _weights(defn, quantization="int8") == pytest.approx(want, abs=1e-6)


def test_a_swapped_kandinsky5_block_leaves_with_its_kept_modules_at_float32():
    defn = _definition(_FAMILIES / K5_PRO)
    model = _meta_instance(defn, _transformer_class(defn))
    group = defn.block_topology[0]
    assert group["name"] == "visual_transformer_blocks"
    per_block_kept = _kept(model, f"{group['attr_path']}.") / group["count"]
    assert per_block_kept == 37_785_600
    swapped = round(group["count"] * 50 / 100)
    on_card_block_mb = group["approx_vram_mb"] * 0.5 + per_block_kept * (FP32 - 1) / MIB
    unswapped = _weights(defn, quantization="int8")
    swapped_weights = _weights(defn, quantization="int8", block_swap_config={group["name"]: 50})
    assert unswapped - swapped_weights == pytest.approx(swapped * on_card_block_mb, abs=1e-6)


@pytest.mark.parametrize("scheme", ["none", "bf16"])
def test_an_unquantized_transformer_is_charged_as_before(scheme):
    defn = _definition(_FAMILIES / K5_PRO)
    plain = _definition(_FAMILIES / K5_PRO)
    plain.architecture_params = {
        k: v for k, v in plain.architecture_params.items() if k != "transformer._class_name"
    }
    assert _weights(defn, quantization=scheme) == _weights(plain, quantization=scheme)
    swap = {"visual_transformer_blocks": 50}
    assert _weights(defn, quantization=scheme, block_swap_config=swap) == _weights(
        plain, quantization=scheme, block_swap_config=swap
    )


# ── The class: every registered definition whose transformer declares kept modules ──

#: Definitions whose transformer class the test cannot build from the definition alone, with why.
#: Each is checked below to declare no kept module anywhere in its family's code.
_UNRESOLVED_REASONS = {
    "boogu-image-base": "vendored class, not exported from the family's vendor package",
    "boogu-image-edit": "vendored class, not exported from the family's vendor package",
    "boogu-image-turbo": "vendored class, not exported from the family's vendor package",
    "flux1-schnell": "no transformer._class_name (loader: FluxTransformer2DModel)",
    "flux2-klein-base-4b": "no transformer._class_name (loader: Flux2Transformer2DModel)",
    "hidream_o1_image": "no transformer._class_name (a unified vendored model)",
    "microsoft-lens-base": "vendored class, not exported from the family's vendor package",
    "omnigen2": "vendored class, not exported from the family's vendor package",
}


def _declaring() -> list[Path]:
    out = []
    for path in _DEFS:
        defn = _definition(path)
        cls = _transformer_class(defn)
        if cls is not None and getattr(cls, "_keep_in_fp32_modules", None):
            out.append(path)
    return out


def test_the_class_covers_every_family_that_declares_kept_modules():
    families = {_definition(p).family for p in _declaring()}
    assert {"wan21", "wan22", "wan22_ti2v_5b", "bernini_r", "kandinsky5", "krea2", "minimax_h3", "flux3", "minimax_h3_image"} <= families


@pytest.mark.parametrize("path", _declaring(), ids=lambda p: p.stem)
def test_the_estimators_kept_share_is_the_meta_inventory(path):
    """At int8 the weights row rises by exactly the kept share times (float32 - int8) bytes."""
    defn = _definition(path)
    kept = _kept(_meta_instance(defn, _transformer_class(defn)))
    assert kept > 0
    plain = _definition(path)
    plain.architecture_params = {
        k: v for k, v in plain.architecture_params.items() if k != "transformer._class_name"
    }
    precision = defn.detected_precision.get("transformer") or defn.detected_precision.get("unet")
    assert precision == "torch.bfloat16"
    int8 = _QUANT_BITS["int8"] / 8
    # Unquantized, nothing moves: the kept share is charged only when the transformer is quantized.
    assert _weights(defn, quantization="none") == _weights(plain, quantization="none")
    gained = _weights(defn, quantization="int8") - _weights(plain, quantization="int8")
    # A dual-expert run with both experts resident (the "auto" default) charges the share twice.
    resident = round(_weights(plain, quantization="none") / _weights(plain, quantization="none", expert_swap_mode="swap"))
    assert resident in (1, 2)
    assert gained == pytest.approx(resident * kept * (FP32 - int8) / MIB, rel=1e-9)


@pytest.mark.parametrize("def_id", sorted(_UNRESOLVED_REASONS))
def test_an_unresolved_definition_declares_no_kept_module(def_id):
    paths = [p for p in _DEFS if _definition(p).id == def_id]
    assert paths, f"{def_id} is listed but not registered"
    defn = _definition(paths[0])
    assert _transformer_class(defn) is None, f"{def_id} resolves now; drop it from the list"
    family_dir = _FAMILIES / defn.family
    for source in family_dir.rglob("*.py"):
        if "tests" in source.parts:
            continue
        assert not re.search(r"_keep_in_fp32_modules\s*=", source.read_text(encoding="utf-8", errors="ignore")), (
            f"{source} declares kept modules: {def_id} needs a resolvable class"
        )
    loader = family_dir / "loader.py"
    if loader.exists():
        import diffusers

        for name in re.findall(r"diffusers(?:\.models)?\.(\w+)", loader.read_text(encoding="utf-8")):
            assert not getattr(getattr(diffusers, name, None), "_keep_in_fp32_modules", None), (
                f"{def_id}'s loader names {name}, which declares kept modules"
            )


def test_every_registered_definition_is_either_covered_or_listed():
    covered = {_definition(p).id for p in _declaring()}
    for path in _DEFS:
        defn = _definition(path)
        if defn.id in covered or defn.id in _UNRESOLVED_REASONS:
            continue
        cls = _transformer_class(defn)
        assert cls is not None, f"{defn.id}: class unresolved and not listed"
        assert not getattr(cls, "_keep_in_fp32_modules", None)
