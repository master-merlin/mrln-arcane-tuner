"""A module the model declares float32 (``_keep_in_fp32_modules``) is never quantized.

diffusers loads the modules a model names in ``_keep_in_fp32_modules`` in float32 whatever dtype is
asked for, transformers does for ``_keep_in_fp32_modules_strict`` (and for the plain attribute on an
fp16 load), and both libraries' own quantizers leave them unconverted. ``QuantizationFactory``
quantized them like any other ``Linear``: on MiniMax-H3-Image that put the timestep MLP in int8, and
its first sampling steps read the wrong noise level.

The fix leaves every declared module, with all its descendants, as it was loaded: its type, dtype
and values. It does not upcast anything. Every other eligible ``Linear`` is quantized as before.

The fixtures need nothing outside the installed libraries: toy modules that declare the attributes
themselves, and a tiny ``WanTransformer3DModel`` from diffusers. A scheme whose backend needs CUDA
runs where CUDA is present and is skipped with its reason otherwise.
"""

from __future__ import annotations

import contextlib
import copy
import os

import pytest
import torch
import torch.nn as nn

from app.engine.factories import quantization as quantization_module
from app.engine.factories.quantization import QuantizationFactory
from app.engine.factories.quantizers import bitsandbytes as bnb_backend
from app.engine.factories.quantizers.bitsandbytes import BitsAndBytesBackend
from app.engine.factories.quantizers.quanto import QuantoBackend
from app.engine.factories.quantizers.torchao import TorchAOBackend

WIDTH = 256  # int4's group size (128) and nvfp4's block (16) both divide it


def fp32_module_names(model: nn.Module) -> frozenset[str]:
    """Looked up at call time, so a build without it fails each test rather than the collection."""
    return quantization_module.fp32_module_names(model)

# ── Fixtures ─────────────────────────────────────────────────────────────


class _Embedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(WIDTH, WIDTH)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(WIDTH, WIDTH)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(WIDTH, WIDTH, bias=False)
        self.norm2 = nn.LayerNorm(WIDTH)
        self.ff = nn.Linear(WIDTH, WIDTH)


class _Toy(nn.Module):
    """A mixed-precision model shaped like a diffusers transformer.

    ``pre_time_embedder`` holds a declared name as a substring only, so it is quantized: a name
    matches whole segments of a dotted path. ``norm2`` is declared and lives in every block, so it
    matches deeper in the tree. ``scale_shift_table`` is a bare parameter, as WAN declares one."""

    def __init__(self) -> None:
        super().__init__()
        self.time_embedder = _Embedder()
        self.pre_time_embedder = nn.Linear(WIDTH, WIDTH)
        self.blocks = nn.ModuleList([_Block(), _Block()])
        self.norm_out = nn.LayerNorm(WIDTH)
        self.proj_out = nn.Linear(WIDTH, WIDTH)
        self.scale_shift_table = nn.Parameter(torch.randn(2, WIDTH))


DECLARED = ["time_embedder", "norm2", "norm_out", "proj_out", "scale_shift_table"]
TOY_SKIP = {
    "time_embedder", "time_embedder.linear_1", "time_embedder.act", "time_embedder.linear_2",
    "blocks.0.norm2", "blocks.1.norm2", "norm_out", "proj_out",
}
TOY_TOP_LEVEL = sorted(["time_embedder", "blocks.0.norm2", "blocks.1.norm2", "norm_out", "proj_out"])
TOY_QUANTIZED = {"pre_time_embedder", "blocks.0.to_q", "blocks.0.ff", "blocks.1.to_q", "blocks.1.ff"}

#: How each declaration is spread over the two attributes. "both" splits the names and repeats one.
DECLARATIONS = {
    "plain": (DECLARED, None),
    "strict": (None, DECLARED),
    "both": (DECLARED[:3], DECLARED[2:]),
}


def _kept(name: str, declared: list[str]) -> bool:
    """The library's own rule: a declared name is a whole segment of the parameter's path."""
    return any(segment in declared for segment in name.split("."))


def _load_like_the_library(model: nn.Module, dtype: torch.dtype, kept_dtype: torch.dtype) -> nn.Module:
    """``model`` as diffusers or transformers hands it over: ``dtype`` everywhere, the declared
    parameters in ``kept_dtype``."""
    declared = _declared(model)
    model.to(dtype)
    for name, param in model.named_parameters():
        if _kept(name, declared):
            param.data = param.data.to(kept_dtype)
    return model.eval().requires_grad_(False)


def _toy(declaration: str = "plain", dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    torch.manual_seed(0)
    model = _Toy()
    plain, strict = DECLARATIONS[declaration]
    if plain is not None:
        model._keep_in_fp32_modules = list(plain)
    if strict is not None:
        model._keep_in_fp32_modules_strict = list(strict)
    # transformers keeps a plain declaration in float32 on an fp16 load only; every other case
    # (diffusers, the strict attribute) loads it in float32 under any dtype.
    kept_dtype = dtype if declaration == "plain" and dtype == torch.bfloat16 else torch.float32
    return _load_like_the_library(model, dtype, kept_dtype)


def _wan() -> nn.Module:
    """A tiny ``WanTransformer3DModel``, loaded in bf16 the way diffusers loads it."""
    from diffusers import WanTransformer3DModel

    torch.manual_seed(0)
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=64, in_channels=4,
        out_channels=4, text_dim=WIDTH, freq_dim=WIDTH, ffn_dim=WIDTH, num_layers=1,
        rope_max_seq_len=32,
    )
    return _load_like_the_library(model, torch.bfloat16, torch.float32)


def _wan_skip(model: nn.Module) -> set[str]:
    declared = list(model._keep_in_fp32_modules)
    return {name for name, _ in model.named_modules() if name and _kept(name, declared)}


def _wan_quantized(model: nn.Module, skip: set[str]) -> set[str]:
    """Every Linear outside the skip set that every scheme can take: both widths a multiple of
    int4's group size. (``proj_out``'s 16 outputs are not; a scheme may leave it as it is.)"""
    return {
        name for name, m in model.named_modules()
        if isinstance(m, nn.Linear) and name not in skip and m.in_features % 128 == 0 and m.out_features % 128 == 0
    }


# ── The skip set ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("declaration", list(DECLARATIONS))
def test_the_skip_set_is_every_declared_module_with_its_descendants(declaration, dtype):
    assert fp32_module_names(_toy(declaration, dtype)) == TOY_SKIP


def test_a_model_without_a_declaration_has_an_empty_skip_set():
    model = nn.Sequential(nn.Linear(4, 4), nn.SiLU(), nn.Linear(4, 4))
    assert fp32_module_names(model) == frozenset()
    model._keep_in_fp32_modules = None  # the libraries' own default
    model._keep_in_fp32_modules_strict = []
    assert fp32_module_names(model) == frozenset()


def test_a_name_matches_whole_segments_only():
    names = fp32_module_names(_toy())
    assert "pre_time_embedder" not in names
    assert not any(name.startswith("blocks.") and name.endswith((".to_q", ".ff")) for name in names)


def test_a_dotted_name_matches_its_segments_in_a_row():
    model = _toy()
    model._keep_in_fp32_modules = ["time_embedder.linear_2"]
    model._keep_in_fp32_modules_strict = {"to_q.weight"}  # transformers' own instances hold sets
    assert fp32_module_names(model) == {"time_embedder.linear_2", "blocks.0.to_q", "blocks.1.to_q"}


def test_wan_declares_its_time_embedder():
    model = _wan()
    names = fp32_module_names(model)
    assert names == _wan_skip(model)
    assert "condition_embedder.time_embedder" in names
    assert "condition_embedder.time_embedder.linear_1" in names
    assert "condition_embedder.time_embedder.linear_2" in names


# ── Each backend, each scheme ────────────────────────────────────────────

_NEEDS_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="this scheme quantizes on CUDA only")

#: (backend, scheme, device). A CPU scheme runs with the availability probe forced on (the probes
#: ask for CUDA); a CUDA scheme runs on the real probe and is skipped where the card lacks it.
SCHEMES = [
    pytest.param("torchao", "int8", "cpu", id="torchao-int8"),
    pytest.param("torchao", "int8_dynamic", "cpu", id="torchao-int8_dynamic"),
    pytest.param("torchao", "int4", "cuda", id="torchao-int4", marks=_NEEDS_CUDA),
    pytest.param("torchao", "fp8", "cpu", id="torchao-fp8"),
    pytest.param("torchao", "fp8_dynamic", "cuda", id="torchao-fp8_dynamic", marks=_NEEDS_CUDA),
    pytest.param("torchao", "nvfp4", "cpu", id="torchao-nvfp4"),
    pytest.param("optimum-quanto", "qint8", "cpu", id="quanto-qint8"),
    pytest.param("optimum-quanto", "qint4", "cpu", id="quanto-qint4"),
    pytest.param("optimum-quanto", "qfloat8", "cpu", id="quanto-qfloat8"),
    pytest.param("optimum-quanto", "qfloat8_e4m3fn", "cpu", id="quanto-qfloat8_e4m3fn"),
    pytest.param("optimum-quanto", "qfloat8_e5m2", "cpu", id="quanto-qfloat8_e5m2"),
    pytest.param("bitsandbytes", "nf4", "cuda", id="bitsandbytes-nf4", marks=_NEEDS_CUDA),
]

_BACKENDS = {"torchao": TorchAOBackend, "optimum-quanto": QuantoBackend, "bitsandbytes": BitsAndBytesBackend}


def _ready(backend_name: str, scheme: str, device: str, monkeypatch) -> None:
    if backend_name == "optimum-quanto":
        pytest.importorskip("optimum.quanto")
    backend = _BACKENDS[backend_name]
    if device == "cpu":
        monkeypatch.setattr(backend, "is_available", classmethod(lambda cls, s: True))
    elif not backend.is_available(scheme):
        pytest.skip(f"{backend_name} {scheme} is not available on this card")


def _plain_tensors(module: nn.Module) -> bool:
    return all(type(t) is torch.Tensor for t in module.state_dict().values()) and all(
        type(p) is nn.Parameter and type(p.data) is torch.Tensor for p in module.parameters(recurse=False)
    )


def _same_state(got: nn.Module, want: nn.Module) -> bool:
    a, b = got.state_dict(), want.state_dict()
    return a.keys() == b.keys() and all(
        a[k].dtype == b[k].dtype and torch.equal(a[k].detach().cpu(), b[k].detach().cpu()) for k in a
    )


def _declared(model: nn.Module) -> list[str]:
    return [*(getattr(model, "_keep_in_fp32_modules", None) or []),
            *(getattr(model, "_keep_in_fp32_modules_strict", None) or [])]


def _assert_kept_and_quantized(model, reference, skip, quantized):
    """Every module in ``skip`` and every declared parameter (a bare one such as
    ``scale_shift_table`` too) is as it was loaded, bit for bit; every module in ``quantized`` is
    quantized."""
    for name in sorted(skip):
        layer, original = model.get_submodule(name), reference.get_submodule(name)
        assert type(layer) is type(original), f"{name} became {type(layer).__name__}"
        assert _plain_tensors(layer), f"{name} holds a quantized tensor"
        assert _same_state(layer, original), f"{name} changed dtype or values"
    got, want = model.state_dict(), reference.state_dict()
    declared = [key for key in want if _kept(key, _declared(reference))]
    assert any("scale_shift_table" in key for key in declared)
    for key in declared:
        assert type(got[key]) is torch.Tensor, f"{key} is a {type(got[key]).__name__}"
        assert got[key].dtype == want[key].dtype, f"{key} changed dtype"
        assert torch.equal(got[key].detach().cpu(), want[key].detach().cpu()), f"{key} changed values"
    assert quantized, "the fixture holds no eligible Linear outside the skip set"
    for name in sorted(quantized):
        layer = model.get_submodule(name)
        assert not (type(layer) is nn.Linear and _plain_tensors(layer)), f"{name} was not quantized"


@pytest.mark.parametrize(("backend_name", "scheme", "device"), SCHEMES)
def test_every_backend_keeps_the_declared_modules_of_a_toy_model(backend_name, scheme, device, monkeypatch):
    _ready(backend_name, scheme, device, monkeypatch)
    model = _toy()
    reference = copy.deepcopy(model)
    QuantizationFactory.quantize(model, scheme, backend_name=backend_name, device=device)
    _assert_kept_and_quantized(model, reference, TOY_SKIP, TOY_QUANTIZED)


@pytest.mark.parametrize(("backend_name", "scheme", "device"), SCHEMES)
def test_every_backend_keeps_the_declared_modules_of_wan(backend_name, scheme, device, monkeypatch):
    _ready(backend_name, scheme, device, monkeypatch)
    model = _wan()
    reference = copy.deepcopy(model)
    skip = _wan_skip(model)
    QuantizationFactory.quantize(model, scheme, backend_name=backend_name, device=device)
    _assert_kept_and_quantized(model, reference, skip, _wan_quantized(reference, skip))


# ── Both attributes, both load dtypes ────────────────────────────────────

#: One scheme of each backend that quantizes on the CPU; bitsandbytes' swap runs with a stand-in.
_CPU_ONE_EACH = [("torchao", "int8"), ("optimum-quanto", "qint8"), ("bitsandbytes", "nf4")]


class _StandInLinear4bit(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, compute_dtype=None, quant_type=None):
        super().__init__(in_features, out_features, bias=bias)


@pytest.fixture
def stand_in_bnb(monkeypatch):
    """bitsandbytes' swap on the CPU: its ``Linear4bit`` replaced by a plain subclass of Linear."""
    monkeypatch.setattr(bnb_backend, "bnb", type("bnb", (), {"nn": type("nn", (), {"Linear4bit": _StandInLinear4bit})}))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("declaration", list(DECLARATIONS))
@pytest.mark.parametrize(("backend_name", "scheme"), _CPU_ONE_EACH)
def test_either_attribute_and_either_dtype_give_the_same_exclusion(
    backend_name, scheme, declaration, dtype, monkeypatch, stand_in_bnb
):
    _ready(backend_name, scheme, "cpu", monkeypatch)
    model = _toy(declaration, dtype)
    reference = copy.deepcopy(model)
    QuantizationFactory.quantize(model, scheme, backend_name=backend_name, device="cpu")
    _assert_kept_and_quantized(model, reference, TOY_SKIP, TOY_QUANTIZED)
    if backend_name == "bitsandbytes":
        assert all(type(model.get_submodule(n)) is _StandInLinear4bit for n in TOY_QUANTIZED)


# ── The log line ─────────────────────────────────────────────────────────


@contextlib.contextmanager
def _captured_logs():
    """Every structlog event of the quantization module (as test_quantization.py captures them)."""
    from structlog.testing import capture_logs

    quantization_module.logger.__dict__.pop("bind", None)
    try:
        with capture_logs() as logs:
            yield logs
    finally:
        quantization_module.logger.__dict__.pop("bind", None)


def _kept_lines(logs) -> list[dict]:
    return [e for e in logs if e.get("event") == "quantization_kept_fp32_modules"]


def test_the_log_line_names_the_kept_top_level_modules_once(monkeypatch):
    _ready("torchao", "int8", "cpu", monkeypatch)
    with _captured_logs() as logs:
        QuantizationFactory.quantize(_toy(), "int8", backend_name="torchao", device="cpu")
    lines = _kept_lines(logs)
    assert len(lines) == 1
    assert lines[0]["modules"] == TOY_TOP_LEVEL
    assert lines[0]["scheme"] == "int8"


def test_a_model_without_a_declaration_logs_no_kept_line(monkeypatch):
    _ready("torchao", "int8", "cpu", monkeypatch)
    model = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.SiLU(), nn.Linear(WIDTH, WIDTH)).requires_grad_(False)
    with _captured_logs() as logs:
        QuantizationFactory.quantize(model, "int8", backend_name="torchao", device="cpu")
    assert _kept_lines(logs) == []
    assert not _plain_tensors(model[0]) and not _plain_tensors(model[2])


# ── The disk cache ───────────────────────────────────────────────────────


def _source_count() -> int:
    return sum(p.numel() for p in _toy().parameters())


@pytest.mark.parametrize("scheme", ["int8", "int8_dynamic"])
def test_an_entry_that_quantized_a_declared_module_is_stale_and_a_fresh_one_hits(scheme, tmp_path, monkeypatch):
    _ready("torchao", scheme, "cpu", monkeypatch)
    path = str(tmp_path / "entry")

    # The build before the fix: the backend quantizes every Linear, the declared ones too.
    old = TorchAOBackend.quantize(_toy(), scheme, device="cpu")
    assert not _plain_tensors(old.get_submodule("time_embedder.linear_2"))
    QuantizationFactory.save_quantized(old, path, scheme, source_param_count=_source_count())
    with _captured_logs() as logs:
        assert QuantizationFactory.load_quantized(_toy(), path, scheme) is None
    stale = [e for e in logs if e.get("event") == "quantized_cache_stale_representation"]
    assert len(stale) == 1

    # The rebuild: the fixed quantize writes an entry that loads, with the declared modules as loaded.
    fresh = QuantizationFactory.quantize(_toy(), scheme, backend_name="torchao", device="cpu")
    os.remove(os.path.join(path, "metadata.json"))
    QuantizationFactory.save_quantized(fresh, path, scheme, source_param_count=_source_count())
    with _captured_logs() as logs:
        loaded = QuantizationFactory.load_quantized(_toy(), path, scheme)
    assert loaded is not None
    assert not [e for e in logs if e.get("event") == "quantized_cache_stale_representation"]
    _assert_kept_and_quantized(loaded, _toy(), TOY_SKIP, TOY_QUANTIZED)


def test_an_entry_with_a_key_the_plain_module_lacks_is_stale():
    entry = _toy().state_dict()
    entry["time_embedder.linear_2.weight._scale"] = torch.ones(WIDTH)  # a packing backend's key
    found = QuantizationFactory._quantized_fp32_module(_toy(), entry)
    assert found is not None and found.startswith("time_embedder.linear_2.weight._scale")
    assert QuantizationFactory._quantized_fp32_module(_toy(), _toy().state_dict()) is None
