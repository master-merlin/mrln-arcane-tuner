"""Every quantization scheme the training form offers trains a LoRA on its frozen base.

torchao's int4 (tinygemm) and dynamic-FP8 kernels have no derivative, so backward through a frozen
base layer quantized with them raised. A LoRA's base layer is frozen: its backward needs only the
grad of its input, ``grad_out @ W``. Those layers now run forward on the quantized kernel and take
that grad from the dequantized weight. Every other layer is left exactly as it was.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytestmark = [
    pytest.mark.xdist_group("gpu"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the quantized kernels run on a card"),
]

from app.engine.factories.quantization import QuantizationFactory  # noqa: E402
from app.engine.factories.quantizers import trainable_backward  # noqa: E402

WIDTH = 256
WRAPPED = [("torchao", "int4"), ("torchao", "fp8_dynamic"), ("torchao", "int8_dynamic")]
UNTOUCHED = [
    ("torchao", "int8"), ("torchao", "fp8"),
    ("optimum-quanto", "qint8"), ("optimum-quanto", "qint4"), ("bitsandbytes", "nf4"),
]


def offered() -> list[tuple[str, str]]:
    if not torch.cuda.is_available():
        return []
    return [(b, s) for b, schemes in QuantizationFactory.get_supported_capabilities().items() for s in schemes]


def _offered_only(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    have = set(offered())
    return [p for p in pairs if p in have]


class Net(nn.Module):
    def __init__(self, width: int = WIDTH, layers: int = 2, bias: bool = True) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(width, width, bias=bias, dtype=torch.bfloat16) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.gelu(layer(x))
        return x


def frozen_net(**kwargs) -> nn.Module:
    torch.manual_seed(0)
    return Net(**kwargs).to("cuda").requires_grad_(False)


def quantized(backend: str, scheme: str, **kwargs) -> nn.Module:
    return QuantizationFactory.quantize(frozen_net(**kwargs), scheme, backend_name=backend)


def with_lora(net: nn.Module) -> nn.Module:
    from peft import LoraConfig, get_peft_model

    targets = [f"layers.{i}" for i in range(len(net.layers))]
    model = get_peft_model(net, LoraConfig(r=8, target_modules=targets))
    gen = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=gen).to(param.dtype) * 0.01)
    return model


def batch(width: int = WIDTH, seed: int = 1, requires_grad: bool = True) -> torch.Tensor:
    gen = torch.Generator("cuda").manual_seed(seed)
    return torch.randn(8, width, device="cuda", dtype=torch.bfloat16, generator=gen).requires_grad_(requires_grad)


def train_steps(model: nn.Module, steps: int = 3) -> tuple[bool, torch.Tensor]:
    return train_steps_at(model, WIDTH, steps)


def train_steps_at(model: nn.Module, width: int, steps: int = 3) -> tuple[bool, torch.Tensor]:
    """``steps`` AdamW steps; whether the LoRA params changed, and the input's grad at the last."""
    params = [p for p in model.parameters() if p.requires_grad]
    assert params, "nothing to train"
    before = [p.detach().clone() for p in params]
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    x = batch(width)
    for _ in range(steps):
        x.grad = None
        model(x).float().pow(2).mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    changed = any(not torch.equal(a, b) for a, b in zip(before, params, strict=True))
    return changed, x.grad


def first_linear(module: nn.Module) -> nn.Linear:
    return next(m for m in module.modules() if isinstance(m, nn.Linear) and not hasattr(m, "lora_A"))


# ── Task 1: a frozen quantized Linear without a derivative gets its backward from its dense weight ──


@pytest.mark.parametrize(("backend", "scheme"), _offered_only(WRAPPED), ids=lambda v: str(v))
def test_a_scheme_whose_kernel_has_no_derivative_trains_a_lora(backend, scheme):
    changed, grad = train_steps(with_lora(quantized(backend, scheme)))
    assert changed, "the LoRA params never changed"
    assert grad is not None and torch.isfinite(grad).all()


@pytest.mark.parametrize(("backend", "scheme"), offered(), ids=lambda v: str(v))
def test_the_base_layer_alone_passes_the_grad_to_its_input(backend, scheme):
    # No LoRA: a grad the input gets can only come through the quantized base layer. int8_dynamic's
    # kernel dropped it without a word, and a LoRA run still "trained" on the adapters' own branches.
    reference = frozen_net(layers=1)
    q = quantized(backend, scheme, layers=1)
    g = torch.randn(8, WIDTH, device="cuda", dtype=torch.bfloat16, generator=torch.Generator("cuda").manual_seed(5))
    xr, xq = batch(), batch()
    reference.layers[0](xr).backward(g)
    out = q.layers[0](xq)
    assert out.requires_grad, f"{backend}/{scheme}: the base layer's output tracks no grad"
    out.backward(g)
    error = ((xq.grad.float() - xr.grad.float()).norm() / xr.grad.float().norm()).item()
    assert error <= 0.15, f"{backend}/{scheme}: input grad {error:.4f} relative to the unquantized layer's"


@pytest.mark.parametrize(("backend", "scheme"), _offered_only(WRAPPED), ids=lambda v: str(v))
def test_the_input_grad_is_the_output_grad_times_the_dense_weight(backend, scheme):
    lin = quantized(backend, scheme, layers=1).layers[0]
    assert trainable_backward.is_wrapped(lin)
    x = batch()
    g = torch.randn(8, WIDTH, device="cuda", dtype=torch.bfloat16, generator=torch.Generator("cuda").manual_seed(5))
    lin(x).backward(g)
    dense = trainable_backward.dense_weight(lin.weight, torch.bfloat16)
    assert torch.equal(x.grad, g @ dense)
    if scheme == "int4":
        # Weight-only: the dense weight is the very weight the kernel multiplies by. The kernel adds
        # the bias after its product, and sums in another order than cuBLAS: equal to that order's
        # bf16 rounding (2 ulps of the output's magnitude), not bit for bit at every width.
        want, got = F.linear(x.detach(), dense) + lin.bias, lin(x.detach())
        assert (got.float() - want.float()).abs().max() <= 2.0**-6 * want.float().abs().max()


@pytest.mark.parametrize(("backend", "scheme"), _offered_only(UNTOUCHED), ids=lambda v: str(v))
def test_a_scheme_that_trains_already_is_left_bitwise_as_it_was(backend, scheme, monkeypatch):
    def outputs() -> tuple[torch.Tensor, torch.Tensor]:
        net = quantized(backend, scheme, layers=1)
        assert not any(trainable_backward.is_wrapped(m) for m in net.modules())
        x = batch()
        out = net(x)
        out.float().pow(2).mean().backward()
        return out.detach(), x.grad

    with_code = outputs()
    monkeypatch.setattr(trainable_backward, "make_quantized_linears_trainable", lambda module: 0)
    without = outputs()
    assert torch.equal(with_code[0], without[0]) and torch.equal(with_code[1], without[1])


def test_a_wrapped_layer_whose_weight_requires_grad_is_refused():
    lin = quantized("torchao", "int4", layers=1).layers[0]
    lin.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="frozen"):
        lin(batch())


@pytest.mark.parametrize(("backend", "scheme"), _offered_only(WRAPPED), ids=lambda v: str(v))
def test_a_model_from_the_quantized_cache_trains_a_lora(backend, scheme, tmp_path):
    source = frozen_net()
    signature = QuantizationFactory.state_shape_signature(source)
    count = sum(p.numel() for p in source.parameters())
    q = QuantizationFactory.quantize(source, scheme, backend_name=backend)
    path = os.path.join(tmp_path, "cache")
    QuantizationFactory.save_quantized(q, path, scheme, source_param_count=count, source_signature=signature)
    loaded = QuantizationFactory.load_quantized(frozen_net(), path, scheme)
    assert loaded is not None, "the cache did not hit"
    assert all(trainable_backward.is_wrapped(layer) for layer in loaded.layers)
    changed, _ = train_steps(with_lora(loaded))
    assert changed


@pytest.mark.parametrize(("backend", "scheme"), _offered_only(WRAPPED), ids=lambda v: str(v))
def test_the_dense_weight_lives_only_inside_backward(backend, scheme):
    lin = quantized(backend, scheme, layers=1, width=1024).layers[0]
    x = batch(1024)
    out = lin(x)
    for saved in out.grad_fn.saved_tensors:
        is_dense = type(saved) is torch.Tensor and saved.shape == lin.weight.shape and saved.is_floating_point()
        assert not is_dense, "the node keeps a dense copy of the weight"
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    out.float().pow(2).mean().backward()
    del out
    torch.cuda.synchronize()
    grew = torch.cuda.memory_allocated() - before
    assert grew <= x.numel() * x.element_size(), f"backward left {grew} bytes behind (the input grad is {x.numel() * 2})"


# ── Task 3: every offered scheme, fresh and from the cache, for training and for a text encoder ──


@pytest.mark.parametrize(("backend", "scheme"), offered(), ids=lambda v: str(v))
def test_every_offered_scheme_trains_fresh_and_from_its_cache(backend, scheme, tmp_path):
    source = frozen_net()
    signature = QuantizationFactory.state_shape_signature(source)
    count = sum(p.numel() for p in source.parameters())
    fresh = QuantizationFactory.quantize(source, scheme, backend_name=backend)

    with torch.no_grad():  # the text-encoder use: a forward, no grad anywhere
        out = fresh(batch(requires_grad=False))
    assert torch.isfinite(out).all() and not out.requires_grad

    path = os.path.join(tmp_path, "cache")
    QuantizationFactory.save_quantized(fresh, path, scheme, source_param_count=count, source_signature=signature)
    written = os.path.exists(os.path.join(path, "metadata.json"))
    # quanto and bitsandbytes change the state-dict layout: their save is skipped by design.
    assert written == (backend == "torchao"), f"{backend}/{scheme}: cache written={written}"

    wrapped_fresh = [trainable_backward.is_wrapped(m) for m in fresh.layers]  # before LoRA replaces them
    changed, grad = train_steps(with_lora(fresh))
    assert changed and grad is not None and torch.isfinite(grad).all()
    if written:
        loaded = QuantizationFactory.load_quantized(frozen_net(), path, scheme)
        assert loaded is not None, f"{backend}/{scheme}: its own fresh cache did not hit"
        assert [trainable_backward.is_wrapped(m) for m in loaded.layers] == wrapped_fresh
        changed, grad = train_steps(with_lora(loaded))
        assert changed and grad is not None and torch.isfinite(grad).all()


# ── Task 2: nvfp4 is NVFP4, and a cached entry of another representation is never loaded ──


@pytest.mark.skipif(("torchao", "nvfp4") not in offered(), reason="nvfp4 is not offered on this card")
def test_nvfp4_quantizes_to_nvfp4_without_a_fallback(caplog):
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    lin = quantized("torchao", "nvfp4", layers=1).layers[0]
    assert isinstance(lin.weight, NVFP4Tensor), type(lin.weight).__name__
    assert "nvfp4_not_yet_implemented_in_torchao" not in caplog.text


def _cache_of(scheme_made: str, labeled: str, path: str) -> None:
    source = frozen_net()
    signature = QuantizationFactory.state_shape_signature(source)
    count = sum(p.numel() for p in source.parameters())
    q = QuantizationFactory.quantize(source, scheme_made, backend_name="torchao")
    QuantizationFactory.save_quantized(q, path, labeled, source_param_count=count, source_signature=signature)
    assert os.path.exists(os.path.join(path, "metadata.json")), f"{scheme_made}: no cache was written"


@pytest.mark.parametrize(
    ("made", "labeled"), [("int4", "nvfp4"), ("int8", "int4")], ids=["int4-labeled-nvfp4", "int8-labeled-int4"],
)
def test_a_cached_entry_of_another_representation_is_never_loaded(made, labeled, tmp_path, caplog):
    if ("torchao", labeled) not in offered() or ("torchao", made) not in offered():
        pytest.skip(f"{made} or {labeled} is not offered on this card")
    path = os.path.join(tmp_path, "cache")
    _cache_of(made, labeled, path)
    fresh = frozen_net()
    before = {k: v.clone() for k, v in fresh.state_dict().items()}
    assert QuantizationFactory.load_quantized(fresh, path, labeled) is None
    assert "quantized_cache_stale_representation" in caplog.text
    after = fresh.state_dict()
    assert all(type(after[k]) is torch.Tensor and torch.equal(after[k], v) for k, v in before.items())
    # The caller then quantizes fresh, as on any miss, and that trains.
    changed, _ = train_steps(with_lora(QuantizationFactory.quantize(fresh, labeled, backend_name="torchao")))
    assert changed


def test_a_current_cached_entry_still_hits(tmp_path):
    path = os.path.join(tmp_path, "cache")
    _cache_of("int8", "int8", path)
    assert QuantizationFactory.load_quantized(frozen_net(), path, "int8") is not None


# ── Review round 1: the precisions, pipeline order and memory production uses ──

# What layer_manifest.resolve_precision yields: fp32 weights (SDXL) honour the configured autocast;
# bf16 weights make autocast follow the model, so they never run under fp16.
PRECISIONS = [
    (torch.float32, None), (torch.float32, torch.bfloat16), (torch.float32, torch.float16),
    (torch.bfloat16, None), (torch.bfloat16, torch.bfloat16),
]


def _precision_id(p) -> str:
    dtype, autocast = p
    return f"{str(dtype)[6:]}-{'none' if autocast is None else str(autocast)[6:]}"


def amp(autocast: torch.dtype | None):
    import contextlib

    return torch.autocast("cuda", dtype=autocast) if autocast is not None else contextlib.nullcontext()


@pytest.mark.parametrize("precision", PRECISIONS, ids=_precision_id)
@pytest.mark.parametrize(("backend", "scheme"), offered(), ids=lambda v: str(v))
def test_every_offered_scheme_trains_at_every_production_precision(backend, scheme, precision):
    dtype, autocast = precision

    def layer() -> nn.Module:
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(WIDTH, WIDTH, dtype=dtype)).to("cuda").requires_grad_(False)

    def through(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator("cuda").manual_seed(4)
        x = torch.randn(8, WIDTH, device="cuda", dtype=dtype, generator=gen).requires_grad_(True)
        g = torch.randn(8, WIDTH, device="cuda", dtype=dtype, generator=gen)
        with amp(autocast):
            out = module(x)
        out.backward(g.to(out.dtype))
        return out.detach().float(), x.grad.float()

    want_out, want_grad = through(layer())
    q = QuantizationFactory.quantize(layer(), scheme, backend_name=backend)
    got_out, got_grad = through(q)
    for what, got, want in (("output", got_out, want_out), ("input grad", got_grad, want_grad)):
        error = ((got - want).norm() / want.norm()).item()
        assert error <= 0.15, f"{backend}/{scheme} {_precision_id(precision)}: {what} {error:.4f} off the unquantized"

    model = with_lora(QuantizationFactory.quantize(frozen_net().to(dtype), scheme, backend_name=backend))
    params = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in params]
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    x = batch().to(dtype).detach().requires_grad_(True)
    with amp(autocast):
        loss = model(x).float().pow(2).mean()
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, params, strict=True))
    assert torch.isfinite(x.grad).all()


from app.engine.core.pipeline.pipeline_optimization import PipelineOptimizationMixin  # noqa: E402


class _FusionHost(PipelineOptimizationMixin):
    """What prepare_for_training's first steps touch, with the real fusion and the real factory."""

    def __init__(self, model: nn.Module, scheme: str, cache: str | None = None) -> None:
        import structlog

        from app.engine.models.families.flux2.driver import Flux2Driver

        self.driver = object.__new__(Flux2Driver)  # should_fuse_qkv_projections() is True
        self.logger = structlog.get_logger("test")
        self.model, self.scheme, self.cache = model, scheme, cache

    def _get_primary_model(self) -> nn.Module:
        return self.model

    def _freeze_all(self) -> None:
        self.model.requires_grad_(False)

    def _quantize_components(self) -> None:  # _quantize_primary_model's cache-then-quantize, in short
        if self.cache and os.path.exists(os.path.join(self.cache, "metadata.json")):
            loaded = QuantizationFactory.load_quantized(self.model, self.cache, self.scheme)
            assert loaded is not None, "the fused model's cache did not hit"
            self.model = loaded
            return
        count = sum(p.numel() for p in self.model.parameters())
        signature = QuantizationFactory.state_shape_signature(self.model)
        self.model = QuantizationFactory.quantize(self.model, self.scheme, backend_name="torchao")
        if self.cache:
            QuantizationFactory.save_quantized(
                self.model, self.cache, self.scheme, source_param_count=count, source_signature=signature
            )


def _flux2_attention() -> nn.Module:
    from diffusers.models.transformers.transformer_flux2 import Flux2Attention

    torch.manual_seed(0)
    attention = Flux2Attention(query_dim=WIDTH, heads=2, dim_head=WIDTH // 2, added_kv_proj_dim=WIDTH)
    return nn.ModuleDict({"attn": attention}).to("cuda", torch.bfloat16)


@pytest.mark.parametrize("scheme", [s for b, s in offered() if b == "torchao"])
def test_flux2_fuses_its_qkv_before_quantizing_and_trains_fresh_and_cached(scheme, tmp_path):
    from peft import LoraConfig, get_peft_model

    cache = os.path.join(tmp_path, "cache")
    for _run in ("fresh", "cached"):
        host = _FusionHost(_flux2_attention(), scheme, cache)
        host._fuse_freeze_and_quantize()
        attn = host.model["attn"]
        assert attn.fused_projections and not any(p.requires_grad for p in host.model.parameters())
        model = get_peft_model(host.model, LoraConfig(r=4, target_modules=["to_qkv"]))
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    param.normal_(std=0.01)
        params = [p for p in model.parameters() if p.requires_grad]
        before = [p.detach().clone() for p in params]
        x = batch()
        attn.to_qkv(x).float().pow(2).mean().backward()
        torch.optim.AdamW(params, lr=1e-2).step()
        assert any(not torch.equal(a, b) for a, b in zip(before, params, strict=True)), _run
        assert x.grad is not None and torch.isfinite(x.grad).all(), _run


def test_the_int4_backward_peak_is_bounded_by_the_dense_weight():
    # Wan-like rectangular FFN shape, scaled down: the identity a whole-matrix rebuild needs is
    # in_features², far above the in x out dense weight itself.
    fin, fout = 8192, 1024
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(fin, fout, bias=False, dtype=torch.bfloat16)).to("cuda").requires_grad_(False)
    lin = QuantizationFactory.quantize(net, "int4", backend_name="torchao")[0]
    x = torch.randn(2, fin, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = lin(x)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out.float().pow(2).mean().backward()
    torch.cuda.synchronize()
    dense = fin * fout * 2
    grew = torch.cuda.max_memory_allocated() - base
    assert grew <= 1.5 * dense, f"backward peaked {grew / 2**20:.1f} MiB over a {dense / 2**20:.1f} MiB dense weight"


# ── LANE-187: any input layout ──

# A bf16 Linear(128, 256)'s input as a view of a contiguous source (shape seen: (2, 64, 128)):
# name -> (source shape, the view, the strides the layer must see).
LAYOUT_CASES = {
    "contiguous": ((2, 64, 128), lambda t: t, (8192, 128, 1)),
    "transposed": ((2, 128, 64), lambda t: t.transpose(1, 2), (8192, 1, 64)),
    "expanded": ((1, 64, 128), lambda t: t.expand(2, 64, 128), (0, 128, 1)),
    "sliced": ((2, 64, 256), lambda t: t[..., :128], (16384, 256, 1)),
    # LANE-189: FLUX.2's text tokens, a batch-1 split along the token dim. is_contiguous() is True
    # (a size-1 dim's stride is ignored), but the strides are not canonical: matmul's fold rejects them.
    "batch1_split": ((1, 96, 128), lambda t: t.split_with_sizes([64, 32], dim=1)[0], (12288, 128, 1)),
}


def _canonical(shape) -> tuple[int, ...]:
    strides, step = [], 1
    for size in reversed(shape):
        strides.append(step)
        step *= size
    return tuple(reversed(strides))


def _canonical_copy(x: torch.Tensor) -> torch.Tensor:
    """A copy with canonical strides: ``.contiguous()`` keeps a non-canonical size-1-dim stride."""
    y = x.clone(memory_format=torch.contiguous_format)
    assert y.stride() == _canonical(y.shape), f"not canonical: {y.stride()}"
    return y


def _layout_run(lin: nn.Module, layout: str, autocast, contiguous_copy: bool = False):
    """Output and source grad of ``lin`` on the layout's view (or on its canonical contiguous copy)."""
    source_shape, view, strides = LAYOUT_CASES[layout]
    gen = torch.Generator("cuda").manual_seed(11)
    leaf = torch.randn(source_shape, device="cuda", dtype=torch.bfloat16, generator=gen).requires_grad_(True)
    x = view(leaf)
    if contiguous_copy:
        x = _canonical_copy(x)
    else:
        assert x.stride() == strides, f"{layout}: the layer would see strides {x.stride()}"
        assert layout != "batch1_split" or x.is_contiguous(), "batch1_split must look contiguous"
    g = torch.randn(*x.shape[:-1], 256, device="cuda", dtype=torch.bfloat16, generator=gen)
    with amp(autocast):
        out = lin(x)
    out.backward(g.to(out.dtype))
    return out.detach().float(), leaf.grad.float()


def _close(got: torch.Tensor, ref: torch.Tensor) -> bool:
    if not ref.any():
        return not got.any()
    return ((got - ref).norm() <= 0.01 * ref.norm()).item()


def _linear_128(backend: str, scheme: str, bias: bool = True) -> nn.Module:
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(128, 256, bias=bias, dtype=torch.bfloat16)).to("cuda").requires_grad_(False)
    return QuantizationFactory.quantize(net, scheme, backend_name=backend)[0]


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("autocast", [None, torch.bfloat16], ids=["no-autocast", "bf16-autocast"])
@pytest.mark.parametrize("layout", sorted(LAYOUT_CASES))
@pytest.mark.parametrize(("backend", "scheme"), offered(), ids=lambda v: str(v))
def test_every_offered_scheme_trains_on_every_input_layout(backend, scheme, layout, autocast, bias):
    lin = _linear_128(backend, scheme, bias)
    got_out, got_grad = _layout_run(lin, layout, autocast)
    ref_out, ref_grad = _layout_run(lin, layout, autocast, contiguous_copy=True)
    if trainable_backward.makes_input_contiguous(lin):  # nvfp4, quanto: the very same computation
        assert torch.equal(got_out, ref_out) and torch.equal(got_grad, ref_grad)
    else:
        assert _close(got_out, ref_out) and _close(got_grad, ref_grad), f"{backend}/{scheme} {layout}"


_CONTIGUOUS_INPUT_SCHEMES = [p for p in offered() if p[1] == "nvfp4" or p[0] == "optimum-quanto"]


@pytest.mark.parametrize("shape", [(128,), (8, 128), (2, 8, 128), (2, 3, 4, 128)], ids=lambda s: f"rank{len(s)}")
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize(("backend", "scheme"), _CONTIGUOUS_INPUT_SCHEMES, ids=lambda v: str(v))
def test_a_contiguous_input_of_any_rank_computes_what_it_did_without_the_wrap(backend, scheme, bias, shape, monkeypatch):
    def run(lin):
        gen = torch.Generator("cuda").manual_seed(14)
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen).requires_grad_(True)
        out = lin(x)
        out.float().pow(2).mean().backward()
        return out.detach(), x.grad

    with_wrap = run(_linear_128(backend, scheme, bias))
    monkeypatch.setattr(trainable_backward, "make_quantized_linears_trainable", lambda module: 0)
    without = run(_linear_128(backend, scheme, bias))
    assert with_wrap[0].shape == (*shape[:-1], 256)
    assert torch.equal(with_wrap[0], without[0]) and torch.equal(with_wrap[1], without[1])


def test_nvfp4_from_the_cache_trains_on_a_non_contiguous_input(tmp_path):
    if ("torchao", "nvfp4") not in offered():
        pytest.skip("nvfp4 is not offered on this card")
    path = os.path.join(tmp_path, "cache")
    _cache_of("nvfp4", "nvfp4", path)
    loaded = QuantizationFactory.load_quantized(frozen_net(), path, "nvfp4")
    assert loaded is not None
    gen = torch.Generator("cuda").manual_seed(12)
    leaf = torch.randn(2, WIDTH, 16, device="cuda", dtype=torch.bfloat16, generator=gen).requires_grad_(True)
    loaded(leaf.transpose(1, 2)).float().pow(2).mean().backward()
    assert leaf.grad is not None and torch.isfinite(leaf.grad).all()


@pytest.mark.parametrize(
    ("backend", "scheme"), [p for p in offered() if p[1] == "nvfp4" or p[0] == "optimum-quanto"], ids=lambda v: str(v),
)
def test_a_layer_fed_a_contiguous_input_computes_what_it_did_without_the_wrap(backend, scheme, monkeypatch):
    with_wrap = _layout_run(_linear_128(backend, scheme), "contiguous", None)
    monkeypatch.setattr(trainable_backward, "make_quantized_linears_trainable", lambda module: 0)
    without = _layout_run(_linear_128(backend, scheme), "contiguous", None)
    assert torch.equal(with_wrap[0], without[0]) and torch.equal(with_wrap[1], without[1])


@pytest.mark.parametrize("autocast", [None, torch.bfloat16], ids=["no-autocast", "bf16-autocast"])
@pytest.mark.parametrize(("backend", "scheme"), offered(), ids=lambda v: str(v))  # every wrap, every scheme
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
def test_a_wrapped_layer_takes_its_input_by_keyword_as_a_linear_does(backend, scheme, bias, autocast):
    lin = _linear_128(backend, scheme, bias)
    tb = trainable_backward
    if not (tb.is_wrapped(lin) or tb.makes_input_contiguous(lin) or tb.in_weight_dtype(lin)):
        pytest.skip(f"no wrap of ours on {type(lin).__name__}: its forward is the library's own")

    def run(by_keyword: bool):
        gen = torch.Generator("cuda").manual_seed(13)
        leaf = torch.randn(2, 128, 64, device="cuda", dtype=torch.bfloat16, generator=gen).requires_grad_(True)
        x = leaf.transpose(1, 2)
        with amp(autocast):
            out = lin(input=x) if by_keyword else lin(x)  # nn.Linear.forward(self, input)
        out.float().pow(2).mean().backward()
        return out.detach(), leaf.grad

    by_keyword, positional = run(True), run(False)
    assert torch.equal(by_keyword[0], positional[0]) and torch.equal(by_keyword[1], positional[1])


@pytest.mark.parametrize(("backend", "scheme"), offered(), ids=lambda v: str(v))
def test_only_nvfp4_and_quanto_layers_make_their_input_contiguous(backend, scheme):
    lin = _linear_128(backend, scheme)
    expected = scheme == "nvfp4" or backend == "optimum-quanto"
    assert trainable_backward.needs_contiguous_input(lin) == expected
    assert trainable_backward.makes_input_contiguous(lin) == expected
    if expected:
        assert not trainable_backward.is_wrapped(lin)  # they keep their own backward


# ── Task 4's draws: layers a scheme cannot quantize ──


ODD = 200  # no int4 group (128) and no NVFP4 block (16) divides it


@pytest.mark.parametrize("scheme", ["int4", "nvfp4"])
def test_a_layer_the_scheme_cannot_quantize_stays_unquantized_and_trains(scheme):
    if ("torchao", scheme) not in offered():
        pytest.skip(f"{scheme} is not offered on this card")
    net = QuantizationFactory.quantize(frozen_net(width=ODD, layers=1), scheme, backend_name="torchao")
    assert type(net.layers[0].weight) is nn.Parameter  # left as it was, not raised on
    changed, _ = train_steps_at(with_lora(net), ODD)
    assert changed


def test_a_cached_int4_entry_with_no_layer_int4_could_quantize_still_hits(tmp_path):
    if ("torchao", "int4") not in offered():
        pytest.skip("int4 is not offered on this card")
    source = frozen_net(width=ODD)
    signature = QuantizationFactory.state_shape_signature(source)
    q = QuantizationFactory.quantize(source, "int4", backend_name="torchao")
    path = os.path.join(tmp_path, "cache")
    QuantizationFactory.save_quantized(q, path, "int4", source_param_count=2 * (ODD * ODD + ODD), source_signature=signature)
    loaded = QuantizationFactory.load_quantized(frozen_net(width=ODD).to("cpu"), path, "int4")
    assert loaded is not None
    # Where quantize() would have put it: a plain-tensor (safetensors) entry loaded onto the CPU.
    assert all(p.device.type == "cuda" for p in loaded.parameters())
    changed, _ = train_steps_at(with_lora(loaded), ODD)
    assert changed


# ── LANE-186: quanto layers under autocast ──

QUANTO = [p for p in offered() if p[0] == "optimum-quanto"]
# kind -> (the layer, its input shape)
QUANTO_KINDS = {
    "linear": (lambda bias, dtype: nn.Linear(64, 64, bias=bias, dtype=dtype), (2, 16, 64)),
    "conv": (lambda bias, dtype: nn.Conv2d(8, 8, 3, padding=1, bias=bias, dtype=dtype), (2, 8, 16, 16)),
}


def _plain(kind: str, bias: bool, dtype: torch.dtype) -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(QUANTO_KINDS[kind][0](bias, dtype)).to("cuda").requires_grad_(False)


def _quanto(scheme: str, kind: str, bias: bool, dtype: torch.dtype) -> nn.Module:
    return QuantizationFactory.quantize(_plain(kind, bias, dtype), scheme, backend_name="optimum-quanto")


def _through(net: nn.Module, kind: str, dtype: torch.dtype, autocast, *, outside_autocast: bool = False):
    """Output and input grad of ``net``; with ``outside_autocast``, the recipe the wrap must follow:
    autocast off, the input in the weight's dtype, the output cast to the autocast dtype.

    cuDNN runs deterministic algorithms here: an fp32 conv otherwise differs between two runs of
    the very same layer, and the tests compare runs with ``torch.equal``."""
    gen = torch.Generator("cuda").manual_seed(21)
    x = torch.randn(QUANTO_KINDS[kind][1], device="cuda", dtype=dtype, generator=gen).requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True):
        if outside_autocast:
            with torch.autocast("cuda", enabled=False):
                out = net(x.to(net[0].weight.dtype)).to(autocast)
        else:
            with amp(autocast):
                out = net(x)
        g = torch.randn(out.shape, device="cuda", dtype=out.dtype, generator=gen)
        out.backward(g)
    return out.detach(), x.grad


def _bypassed(scheme, kind, bias, dtype, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(trainable_backward, "make_quantized_linears_trainable", lambda module: 0)
        return _quanto(scheme, kind, bias, dtype)


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("autocast", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("kind", sorted(QUANTO_KINDS))
@pytest.mark.parametrize(("backend", "scheme"), QUANTO, ids=lambda v: str(v))
def test_every_quanto_layer_with_fp32_weights_trains_under_autocast(backend, scheme, kind, bias, autocast):
    net = _quanto(scheme, kind, bias, torch.float32)
    got_out, got_grad = _through(net, kind, torch.float32, autocast)
    want_out, want_grad = _through(_plain(kind, bias, torch.float32), kind, torch.float32, autocast)
    assert got_out.dtype == want_out.dtype == autocast, "the output dtype a plain layer gives under autocast"
    assert got_grad is not None and torch.isfinite(got_grad).all()
    for what, a, b in (("output", got_out, want_out), ("input grad", got_grad, want_grad)):
        assert _relative(a, b) <= 0.15, f"{scheme} {kind} {what} vs the unquantized layer"


@pytest.mark.parametrize("autocast", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("kind", sorted(QUANTO_KINDS))
@pytest.mark.parametrize(("backend", "scheme"), QUANTO, ids=lambda v: str(v))
def test_a_lora_on_every_quanto_layer_with_fp32_weights_trains_under_autocast(backend, scheme, kind, bias, autocast):
    from peft import LoraConfig, get_peft_model

    net = _quanto(scheme, kind, bias, torch.float32)
    model = get_peft_model(net, LoraConfig(r=4, target_modules=["0"]))  # as the product applies it
    trainable_backward.plain_conv_adapters(model)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 0.01)
    params = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in params]
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    x = torch.randn(QUANTO_KINDS[kind][1], device="cuda")
    with amp(autocast):
        loss = model(x).float().pow(2).mean()
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, params, strict=True)), "the LoRA never changed"


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("kind", sorted(QUANTO_KINDS))
@pytest.mark.parametrize(("backend", "scheme"), QUANTO, ids=lambda v: str(v))
def test_an_fp32_quanto_layer_under_autocast_runs_outside_it_in_its_weight_dtype(backend, scheme, kind, bias, monkeypatch):
    for autocast in (torch.float16, torch.bfloat16):
        got = _through(_quanto(scheme, kind, bias, torch.float32), kind, torch.float32, autocast)
        want = _through(_bypassed(scheme, kind, bias, torch.float32, monkeypatch), kind, torch.float32, autocast,
                        outside_autocast=True)
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1]), f"{scheme} {kind} {autocast}"


# (weight dtype, autocast): every production case the wrap must leave as it was
UNCHANGED_PRECISIONS = [(torch.float32, None), (torch.bfloat16, None), (torch.bfloat16, torch.bfloat16)]


@pytest.mark.parametrize(("dtype", "autocast"), UNCHANGED_PRECISIONS, ids=["fp32", "bf16", "bf16-under-bf16"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("kind", sorted(QUANTO_KINDS))
@pytest.mark.parametrize(("backend", "scheme"), QUANTO, ids=lambda v: str(v))
def test_a_quanto_layer_otherwise_computes_what_it_did(backend, scheme, kind, bias, dtype, autocast, monkeypatch):
    got = _through(_quanto(scheme, kind, bias, dtype), kind, dtype, autocast)
    want = _through(_bypassed(scheme, kind, bias, dtype, monkeypatch), kind, dtype, autocast)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


@pytest.mark.parametrize(("backend", "scheme"), offered(), ids=lambda v: str(v))
def test_only_quanto_layers_run_in_their_weight_dtype(backend, scheme):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(64, 64, dtype=torch.bfloat16), nn.Conv2d(8, 8, 3, dtype=torch.bfloat16))
    net = QuantizationFactory.quantize(net.to("cuda").requires_grad_(False), scheme, backend_name=backend)
    for layer in net:
        quanto = backend == "optimum-quanto"
        assert trainable_backward.runs_in_weight_dtype(layer) == quanto, f"{type(layer).__name__}"
        assert trainable_backward.in_weight_dtype(layer) == quanto, f"{type(layer).__name__}"


class _ConvAndLinear(nn.Module):
    """A conv then a Linear, under the names SDXL's LoRA targets use."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(8, 64, 3, padding=1)  # widths every quanto scheme quantizes (qint4 groups)
        self.to_q = nn.Linear(64, 64, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_q(self.conv1(x).permute(0, 2, 3, 1))


def _adapters(model: nn.Module):
    """(name, module) of every LoRA adapter layer under ``model``."""
    return [(n, m) for n, m in model.named_modules() if n.rsplit(".", 2)[-2:-1] in (["lora_A"], ["lora_B"])]


def _snapshot_quanto_conv_adapters(model: nn.Module) -> dict:
    from optimum.quanto.nn import QConv2d

    return {
        name: (m.weight.detach().clone(), None if m.bias is None else m.bias.detach().clone(), m.weight.data_ptr(),
               m.weight.requires_grad)
        for name, m in _adapters(model) if isinstance(m, QConv2d)
    }


def _assert_replaced_as_they_were(model: nn.Module, snaps: dict) -> None:
    assert snaps, "no quanto conv adapter to replace: the fixture does not reach the defect"
    for name, (weight, bias, ptr, requires_grad) in snaps.items():
        new = model.get_submodule(name)
        assert type(new) is nn.Conv2d, f"{name} is {type(new).__name__}"
        assert torch.equal(new.weight, weight), f"{name}: not the values peft initialised"
        assert new.weight.data_ptr() != ptr, f"{name}: shares the old adapter's storage"
        assert (new.bias is None) == (bias is None) and (bias is None or torch.equal(new.bias, bias))
        assert new.weight.requires_grad == requires_grad


@pytest.mark.parametrize(("backend", "scheme"), QUANTO, ids=lambda v: str(v))
def test_quanto_conv_adapters_become_plain_convs_holding_peft_s_values(backend, scheme):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    source = _ConvAndLinear().to("cuda").requires_grad_(False)
    net = QuantizationFactory.quantize(source, scheme, backend_name=backend)
    model = get_peft_model(net, LoraConfig(r=4, target_modules=["conv1", "to_q"]))
    snaps = _snapshot_quanto_conv_adapters(model)
    before = {name: model.get_submodule(name) for name in snaps}
    trainable_backward.plain_conv_adapters(model)
    _assert_replaced_as_they_were(model, snaps)
    for name, old in before.items():
        new = model.get_submodule(name)
        for attr in ("in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups",
                     "padding_mode"):
            assert getattr(new, attr) == getattr(old, attr), f"{name}.{attr}"
        assert new.weight.device == old.weight.device and new.weight.dtype == old.weight.dtype


class _PeftHost(PipelineOptimizationMixin):
    """What ``_apply_peft`` touches, with the real method."""

    def __init__(self, model: nn.Module) -> None:
        import structlog

        self.logger = structlog.get_logger("test")
        self.model = model
        self.config = {"network_rank": 4, "network_alpha": 4}

    def _get_primary_model(self) -> nn.Module:
        return self.model

    def _update_primary_model(self, model: nn.Module) -> None:
        self.model = model

    def get_lora_targets(self) -> list[str]:
        return ["conv1", "to_q"]

    def get_lora_exclude_modules(self):
        return None


@pytest.mark.parametrize(("backend", "scheme"), QUANTO, ids=lambda v: str(v))
def test_the_product_s_lora_on_a_quanto_model_has_plain_adapters_and_trains(backend, scheme, monkeypatch):
    from peft import get_peft_model_state_dict

    real, calls = trainable_backward.plain_conv_adapters, []

    def spy(model):
        snaps = _snapshot_quanto_conv_adapters(model)
        replaced = real(model)
        _assert_replaced_as_they_were(model, snaps)
        calls.append(len(snaps))
        return replaced

    monkeypatch.setattr(trainable_backward, "plain_conv_adapters", spy)
    torch.manual_seed(0)
    host = _PeftHost(QuantizationFactory.quantize(_ConvAndLinear().to("cuda").requires_grad_(False), scheme,
                                                  backend_name=backend))
    host._apply_peft()
    assert calls and calls[0] > 0, "_apply_peft never replaced its quanto conv adapters"
    model = host.model
    adapters = _adapters(model)
    assert adapters and all(type(m) in (nn.Conv2d, nn.Linear) for _, m in adapters), [type(m) for _, m in adapters]
    keys = get_peft_model_state_dict(model).keys()
    assert keys and all(("lora_A" in k or "lora_B" in k) and k.endswith(".weight") for k in keys), sorted(keys)

    for autocast in (None, torch.float16):
        params = [p for p in model.parameters() if p.requires_grad]
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "lora_B" in name:
                    p.normal_(0, 0.01)
        before = [p.detach().clone() for p in params]
        optimizer = torch.optim.AdamW(params, lr=1e-2)
        for _ in range(2):
            with amp(autocast):
                loss = model(torch.randn(2, 8, 8, 8, device="cuda")).float().pow(2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        assert any(not torch.equal(a, b) for a, b in zip(before, params, strict=True)), f"autocast={autocast}"


@pytest.mark.parametrize(("backend", "scheme"), QUANTO, ids=lambda v: str(v))
def test_quanto_still_writes_no_cache_entry(backend, scheme, tmp_path):
    """The wrap is applied after quantize only: quanto models are never loaded from the cache.
    If quanto caching is added, this turns red, and the wrap must then be applied on load too."""
    torch.manual_seed(0)
    source = nn.Sequential(nn.Linear(64, 64), nn.Conv2d(8, 8, 3)).to("cuda").requires_grad_(False)
    signature = QuantizationFactory.state_shape_signature(source)
    count = sum(p.numel() for p in source.parameters())
    q = QuantizationFactory.quantize(source, scheme, backend_name=backend)
    path = os.path.join(tmp_path, "cache")
    QuantizationFactory.save_quantized(q, path, scheme, source_param_count=count, source_signature=signature)
    assert not os.path.exists(os.path.join(path, "metadata.json"))
