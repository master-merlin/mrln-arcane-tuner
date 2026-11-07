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
        # the bias after its product (a fused addmm would round once, not twice).
        assert torch.equal(F.linear(x.detach(), dense) + lin.bias, lin(x.detach()))


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
