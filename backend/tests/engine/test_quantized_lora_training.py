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
    """``steps`` AdamW steps; whether the LoRA params changed, and the input's grad at the last."""
    params = [p for p in model.parameters() if p.requires_grad]
    assert params, "nothing to train"
    before = [p.detach().clone() for p in params]
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    x = batch()
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


def _nvfp4_until_task_2(pairs: list[tuple[str, str]]) -> list:
    nvfp4 = pytest.mark.xfail(strict=True, reason="nvfp4 still falls back to int4 that needs mslk")
    return [pytest.param(*p, marks=nvfp4) if p == ("torchao", "nvfp4") else p for p in pairs]


@pytest.mark.parametrize(("backend", "scheme"), _nvfp4_until_task_2(offered()), ids=lambda v: str(v))
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
