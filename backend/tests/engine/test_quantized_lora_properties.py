"""One property test per invariant of the quantized-LoRA fix, 50 seeded draws each.

A draw: a scheme the form offers on the card, a chain of one to three bf16 Linear layers of widths
that include ones no group or block size divides (such a layer may stay unquantized, but nothing may
raise), bias on or off, 2D or 3D inputs, and a LoRA on some of the layers only.

- I1: the quantized net trains its LoRA, fresh and, for a scheme whose save writes one, from the
  cache; each base layer alone passes its input a grad within 0.15 of the unquantized layer's.
- I2: a wrapped layer's input grad is exactly ``grad_out @ dense(W)`` (and, weight-only, its forward
  is ``linear(x, dense(W))``); a layer that is not wrapped gives bitwise what it gave with no wrapper
  code at all.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

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

DRAWS = 50
WIDTHS = [48, 96, 128, 200, 256, 384]


def offered() -> list[tuple[str, str]]:
    if not torch.cuda.is_available():
        return []
    return [(b, s) for b, schemes in QuantizationFactory.get_supported_capabilities().items() for s in schemes]


@dataclass
class Draw:
    backend: str
    scheme: str
    widths: list[int]  # widths[i] -> widths[i + 1] is layer i
    bias: bool
    tokens: int | None  # None: a 2D input; else the middle dim of a 3D one
    lora_on: list[int]
    seed: int
    dtype: torch.dtype = torch.bfloat16  # the loading dtype
    autocast: torch.dtype | None = None

    def __str__(self) -> str:
        shape = "2D" if self.tokens is None else f"3D(T={self.tokens})"
        amp = "none" if self.autocast is None else str(self.autocast)[6:]
        return (f"seed={self.seed} {self.backend}/{self.scheme} widths={self.widths} bias={self.bias} "
                f"{shape} lora_on={self.lora_on} dtype={str(self.dtype)[6:]} autocast={amp}")


# layer_manifest.resolve_precision: fp32 weights (SDXL) honour the configured autocast; bf16 weights
# make autocast follow the model, so they never run under fp16.
PRECISIONS = [
    (torch.float32, None), (torch.float32, torch.bfloat16), (torch.float32, torch.float16),
    (torch.bfloat16, None), (torch.bfloat16, torch.bfloat16),
]


def draw(seed: int) -> Draw:
    rng = random.Random(seed)
    backend, scheme = rng.choice(offered())
    layers = rng.randint(1, 3)
    widths = [rng.choice(WIDTHS) for _ in range(layers + 1)]
    lora_on = sorted(rng.sample(range(layers), rng.randint(1, layers)))
    dtype, autocast = rng.choice(PRECISIONS)
    return Draw(backend, scheme, widths, rng.random() < 0.5, rng.choice([None, 3]), lora_on, seed, dtype, autocast)


def amp(d: Draw):
    import contextlib

    return torch.autocast("cuda", dtype=d.autocast) if d.autocast is not None else contextlib.nullcontext()


def known_quanto_autocast_gap(d: Draw) -> bool:
    """LANE-186, pre-existing: quanto fails on a bias-less fp32 Linear under autocast.

    Such a draw asserts the failure instead, so it turns red the day LANE-186 fixes it, and its
    exclusion is removed then.
    """
    return d.backend == "optimum-quanto" and d.dtype == torch.float32 and d.autocast is not None and not d.bias


def assert_the_known_gap(d: Draw) -> None:
    net = quantized(d)
    x = rand(d, d.widths[0], salt=0, requires_grad=True)
    with pytest.raises(RuntimeError, match="same dtype"):  # in backward: the forward runs
        with amp(d):
            out = net(x)
        out.float().sum().backward()


def close_up_to_accumulation_order(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal but for the order a kernel sums in: within 2 bf16 ulps of the larger magnitude."""
    return (a.float() - b.float()).abs().max().item() <= 2.0**-6 * b.float().abs().max().item()


class Chain(nn.Module):
    def __init__(self, d: Draw) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(a, b, bias=d.bias, dtype=d.dtype) for a, b in zip(d.widths, d.widths[1:], strict=False)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.gelu(layer(x))
        return x


def frozen(d: Draw) -> nn.Module:
    torch.manual_seed(d.seed)
    return Chain(d).to("cuda").requires_grad_(False)


def quantized(d: Draw) -> nn.Module:
    return QuantizationFactory.quantize(frozen(d), d.scheme, backend_name=d.backend)


def rand(d: Draw, width: int, salt: int, requires_grad: bool = False) -> torch.Tensor:
    gen = torch.Generator("cuda").manual_seed(d.seed * 1000 + salt)
    shape = (4, width) if d.tokens is None else (4, d.tokens, width)
    return torch.randn(shape, device="cuda", dtype=d.dtype, generator=gen).requires_grad_(requires_grad)


def through(d: Draw, layer: nn.Module, x: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The layer's output under the draw's autocast, and its input's grad for the output grad ``g``."""
    x = x.detach().clone().requires_grad_(True)
    with amp(d):
        out = layer(x)
    assert out.requires_grad, "the base layer's output tracks no grad"
    out.backward(g.to(out.dtype))
    return out.detach(), x.grad


def trains(d: Draw, net: nn.Module) -> None:
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(net, LoraConfig(r=4, target_modules=[f"layers.{i}" for i in d.lora_on]))
    gen = torch.Generator().manual_seed(d.seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=gen).to(param.dtype) * 0.01)
    params = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in params]
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    x = rand(d, d.widths[0], salt=7, requires_grad=True)
    for _ in range(2):
        x.grad = None
        with amp(d):
            loss = model(x).float().pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    assert any(not torch.equal(a, b) for a, b in zip(before, params, strict=True)), "the LoRA never changed"
    assert x.grad is not None and torch.isfinite(x.grad).all(), "the input got no finite grad"


@pytest.mark.parametrize("seed", range(DRAWS))
def test_i1_every_draw_trains_its_lora_and_each_base_layer_passes_its_grad(seed, tmp_path):
    d = draw(seed)
    if known_quanto_autocast_gap(d):
        return assert_the_known_gap(d)
    reference, q = frozen(d), quantized(d)
    for i, (ref_layer, q_layer) in enumerate(zip(reference.layers, q.layers, strict=True)):
        x, g = rand(d, d.widths[i], salt=i), rand(d, d.widths[i + 1], salt=100 + i)
        want_out, want = through(d, ref_layer, x, g)
        got_out, got = through(d, q_layer, x, g)
        for what, a, b in (("output", got_out, want_out), ("input grad", got, want)):
            error = ((a.float() - b.float()).norm() / b.float().norm()).item()
            assert error <= 0.15, f"{d}: layer {i} {what} {error:.4f} relative to the unquantized layer's"

    source = frozen(d)
    signature = QuantizationFactory.state_shape_signature(source)
    count = sum(p.numel() for p in source.parameters())
    fresh = QuantizationFactory.quantize(source, d.scheme, backend_name=d.backend)
    path = os.path.join(tmp_path, "cache")
    QuantizationFactory.save_quantized(fresh, path, d.scheme, source_param_count=count, source_signature=signature)
    trains(d, fresh)
    if os.path.exists(os.path.join(path, "metadata.json")):
        loaded = QuantizationFactory.load_quantized(frozen(d), path, d.scheme)
        assert loaded is not None, f"{d}: its own fresh cache did not hit"
        trains(d, loaded)


@pytest.mark.parametrize("seed", range(DRAWS))
def test_i2_a_wrapped_layer_backs_off_its_dense_weight_and_any_other_is_untouched(seed, monkeypatch):
    d = draw(seed)
    if known_quanto_autocast_gap(d):
        return assert_the_known_gap(d)
    q = quantized(d)
    with monkeypatch.context() as m:
        m.setattr(trainable_backward, "make_quantized_linears_trainable", lambda module: 0)
        bypassed = quantized(d)
    for i, (layer, plain) in enumerate(zip(q.layers, bypassed.layers, strict=True)):
        wrapped = trainable_backward.is_wrapped(layer)
        assert wrapped == trainable_backward.needs_dequantized_backward(layer.weight), f"{d}: layer {i}"
        x, g = rand(d, d.widths[i], salt=i), rand(d, d.widths[i + 1], salt=100 + i)
        out, grad = through(d, layer, x, g)
        if wrapped:
            # The kernel runs in the weight's dtype, outside autocast; the casts at its boundary are
            # the only other ops between the output grad and the input grad.
            compute = layer.weight.dtype
            dense = trainable_backward.dense_weight(layer.weight, compute)
            expected_grad = (g.to(out.dtype).to(compute) @ dense).to(x.dtype)
            assert torch.equal(grad, expected_grad), f"{d}: layer {i} input grad is not grad_out @ dense(W)"
            if d.scheme == "int4":
                product = F.linear(x.to(compute), dense)
                expected = (product if layer.bias is None else product + layer.bias).to(out.dtype)
                # The dense weight is the kernel's own values; tinygemm sums in another order than
                # cuBLAS, so the products agree to the bf16 rounding of that order, not bit for bit.
                assert close_up_to_accumulation_order(out, expected), f"{d}: layer {i} forward is not linear(x, dense(W))"
        else:
            out_plain, grad_plain = through(d, plain, x, g)
            assert torch.equal(out, out_plain) and torch.equal(grad, grad_plain), f"{d}: layer {i} changed"
