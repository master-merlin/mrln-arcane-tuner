"""A frozen quantized ``Linear`` whose kernel has no derivative gets its backward from its dense weight.

LoRA trains adapters beside a frozen base layer, so the base layer's backward needs only the grad of
its input, ``grad_out @ W``. Three torchao weights cannot give it:

- tile-packed int4 (``Int4TilePackedTo4dTensor``, PyTorch's tinygemm kernel): no derivative for
  ``aten::_weight_int4pack_mm``, so backward raises;
- Float8 with dynamic activation quantization (``Float8Tensor`` with ``act_quant_kwargs``): no
  derivative for ``aten::_scaled_mm``, so backward raises;
- int8 with dynamic activation quantization (``LinearActivationQuantizedTensor``): the output
  tracks no grad at all, so the input silently gets none through the layer.

For a ``Linear`` holding one of these, the forward still runs the layer's own quantized kernel,
bit for bit. The backward multiplies the output grad by the dense weight, rebuilt just for it and
gone once it returns. Every other weight is left alone.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # Guarded like the rest of torchao: absent or broken, nothing here wraps anything.
    from torchao.quantization import Float8Tensor, Int4TilePackedTo4dTensor, LinearActivationQuantizedTensor
except Exception:  # noqa: BLE001 - see quantizers/torchao.py
    Float8Tensor = Int4TilePackedTo4dTensor = LinearActivationQuantizedTensor = None

try:  # a torchao prototype, guarded on its own (as in quantizers/torchao.py)
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
except Exception:  # noqa: BLE001
    NVFP4Tensor = None

try:  # guarded like quantizers/quanto.py
    from optimum.quanto.nn import QLinear
except Exception:  # noqa: BLE001
    QLinear = None

_CONTIGUOUS_FLAG = "_contiguous_input"

_WRAPPED_FLAG = "_dequantized_backward"
_EYE_BLOCK = 1 << 20  # elements of identity per block of the int4 rebuild: 2 MiB in bf16


def needs_dequantized_backward(weight: torch.Tensor) -> bool:
    """A weight whose quantized kernel raises in backward, or drops the input's grad."""
    if Int4TilePackedTo4dTensor is not None and isinstance(weight, Int4TilePackedTo4dTensor):
        return True
    if Float8Tensor is not None and isinstance(weight, Float8Tensor):
        return weight.act_quant_kwargs is not None
    if LinearActivationQuantizedTensor is not None and isinstance(weight, LinearActivationQuantizedTensor):
        return True
    return False


def dense_weight(weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The weight as a plain ``(out, in)`` tensor of ``dtype``.

    Its own ``dequantize()``, or, for tinygemm int4, which has none, the forward kernel on an
    identity: ``linear(I, W) = I @ W.T = W.T``, so the values are exactly the ones the kernel uses.
    The identity goes in row blocks of at most ``_EYE_BLOCK`` elements, never in_features²: each
    output row has one nonzero term, so the blocks give the same values bit for bit.
    """
    if Int4TilePackedTo4dTensor is not None and isinstance(weight, Int4TilePackedTo4dTensor):
        out_features, in_features = weight.shape
        rows = max(1, min(in_features, _EYE_BLOCK // in_features))
        dense_t = torch.empty(in_features, out_features, device=weight.device, dtype=dtype)
        for start in range(0, in_features, rows):
            stop = min(start + rows, in_features)
            eye = torch.zeros(stop - start, in_features, device=weight.device, dtype=dtype)
            index = torch.arange(stop - start, device=weight.device)
            eye[index, index + start] = 1
            dense_t[start:stop] = F.linear(eye, weight)
            del eye
        return dense_t.t()
    if LinearActivationQuantizedTensor is not None and isinstance(weight, LinearActivationQuantizedTensor):
        weight = weight.original_weight_tensor  # the activation quantization wraps the int8 weight
    return weight.dequantize().to(dtype)


class _DequantizedBackwardLinear(torch.autograd.Function):
    """``linear(x, W, b)`` on the quantized kernel; the input's grad from the dense ``W``."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        ctx.save_for_backward(weight)  # the quantized weight only: the dense one is made in backward
        ctx.x_dtype = x.dtype
        ctx.bias_needs_grad = bias is not None and bias.requires_grad
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, None, torch.Tensor | None]:
        (weight,) = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = (grad_out @ dense_weight(weight, grad_out.dtype)).to(ctx.x_dtype)
        grad_bias = grad_out.reshape(-1, grad_out.shape[-1]).sum(0) if ctx.bias_needs_grad else None
        return grad_x, None, grad_bias


def _forward(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """The kernel in its weight's own dtype, outside autocast; the output as autocast would give it.

    These kernels are not autocast-safe: under fp16 autocast, int8_dynamic's output was thousands of
    times off, and tile-packed int4 takes bf16 only (an fp32 model's int4 layers are kept in bf16, see
    quantizers/torchao.py). So the input is cast to the weight's dtype, autocast is off around the
    kernel, and the output is cast to what a plain Linear returns here: the autocast dtype under
    autocast, else the input's dtype. In bf16 with no autocast every cast is a no-op.
    """
    weight = linear.weight
    if weight.requires_grad:
        raise RuntimeError(
            "This quantized layer's weight must stay frozen: its kernel has no weight grad, and its "
            "backward gives the input's grad only. Train it unquantized, or train adapters beside it."
        )
    device = x.device.type
    autocast = torch.is_autocast_enabled(device)
    out_dtype = torch.get_autocast_dtype(device) if autocast else x.dtype
    with torch.autocast(device, enabled=False):
        out = _DequantizedBackwardLinear.apply(x.to(weight.dtype), weight, linear.bias)
    return out.to(out_dtype)


def needs_contiguous_input(layer: nn.Module) -> bool:
    """A quantized Linear whose kernel fails on a non-contiguous input.

    - torchao's NVFP4: its forward on a transposed or batch-expanded 3D input falls back to ``bmm``,
      which calls ``expand`` on the weight, and ``NVFP4Tensor`` has none.
    - optimum-quanto's ``QLinear``: its backward takes ``input.view(-1, in_features)``, which a
      transposed or expanded input that needs its grad cannot give.
    Every other offered scheme runs any layout.
    """
    if NVFP4Tensor is not None and isinstance(getattr(layer, "weight", None), NVFP4Tensor):
        return True
    return QLinear is not None and isinstance(layer, QLinear)


def makes_input_contiguous(layer: nn.Module) -> bool:
    return bool(getattr(layer, _CONTIGUOUS_FLAG, False))


def _contiguous_input_forward(original):
    def forward(*args, **kwargs) -> torch.Tensor:
        # The input as nn.Linear.forward takes it: first positional, or the keyword ``input``.
        # A contiguous input is returned as it is: such a call computes exactly what it did.
        if args:
            args = (args[0].contiguous(), *args[1:])
        elif "input" in kwargs:
            kwargs["input"] = kwargs["input"].contiguous()
        return original(*args, **kwargs)

    return forward


def is_wrapped(module: nn.Module) -> bool:
    return bool(getattr(module, _WRAPPED_FLAG, False))


def make_quantized_linears_trainable(module: nn.Module) -> int:
    """Give every ``Linear`` under ``module`` whose weight needs it the dense-weight backward, and
    every one whose kernel fails on a non-contiguous input a contiguous one.

    Idempotent. Returns how many layers got the dense-weight backward in this call.
    """
    wrapped = 0
    for layer in module.modules():
        if not isinstance(layer, nn.Linear):
            continue
        if needs_contiguous_input(layer) and not makes_input_contiguous(layer):
            layer.forward = _contiguous_input_forward(layer.forward)  # its own forward, any layout
            setattr(layer, _CONTIGUOUS_FLAG, True)
        if is_wrapped(layer):
            continue
        if needs_dequantized_backward(layer.weight):
            layer.forward = _forward.__get__(layer)  # bound to the layer, like its own forward
            setattr(layer, _WRAPPED_FLAG, True)
            wrapped += 1
    return wrapped
