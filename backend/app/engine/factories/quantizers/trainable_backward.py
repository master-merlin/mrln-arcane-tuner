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

_WRAPPED_FLAG = "_dequantized_backward"


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
    """
    if Int4TilePackedTo4dTensor is not None and isinstance(weight, Int4TilePackedTo4dTensor):
        eye = torch.eye(weight.shape[1], device=weight.device, dtype=dtype)
        return F.linear(eye, weight).t().contiguous()
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
    if linear.weight.requires_grad:
        raise RuntimeError(
            "This quantized layer's weight must stay frozen: its kernel has no weight grad, and its "
            "backward gives the input's grad only. Train it unquantized, or train adapters beside it."
        )
    return _DequantizedBackwardLinear.apply(x, linear.weight, linear.bias)


def is_wrapped(module: nn.Module) -> bool:
    return bool(getattr(module, _WRAPPED_FLAG, False))


def make_quantized_linears_trainable(module: nn.Module) -> int:
    """Give every ``Linear`` under ``module`` whose weight needs it the dense-weight backward.

    Idempotent. Returns how many layers were wrapped by this call.
    """
    wrapped = 0
    for layer in module.modules():
        if not isinstance(layer, nn.Linear) or is_wrapped(layer):
            continue
        if needs_dequantized_backward(layer.weight):
            layer.forward = _forward.__get__(layer)  # bound to the layer, like its own forward
            setattr(layer, _WRAPPED_FLAG, True)
            wrapped += 1
    return wrapped
