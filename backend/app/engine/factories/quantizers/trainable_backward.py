"""A frozen quantized ``Linear`` whose kernel has no derivative gets its backward from its dense weight.

LoRA trains adapters beside a frozen base layer, so the base layer's backward needs only the grad of
its input, ``grad_out @ W``. Three torchao weights cannot give it:

- tile-packed int4 (``Int4TilePackedTo4dTensor``, PyTorch's tinygemm kernel): no derivative for
  ``aten::_weight_int4pack_mm``, so backward raises;
- Float8 with dynamic activation quantization (``Float8Tensor`` with ``act_quant_kwargs``): no
  derivative for ``aten::_scaled_mm``, so backward raises;
- int8 with dynamic activation quantization (``Int8Tensor`` with ``act_quant_kwargs``): the
  output tracks no grad at all, so the input silently gets none through the layer.

Weight-only int8 (``Int8Tensor`` without ``act_quant_kwargs``) and weight-only Float8 dequantize in
their matmul, so their own backward already gives the input's grad.

For a ``Linear`` holding one of these, the forward still runs the layer's own quantized kernel,
bit for bit. The backward multiplies the output grad by the dense weight, rebuilt just for it and
gone once it returns. Every other weight is left alone.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Guarded like the rest of torchao, and each name on its own: a torchao that drops one (0.18 dropped
# LinearActivationQuantizedTensor) disables only the branch that names it, never the others.
try:
    from torchao.quantization import Float8Tensor
except Exception:  # noqa: BLE001 - see quantizers/torchao.py
    Float8Tensor = None

try:
    from torchao.quantization import Int4TilePackedTo4dTensor
except Exception:  # noqa: BLE001
    Int4TilePackedTo4dTensor = None

try:
    from torchao.quantization import Int8Tensor
except Exception:  # noqa: BLE001
    Int8Tensor = None

try:  # a torchao prototype, guarded on its own (as in quantizers/torchao.py)
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
except Exception:  # noqa: BLE001
    NVFP4Tensor = None

try:  # guarded like quantizers/quanto.py
    from optimum.quanto.nn import QConv2d, QLinear
except Exception:  # noqa: BLE001
    QConv2d = QLinear = None

_CONTIGUOUS_FLAG = "_contiguous_input"
_WEIGHT_DTYPE_FLAG = "_weight_dtype"

_WRAPPED_FLAG = "_dequantized_backward"
_EYE_BLOCK = 1 << 20  # elements of identity per block of the int4 rebuild: 2 MiB in bf16


def needs_dequantized_backward(weight: torch.Tensor) -> bool:
    """A weight whose quantized kernel raises in backward, or drops the input's grad."""
    if Int4TilePackedTo4dTensor is not None and isinstance(weight, Int4TilePackedTo4dTensor):
        return True
    for dynamic in (Float8Tensor, Int8Tensor):  # dynamic activation quantization only
        if dynamic is not None and isinstance(weight, dynamic):
            return weight.act_quant_kwargs is not None
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


def _forward(linear: nn.Linear, input: torch.Tensor) -> torch.Tensor:  # noqa: A002 - nn.Linear.forward's name
    """The kernel in its weight's own dtype, outside autocast; the output as autocast would give it.

    These kernels are not autocast-safe: under fp16 autocast, int8_dynamic's output was thousands of
    times off, and tile-packed int4 takes bf16 only (an fp32 model's int4 layers are kept in bf16, see
    quantizers/torchao.py). So the input is cast to the weight's dtype, autocast is off around the
    kernel, and the output is cast to what a plain Linear returns here: the autocast dtype under
    autocast, else the input's dtype. In bf16 with no autocast every cast is a no-op.

    Its input is named ``input``, as ``nn.Linear.forward``'s is, so ``layer(input=x)`` still works.
    """
    x = input
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

    - torchao's NVFP4: its forward on a transposed or batch-expanded 3D input, or (without a bias)
      on a batch-1 token split, falls back to ``bmm``, which calls ``expand`` on the weight, and
      ``NVFP4Tensor`` has none.
    - optimum-quanto's ``QLinear``: its backward takes ``input.view(-1, in_features)``, which a
      transposed or expanded input that needs its grad cannot give.
    Every other offered scheme runs any layout.
    """
    if NVFP4Tensor is not None and isinstance(getattr(layer, "weight", None), NVFP4Tensor):
        return True
    return QLinear is not None and isinstance(layer, QLinear)


def makes_input_contiguous(layer: nn.Module) -> bool:
    return bool(getattr(layer, _CONTIGUOUS_FLAG, False))


def _as_2d(x: torch.Tensor, in_features: int) -> torch.Tensor:
    # ``contiguous()`` alone is not enough: it keeps a size-1 dim's stride, which ``is_contiguous()``
    # ignores but matmul's fold does not (a batch-1 token split), so NVFP4 still reaches ``bmm``.
    # The 2D view of a contiguous tensor always has canonical strides.
    return x.contiguous().reshape(-1, in_features)


def _contiguous_input_forward(original, in_features: int):
    def forward(*args, **kwargs) -> torch.Tensor:
        # The input as nn.Linear.forward takes it: first positional, or the keyword ``input``.
        # A canonical contiguous input is only re-viewed: such a call computes exactly what it did.
        if args:
            x = args[0]
            args = (_as_2d(x, in_features), *args[1:])
        elif "input" in kwargs:
            x = kwargs["input"]
            kwargs["input"] = _as_2d(x, in_features)
        else:
            return original(*args, **kwargs)
        out = original(*args, **kwargs)
        return out.reshape(*x.shape[:-1], out.shape[-1])

    return forward


def runs_in_weight_dtype(layer: nn.Module) -> bool:
    """An optimum-quanto layer, which cannot run under autocast with fp32 weights.

    - ``QLinear``: quanto's ``QuantizedLinearFunction`` multiplies under autocast in its forward
      (fp16 out, without a bias) but outside it in its backward, where an fp16 grad meets the fp32
      weight: "expected mat1 and mat2 to have the same dtype".
    - ``QConv2d``: autocast casts the conv's weight, and a quanto weight refuses ``.to(dtype)``:
      "The dtype of a ... Tensor cannot be changed", already in the forward.
    These are the only modules quanto's weight quantization creates.
    """
    return any(cls is not None and isinstance(layer, cls) for cls in (QLinear, QConv2d))


def in_weight_dtype(layer: nn.Module) -> bool:
    return bool(getattr(layer, _WEIGHT_DTYPE_FLAG, False))


def _weight_dtype_forward(original, layer: nn.Module):
    def forward(*args, **kwargs) -> torch.Tensor:
        # The input as the layer's forward takes it: first positional, or the keyword ``input``.
        x = args[0] if args else kwargs.get("input")
        if not isinstance(x, torch.Tensor) or not torch.is_autocast_enabled(x.device.type):
            return original(*args, **kwargs)  # no autocast: exactly the call it was
        device = x.device.type
        out_dtype = torch.get_autocast_dtype(device)
        x = x.to(layer.weight.dtype)
        if args:
            args = (x, *args[1:])
        else:
            kwargs["input"] = x
        # Forward and backward in the weight's dtype; the output as a plain layer gives it here.
        with torch.autocast(device, enabled=False):
            out = original(*args, **kwargs)
        return out.to(out_dtype)

    return forward


def plain_conv_adapters(model: nn.Module) -> int:
    """Make every LoRA adapter peft built as a quanto ``QConv2d`` a plain ``nn.Conv2d``.

    peft builds a conv LoRA's ``lora_A`` / ``lora_B`` with ``type(base_layer)``: on a quanto conv
    they are ``QConv2d`` layers with no quantized weight, whose forward passes ``None`` as the
    weight. Each one is replaced by an ``nn.Conv2d`` of the same configuration, on the same device
    and in the same dtype, holding the very values peft initialised; so its state-dict keys are a
    plain LoRA's, without quanto's scale buffers. Call it right after ``get_peft_model``.

    Returns how many adapters it replaced.
    """
    if QConv2d is None:
        return 0
    replaced = 0
    for module in model.modules():
        for attr in ("lora_A", "lora_B"):
            adapters = getattr(module, attr, None)
            if not isinstance(adapters, nn.ModuleDict):
                continue
            for name, conv in list(adapters.items()):
                if not isinstance(conv, QConv2d):
                    continue
                plain = nn.Conv2d(
                    conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
                    padding=conv.padding, dilation=conv.dilation, groups=conv.groups,
                    bias=conv.bias is not None, padding_mode=conv.padding_mode,
                    device=conv.weight.device, dtype=conv.weight.dtype,
                )
                with torch.no_grad():
                    plain.weight.copy_(conv.weight)
                    if conv.bias is not None:
                        plain.bias.copy_(conv.bias)
                plain.weight.requires_grad_(conv.weight.requires_grad)
                if conv.bias is not None:
                    plain.bias.requires_grad_(conv.bias.requires_grad)
                adapters[name] = plain
                replaced += 1
    return replaced


def is_wrapped(module: nn.Module) -> bool:
    return bool(getattr(module, _WRAPPED_FLAG, False))


def make_quantized_linears_trainable(module: nn.Module) -> int:
    """Give every ``Linear`` under ``module`` whose weight needs it the dense-weight backward, every
    one whose kernel fails on a non-contiguous input a contiguous one, and every quanto layer its
    weight's dtype under autocast.

    Idempotent. Returns how many layers got the dense-weight backward in this call.
    """
    wrapped = 0
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            if needs_contiguous_input(layer) and not makes_input_contiguous(layer):
                layer.forward = _contiguous_input_forward(layer.forward, layer.in_features)  # its own forward, any layout
                setattr(layer, _CONTIGUOUS_FLAG, True)
            if not is_wrapped(layer) and needs_dequantized_backward(layer.weight):
                layer.forward = _forward.__get__(layer)  # bound to the layer, like its own forward
                setattr(layer, _WRAPPED_FLAG, True)
                wrapped += 1
        if runs_in_weight_dtype(layer) and not in_weight_dtype(layer):
            layer.forward = _weight_dtype_forward(layer.forward, layer)  # outermost: around the wraps above
            setattr(layer, _WEIGHT_DTYPE_FLAG, True)
    return wrapped
