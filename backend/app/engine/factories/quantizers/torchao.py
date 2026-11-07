import torch
import torch.nn as nn

# `except Exception`, NOT `except ImportError` (ARCHITECTURE D1: nothing
# imported at startup may raise, ever). This runs during `import app.main`.
# A dependency that is PRESENT but cannot initialise -- a numba JIT cache it
# cannot write, a CUDA library it cannot open -- raises something other than
# ImportError, and before 2026-09-03 that escaped and killed the process
# rather than disabling the feature. Absent and broken are the same outcome
# here, so the guard is written about the outcome.
try:
    from torchao.quantization import (
        quantize_,
        Int4WeightOnlyConfig,
        Int8WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
        Float8WeightOnlyConfig,
        Float8DynamicActivationFloat8WeightConfig,
    )
    import torchao
except Exception:  # noqa: BLE001 — see the note above
    torchao = None

# NVFP4 is a torchao prototype: guarded on its own, so its absence disables nvfp4 only.
try:
    from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
except Exception:  # noqa: BLE001 — see the note above
    NVFP4WeightOnlyConfig = NVFP4Tensor = None

try:
    from torchao.quantization import Int4TilePackedTo4dTensor
except Exception:  # noqa: BLE001 — see the note above
    Int4TilePackedTo4dTensor = None

from ..quantization_base import QuantizationBase
from app.core.logger import get_logger

logger = get_logger(__name__)

_NVFP4_BLOCK = 16


def _nvfp4_quantizable(module: nn.Module, fqn: str) -> bool:
    return isinstance(module, nn.Linear) and all(d % _NVFP4_BLOCK == 0 for d in module.weight.shape)


def _quantize_in_bf16(module: nn.Module, config: object) -> None:
    """``quantize_`` for a layout that takes bf16 weights only, in a model of any dtype.

    Each ``Linear`` of another dtype goes to bf16 first. One the config then leaves unquantized
    (int4 skips a layer whose in_features its group size does not divide) goes back to its own
    dtype, so only quantized layers change dtype; their forward casts at the boundary
    (trainable_backward._forward).
    """
    cast: dict[str, torch.dtype] = {}
    for name, layer in module.named_modules():
        if isinstance(layer, nn.Linear) and layer.weight.dtype != torch.bfloat16:
            cast[name] = layer.weight.dtype
            layer.to(torch.bfloat16)
    quantize_(module, config)
    for name, dtype in cast.items():
        layer = module.get_submodule(name)
        if type(layer.weight) is nn.Parameter and type(layer.weight.data) is torch.Tensor:
            layer.to(dtype)


class TorchAOBackend(QuantizationBase):
    """
    Native PyTorch Quantization utilizing `torchao` (Architecture Optimization).
    Focuses on performant integer schemes and highly optimized Blackwell/Ada/Hopper FP8/NVFP4.
    """

    # Estimated bits per param for VRAM calculation
    _BITS_MAP = {
        "int4": 4.5,
        "int8": 8.5,
        "int8_dynamic": 8.5,
        "fp8": 8.5,
        "fp8_dynamic": 8.5,
        "nvfp4": 4.5, # Native Hopper/Blackwell
    }

    @classmethod
    def supported_schemes(cls) -> list[str]:
        return list(cls._BITS_MAP.keys())

    @classmethod
    def get_bits(cls, scheme: str) -> float:
        return cls._BITS_MAP.get(scheme, 16.0)

    @classmethod
    def is_available(cls, scheme: str) -> bool:
        if torchao is None:
            return False

        if not torch.cuda.is_available():
            return False

        compute_cap = torch.cuda.get_device_capability()
        major, minor = compute_cap

        # Ada / Hopper / Blackwell (SM 89+)
        if scheme in ["fp8", "fp8_dynamic"]:
            return major >= 8 and (major > 8 or minor >= 9)
            
        # Blackwell specific (SM 100+)
        if scheme == "nvfp4":
            return NVFP4WeightOnlyConfig is not None and major >= 10

        # General integer schemes (SM 80+)
        return major >= 8

    @classmethod
    def quantize(cls, module: nn.Module, scheme: str, device: str | None = None) -> nn.Module:
        """Applies torchao quantization in-place."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not cls.is_available(scheme):
            logger.warning("torchao_unavailable", scheme=scheme)
            return module

        logger.info("quantizing_torchao", scheme=scheme)

        # Ensure module is on GPU and half-precision before quantizing
        module = module.to(device)

        if scheme == "int4":
            # PyTorch's own tinygemm kernel. The default "plain" packing needs mslk, which has no
            # Windows build; tile-packed runs everywhere a card does. Its backward comes from
            # trainable_backward, wired in by QuantizationFactory. It takes bf16 weights only.
            _quantize_in_bf16(module, Int4WeightOnlyConfig(int4_packing_format="tile_packed_to_4d"))
        elif scheme == "int8":
            quantize_(module, Int8WeightOnlyConfig())
        elif scheme == "int8_dynamic":
            quantize_(module, Int8DynamicActivationInt8WeightConfig())
        elif scheme == "fp8":
            quantize_(module, Float8WeightOnlyConfig())
        elif scheme == "fp8_dynamic":
            quantize_(module, Float8DynamicActivationFloat8WeightConfig())
        elif scheme == "nvfp4":
            # Weight-only NVFP4: its tensor dequantizes in the matmul, so it trains as it is. It
            # raises on a weight whose dims 16 does not divide; such a layer stays unquantized, as
            # int4 and fp8_dynamic leave theirs.
            quantize_(module, NVFP4WeightOnlyConfig(), filter_fn=_nvfp4_quantizable)
        else:
            raise ValueError(f"Unknown torchao scheme: {scheme}")

        return module

    @classmethod
    def weight_type(cls, scheme: str) -> type | None:
        """The tensor type ``quantize`` makes for ``scheme``, where an older build made another.

        A cached entry for these schemes is loaded only if its quantized weights are of this type:
        until 2026-11 ``int4`` (and ``nvfp4``, through a fallback) quantized to torchao's plain int4,
        which has no Windows kernel and no backward.
        """
        return {"int4": Int4TilePackedTo4dTensor, "nvfp4": NVFP4Tensor}.get(scheme)
