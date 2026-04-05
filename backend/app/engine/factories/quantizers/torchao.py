import torch
import torch.nn as nn

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
except ImportError:
    torchao = None

from ..quantization_base import QuantizationBase
from app.core.logger import get_logger

logger = get_logger(__name__)


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
            return major >= 10

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
            quantize_(module, Int4WeightOnlyConfig())
        elif scheme == "int8":
            quantize_(module, Int8WeightOnlyConfig())
        elif scheme == "int8_dynamic":
            quantize_(module, Int8DynamicActivationInt8WeightConfig())
        elif scheme == "fp8":
            quantize_(module, Float8WeightOnlyConfig())
        elif scheme == "fp8_dynamic":
            quantize_(module, Float8DynamicActivationFloat8WeightConfig())
        elif scheme == "nvfp4":
            # Future-proofing for torchao NVFP4 additions
            # For now, fallback to int4 or wait for native torchao support
            logger.warning("nvfp4_not_yet_implemented_in_torchao", fallback="int4")
            quantize_(module, Int4WeightOnlyConfig())
        else:
            raise ValueError(f"Unknown torchao scheme: {scheme}")

        return module
