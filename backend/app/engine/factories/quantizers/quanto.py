import torch
import torch.nn as nn

try:
    from optimum.quanto import quantize, freeze, qint4, qint8, qfloat8, qfloat8_e4m3fn, qfloat8_e5m2
    import optimum.quanto  # noqa: F401
except ImportError:
    quanto = None

from ..quantization_base import QuantizationBase
from app.core.logger import get_logger

logger = get_logger(__name__)


class QuantoBackend(QuantizationBase):
    """
    Optimum-Quanto provides model-agnostic quantization for PyTorch.
    It supports int8, int4, and float8 mapping cleanly to Mac, CPU, and GPU devices.
    """

    _SCHEME_MAP = {
        "qint4": 4.5,
        "qint8": 8.5,
        "qfloat8": 8.5,
        "qfloat8_e4m3fn": 8.5,
        "qfloat8_e5m2": 8.5,
    }

    @classmethod
    def supported_schemes(cls) -> list[str]:
        return list(cls._SCHEME_MAP.keys())

    @classmethod
    def get_bits(cls, scheme: str) -> float:
        return cls._SCHEME_MAP.get(scheme, 16.0)

    @classmethod
    def is_available(cls, scheme: str) -> bool:
        if 'optimum' not in globals() or scheme not in cls._SCHEME_MAP:
            return False
            
        # Float8 requires Ampere/sm_80 + (unlike PyTorch native which is Hopper sm_89)
        if "float8" in scheme:
            if not torch.cuda.is_available():
                return False
            major, _ = torch.cuda.get_device_capability()
            return major >= 8

        return True

    @classmethod
    def quantize(cls, module: nn.Module, scheme: str, device: str | None = None) -> nn.Module:
        """Applies Hugging Face's optimum-quanto quantization in-place."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not cls.is_available(scheme):
             logger.warning("optimum_quanto_unavailable", scheme=scheme)
             return module
             
        logger.info("quantizing_quanto", scheme=scheme)

        # Move to GPU first, ensure FP16 weights
        module.to(device)

        qtype = None
        if scheme == "qint4":
            qtype = qint4
        elif scheme == "qint8":
            qtype = qint8
        elif scheme == "qfloat8":
             qtype = qfloat8
        elif scheme == "qfloat8_e4m3fn":
             qtype = qfloat8_e4m3fn
        elif scheme == "qfloat8_e5m2":
             qtype = qfloat8_e5m2
             
        if qtype is None:
             raise ValueError(f"Unknown quanto scheme: {scheme}")

        # Quanto modifies the graph in place and requires freezing.
        quantize(module, weights=qtype)
        freeze(module)

        return module
