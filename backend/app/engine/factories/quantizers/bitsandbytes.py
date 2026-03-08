import torch
import torch.nn as nn

try:
    from transformers import BitsAndBytesConfig  # noqa: F401
    import bitsandbytes as bnb
except ImportError:
    bnb = None

from ..quantization_base import QuantizationBase
from app.core.logger import get_logger

logger = get_logger(__name__)


class BitsAndBytesBackend(QuantizationBase):
    """
    Bits and Bytes Quantization focusing primarily on NF4.
    Replaces older bitsandbytes-based diffusers integrations.
    """

    _BITS_MAP = {
        "nf4": 4.5,   # NormalFloat4 (highly optimized for LLM/Diffusion)
    }

    @classmethod
    def supported_schemes(cls) -> list[str]:
        return list(cls._BITS_MAP.keys())

    @classmethod
    def get_bits(cls, scheme: str) -> float:
        return cls._BITS_MAP.get(scheme, 16.0)

    @classmethod
    def is_available(cls, scheme: str) -> bool:
        if bnb is None or scheme not in cls._BITS_MAP:
            return False

        if not torch.cuda.is_available():
            return False

        # BitsAndBytes nf4 compute capability
        major, minor = torch.cuda.get_device_capability()
        return major >= 7

    @classmethod
    def quantize(cls, module: nn.Module, scheme: str, device: str = "cuda") -> nn.Module:
        """Applies bitsandbytes NF4 quantization in-place via PyTorch swapping."""
        if not cls.is_available(scheme):
             logger.warning("bitsandbytes_unavailable", scheme=scheme)
             return module
             
        logger.info("quantizing_bitsandbytes", scheme=scheme)

        if scheme == "nf4":
             module = cls._quantize_nf4(module, device)
        else:
             raise ValueError(f"Unknown bitsandbytes scheme: {scheme}")

        return module
        
    @classmethod
    def _quantize_nf4(cls, module: nn.Module, device: str) -> nn.Module:
        """Recursively replaces Linear and Conv1D layers with NF4 bnb layers."""
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                in_features = child.in_features
                out_features = child.out_features
                has_bias = child.bias is not None
                
                # Replace with BnB Linear4bit
                new_layer = bnb.nn.Linear4bit(
                    in_features, out_features, bias=has_bias, 
                    compute_dtype=torch.bfloat16, 
                    quant_type="nf4"
                )
                
                cls._transfer_weights(child, new_layer)
                setattr(module, name, new_layer)
            elif isinstance(child, nn.Conv2d):
                # Optionally handle Conv2D swapping if needed (usually ignored in LoRA training)
                pass
            else:
                cls._quantize_nf4(child, device)


        return module.to(device)
        
    @classmethod
    def _transfer_weights(cls, old_layer: nn.Module, new_layer: nn.Module):
        """Helper to move unquantized weights into the quantized layer cleanly."""
        new_layer.weight.data.copy_(old_layer.weight.data)
        if old_layer.bias is not None and new_layer.bias is not None:
            new_layer.bias.data.copy_(old_layer.bias.data)
