from abc import ABC, abstractmethod
import torch.nn as nn

class QuantizationBase(ABC):
    """Abstract interface for a quantization strategy backend."""

    @classmethod
    @abstractmethod
    def supported_schemes(cls) -> list[str]:
        """Return a list of quantization schemes supported by this backend."""
        pass

    @classmethod
    @abstractmethod
    def is_available(cls, scheme: str) -> bool:
        """Check if the system supports the requested scheme for this backend."""
        pass

    @classmethod
    @abstractmethod
    def quantize(cls, module: nn.Module, scheme: str, device: str | None = None, **kwargs) -> nn.Module:
        """Apply quantization to the given module in-place if possible."""
        pass
    
    @classmethod
    def get_bits(cls, scheme: str) -> float:
        """Return the estimated bits-per-parameter for the scheme."""
        return 16.0  # Default unquantized
