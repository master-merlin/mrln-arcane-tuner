from abc import ABC, abstractmethod
from typing import Any
import torch

class OptimizerBase(ABC):
    """Abstract base class for all optimizer strategies."""
    
    @abstractmethod
    def create_optimizer(
        self,
        params: Any,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        config: dict[str, Any]
    ) -> torch.optim.Optimizer:
        """Instantiate the optimizer.
        
        Args:
            params: Iterable of network parameters to optimize.
            lr: Base learning rate.
            weight_decay: Weight decay factor.
            betas: Tuple of Adam momentum factors (beta1, beta2).
            config: The raw training dictionary containing all optimizer-specific args.
            
        Returns:
            The configured PyTorch optimizer instance.
        """
        pass
