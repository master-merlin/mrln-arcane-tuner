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

    def kept_state_dtypes(self, optimizer: torch.optim.Optimizer) -> dict[str, torch.dtype]:
        """The state this optimizer keeps at a dtype of its own, not its param's, by key.

        torch's ``load_state_dict`` casts floating state to its param's dtype, which
        is what an optimizer whose state follows its param needs. State named here
        comes back at its own dtype instead.
        """
        return {}
