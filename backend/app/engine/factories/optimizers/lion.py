import structlog
from typing import Any
from ..optimizer_base import OptimizerBase

logger = structlog.get_logger(__name__)

class LionStrategy(OptimizerBase):
    """Lion (EvoLved Sign Momentum) PyTorch optimizer strategy via pytorch_optimizer."""

    def create_optimizer(
        self,
        params: Any,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        config: dict[str, Any]
    ) -> Any:
        logger.debug("creating_optimizer", type="Lion", lr=lr)
        
        try:
            from pytorch_optimizer import load_optimizer
            
            optimizer_class = load_optimizer("lion")
            return optimizer_class(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas
            )
            
        except ImportError:
            logger.error("pytorch_optimizer_library_missing_for_lion")
            raise RuntimeError("pytorch_optimizer is required for Lion. Please install it.")
