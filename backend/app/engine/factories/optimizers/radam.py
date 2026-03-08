import structlog
from typing import Any
from ..optimizer_base import OptimizerBase

logger = structlog.get_logger(__name__)

class RAdamStrategy(OptimizerBase):
    """RAdam (Rectified Adam) PyTorch optimizer strategy via pytorch_optimizer.
    Introduces a term to rectify the variance of the adaptive learning rate.
    """

    def create_optimizer(
        self,
        params: Any,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        config: dict[str, Any]
    ) -> Any:
        n_sma_threshold = int(config.get("radam_n_sma_threshold", 5))
        logger.debug("creating_optimizer", type="RAdam", lr=lr, n_sma_threshold=n_sma_threshold)
        
        try:
            from pytorch_optimizer import load_optimizer
            
            optimizer_class = load_optimizer("radam")
            return optimizer_class(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                n_sma_threshold=n_sma_threshold
            )
            
        except ImportError:
            logger.error("pytorch_optimizer_library_missing_for_radam")
            raise RuntimeError("pytorch_optimizer is required for RAdam. Please install it.")
