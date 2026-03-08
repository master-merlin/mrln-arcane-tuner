import structlog
from typing import Any
from ..optimizer_base import OptimizerBase

logger = structlog.get_logger(__name__)

class StableAdamWStrategy(OptimizerBase):
    """StableAdamW PyTorch optimizer strategy via pytorch_optimizer.
    Robust AdamW variant designed to mitigate training instabilities.
    """

    def create_optimizer(
        self,
        params: Any,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        config: dict[str, Any]
    ) -> Any:
        kahan_sum = config.get("stableadamw_kahan_sum", False)
        logger.debug("creating_optimizer", type="StableAdamW", lr=lr, kahan_sum=kahan_sum)
        
        try:
            from pytorch_optimizer import load_optimizer
            
            optimizer_class = load_optimizer("stableadamw")
            return optimizer_class(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                kahan_sum=kahan_sum
            )
            
        except ImportError:
            logger.error("pytorch_optimizer_library_missing_for_stableadamw")
            raise RuntimeError("pytorch_optimizer is required for StableAdamW. Please install it.")
