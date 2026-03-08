import structlog
from typing import Any
from ..optimizer_base import OptimizerBase

logger = structlog.get_logger(__name__)

class AdEMAMixStrategy(OptimizerBase):
    """AdEMAMix PyTorch optimizer strategy via pytorch_optimizer.
    A recent optimizer with fast convergence and stability.
    """

    def create_optimizer(
        self,
        params: Any,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        config: dict[str, Any]
    ) -> Any:
        beta3 = float(config.get("ademamix_beta3", 0.9999))
        alpha = float(config.get("ademamix_alpha", 5.0))
        
        logger.debug("creating_optimizer", type="AdEMAMix", lr=lr, beta3=beta3, alpha=alpha)
        
        try:
            from pytorch_optimizer import load_optimizer
            
            optimizer_class = load_optimizer("ademamix")
            return optimizer_class(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=(*betas, beta3),
                alpha=alpha
            )
            
        except ImportError:
            logger.error("pytorch_optimizer_library_missing_for_ademamix")
            raise RuntimeError("pytorch_optimizer is required for AdEMAMix. Please install it.")
