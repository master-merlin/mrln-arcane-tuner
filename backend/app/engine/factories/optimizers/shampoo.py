import structlog
from typing import Any
from ..optimizer_base import OptimizerBase

logger = structlog.get_logger(__name__)

class ShampooStrategy(OptimizerBase):
    """Shampoo PyTorch optimizer strategy via pytorch_optimizer.
    A second-order optimizer for large-scale training.
    """

    def create_optimizer(
        self,
        params: Any,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        config: dict[str, Any]
    ) -> Any:
        preconditioning_steps = int(config.get("shampoo_preconditioning_compute_steps", 1))
        logger.debug("creating_optimizer", type="Shampoo", lr=lr, preconditioning_steps=preconditioning_steps)
        
        try:
            from pytorch_optimizer import load_optimizer
            
            optimizer_class = load_optimizer("shampoo")
            return optimizer_class(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                preconditioning_compute_steps=preconditioning_steps
            )
            
        except ImportError:
            logger.error("pytorch_optimizer_library_missing_for_shampoo")
            raise RuntimeError("pytorch_optimizer is required for Shampoo. Please install it.")
