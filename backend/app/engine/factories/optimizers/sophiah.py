import structlog
from typing import Any
from ..optimizer_base import OptimizerBase

logger = structlog.get_logger(__name__)

class SophiaHStrategy(OptimizerBase):
    """SophiaH (Second-order Clipped Stochastic Optimization) PyTorch optimizer strategy via pytorch_optimizer."""

    def create_optimizer(
        self,
        params: Any,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        config: dict[str, Any]
    ) -> Any:
        rho = float(config.get("sophia_rho", 0.04))
        p = float(config.get("sophia_p", 0.01))
        update_period = int(config.get("sophia_update_period", 10))
        num_samples = int(config.get("sophia_num_samples", 1))
        hessian_dist = config.get("sophia_hessian_distribution", "gaussian")
        
        logger.debug("creating_optimizer", type="SophiaH", lr=lr, rho=rho, p=p, update_period=update_period)
        
        try:
            from pytorch_optimizer import load_optimizer
            
            optimizer_class = load_optimizer("sophiah")
            return optimizer_class(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                rho=rho,
                p=p,
                update_period=update_period,
                num_samples=num_samples,
                hessian_distribution=hessian_dist
            )
            
        except ImportError:
            logger.error("pytorch_optimizer_library_missing_for_sophiah")
            raise RuntimeError("pytorch_optimizer is required for SophiaH. Please install it.")
