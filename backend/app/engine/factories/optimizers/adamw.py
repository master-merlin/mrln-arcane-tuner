import torch
import structlog
from typing import Any
from ..optimizer_base import OptimizerBase

logger = structlog.get_logger(__name__)

class AdamWStrategy(OptimizerBase):
    """Standard AdamW PyTorch optimizer strategy."""

    def create_optimizer(
        self,
        params: Any,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        config: dict[str, Any]
    ) -> torch.optim.Optimizer:
        logger.debug("creating_optimizer", type="AdamW", lr=lr)
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
