import torch
import structlog
from typing import Any
from ..optimizer_base import OptimizerBase

logger = structlog.get_logger(__name__)

class AdamW8bitStrategy(OptimizerBase):
    """8-bit quantized AdamW optimizer (via bitsandbytes)."""

    def create_optimizer(
        self,
        params: Any,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        config: dict[str, Any]
    ) -> torch.optim.Optimizer:
        logger.debug("creating_optimizer", type="AdamW8bit", lr=lr)
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay, betas=betas)
        except ImportError:
            logger.warning("bitsandbytes_not_found_fallback_adamw")
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
