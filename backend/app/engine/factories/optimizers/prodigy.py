import torch
import structlog
from typing import Any
from ..optimizer_base import OptimizerBase

logger = structlog.get_logger(__name__)

class ProdigyStrategy(OptimizerBase):
    """Prodigy optimizer with adaptive D-estimate and learning rate."""

    def create_optimizer(
        self,
        params: Any,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        config: dict[str, Any]
    ) -> torch.optim.Optimizer:
        # Prodigy uses (0.9, 0.99) by default for diffusion training.
        # β₂=0.999 causes slow variance adaptation → NaN in early steps.
        # If the user's config exactly matches the Adam default (0.999), we override it.
        # Otherwise, we respect their explicit override.
        prodigy_betas = betas if betas[1] != 0.999 else (0.9, 0.99)
        
        d_coef = float(config.get("d_coef", 0.8))
        growth_rate = float(config.get("growth_rate", 1.02))
        decouple = bool(config.get("decouple", True))
        safeguard_warmup = bool(config.get("safeguard_warmup", True))
        use_bias_correction = bool(config.get("use_bias_correction", True))

        try:
            from prodigyopt import Prodigy
            logger.info(
                "creating_prodigy",
                lr=lr, d_coef=d_coef, growth_rate=growth_rate,
                betas=prodigy_betas, safeguard_warmup=safeguard_warmup,
            )
            return Prodigy(
                params,
                lr=lr,
                betas=prodigy_betas,
                weight_decay=weight_decay,
                d_coef=d_coef,
                growth_rate=growth_rate,
                decouple=decouple,
                safeguard_warmup=safeguard_warmup,
                use_bias_correction=use_bias_correction,
            )
        except ImportError:
            logger.warning("prodigyopt_not_found_fallback_adamw")
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
