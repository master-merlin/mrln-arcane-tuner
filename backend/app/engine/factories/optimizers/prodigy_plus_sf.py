import torch
import structlog
from typing import Any
from ..optimizer_base import OptimizerBase

logger = structlog.get_logger(__name__)

class ProdigyPlusSFStrategy(OptimizerBase):
    """ProdigyPlus ScheduleFree optimizer strategy."""

    def create_optimizer(
        self,
        params: Any,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        config: dict[str, Any]
    ) -> torch.optim.Optimizer:
        ppsf_betas = betas if betas[1] != 0.999 else (0.9, 0.99)
        
        ppsf_d_coef = float(config.get("ppsf_d_coef", 1.0))
        ppsf_prodigy_steps = int(config.get("ppsf_prodigy_steps", 0))
        ppsf_use_bias_correction = bool(config.get("ppsf_use_bias_correction", False))
        ppsf_use_stableadamw = bool(config.get("ppsf_use_stableadamw", True))
        ppsf_factored = bool(config.get("ppsf_factored", True))
        ppsf_eps = float(config.get("ppsf_eps", 1e-8))
        ppsf_use_cautious = bool(config.get("ppsf_use_cautious", False))
        ppsf_use_grams = bool(config.get("ppsf_use_grams", False))
        ppsf_use_adopt = bool(config.get("ppsf_use_adopt", False))
        ppsf_use_orthograd = bool(config.get("ppsf_use_orthograd", False))
        ppsf_use_focus = bool(config.get("ppsf_use_focus", False))
        ppsf_use_speed = bool(config.get("ppsf_use_speed", False))
        ppsf_split_groups = bool(config.get("ppsf_split_groups", True))

        try:
            from prodigyplus import ProdigyPlusScheduleFree
            logger.info(
                "creating_prodigy_plus_sf",
                lr=lr, d_coef=ppsf_d_coef,
                betas=ppsf_betas, prodigy_steps=ppsf_prodigy_steps,
                factored=ppsf_factored, stableadamw=ppsf_use_stableadamw,
            )
            return ProdigyPlusScheduleFree(
                params,
                lr=lr,
                betas=ppsf_betas,
                weight_decay=weight_decay,
                d_coef=ppsf_d_coef,
                prodigy_steps=ppsf_prodigy_steps,
                eps=ppsf_eps,
                factored=ppsf_factored,
                use_bias_correction=ppsf_use_bias_correction,
                use_stableadamw=ppsf_use_stableadamw,
                split_groups=ppsf_split_groups,
                use_cautious=ppsf_use_cautious,
                use_grams=ppsf_use_grams,
                use_adopt=ppsf_use_adopt,
                use_orthograd=ppsf_use_orthograd,
                use_focus=ppsf_use_focus,
                use_speed=ppsf_use_speed,
            )
        except ImportError:
            logger.warning("prodigy_plus_sf_not_found_fallback_adamw")
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
