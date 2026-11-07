import torch
import structlog
from typing import Any
from ..optimizer_base import OptimizerBase

logger = structlog.get_logger(__name__)

_STEPPING_CLASS: type | None = None


def _stepping_device(params: list[torch.Tensor]) -> torch.device:
    """The device of the first param that steps, or of the first param when none does."""
    for param in params:
        if param.grad is not None:
            return param.device
    return params[0].device


def _stepping_prodigy_class() -> type:
    """ProdigyPlusScheduleFree, with its running d kept on the device of the params that step.

    The library keeps its running d numerator and denominator on ``group['params'][0]``'s
    device. In a dual-expert swap-mode run that is the first expert, which goes to the CPU
    while the second one steps on the card, and the step then mixes devices.

    It keeps the library's class name, which checkpoints record. The import runs on
    every call, so a missing package still raises ImportError here.
    """
    global _STEPPING_CLASS
    from prodigyplus import ProdigyPlusScheduleFree as library_class

    if _STEPPING_CLASS is None or library_class not in _STEPPING_CLASS.__mro__:

        class ProdigyPlusScheduleFree(library_class):
            @torch.no_grad()
            def get_running_values_for_group(self, group):
                if group["split_groups"]:
                    params = group["params"]
                else:
                    group = self.param_groups[0]
                    params = [param for each in self.param_groups for param in each["params"]]
                device = _stepping_device(params)
                numerator, denom = group["running_d_numerator"], group["running_d_denom"]
                if numerator.device != device:
                    group["running_d_numerator"] = numerator = numerator.to(device)
                if denom.device != device:
                    group["running_d_denom"] = denom = denom.to(device)
                return numerator, denom

        _STEPPING_CLASS = ProdigyPlusScheduleFree
    return _STEPPING_CLASS

class ProdigyPlusSFStrategy(OptimizerBase):
    """ProdigyPlus ScheduleFree optimizer strategy."""

    def kept_state_dtypes(self, optimizer: Any, param: torch.Tensor, group: dict[str, Any]) -> dict[str, torch.dtype]:
        # As the library makes them: Prodigy's p0 and s at bf16 on a float32 param, else at
        # the param's dtype; the factored variance at float32 when factored_fp32, the default.
        sliced = torch.bfloat16 if param.dtype == torch.float32 else param.dtype
        factored = torch.float32 if group.get("factored_fp32", True) else param.dtype
        return {"p0": sliced, "s": sliced, "exp_avg_sq_row": factored, "exp_avg_sq_col": factored}

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
            optimizer_class = _stepping_prodigy_class()
            logger.info(
                "creating_prodigy_plus_sf",
                lr=lr, d_coef=ppsf_d_coef,
                betas=ppsf_betas, prodigy_steps=ppsf_prodigy_steps,
                factored=ppsf_factored, stableadamw=ppsf_use_stableadamw,
            )
            return optimizer_class(
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
