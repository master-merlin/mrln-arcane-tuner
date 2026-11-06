import torch
from typing import Any
import structlog

from .optimizer_base import OptimizerBase
from .optimizers.adamw import AdamWStrategy
from .optimizers.adamw8bit import AdamW8bitStrategy
from .optimizers.prodigy import ProdigyStrategy
from .optimizers.prodigy_plus_sf import ProdigyPlusSFStrategy
from .optimizers.sophiah import SophiaHStrategy
from .optimizers.sophiag import SophiaGStrategy
from .optimizers.lion import LionStrategy
from .optimizers.adafactor import AdafactorStrategy
from .optimizers.stableadamw import StableAdamWStrategy
from .optimizers.shampoo import ShampooStrategy
from .optimizers.radam import RAdamStrategy
from .optimizers.ademamix import AdEMAMixStrategy

logger = structlog.get_logger(__name__)

class OptimizerFactory:
    """
    Factory for creating PyTorch optimizers via Strategy Pattern.
    Support: AdamW, AdamW8bit, Prodigy, ProdigyPlusSF, SophiaH, SophiaG, Lion, Adafactor, StableAdamW, Shampoo, RAdam, AdEMAMix.
    """
    SUPPORTED_OPTIMIZERS = {
        "AdamW": AdamWStrategy(),
        "AdamW8bit": AdamW8bitStrategy(),
        "Prodigy": ProdigyStrategy(),
        "ProdigyPlusSF": ProdigyPlusSFStrategy(),
        "SophiaH": SophiaHStrategy(),
        "SophiaG": SophiaGStrategy(),
        "Lion": LionStrategy(),
        "Adafactor": AdafactorStrategy(),
        "StableAdamW": StableAdamWStrategy(),
        "Shampoo": ShampooStrategy(),
        "RAdam": RAdamStrategy(),
        "AdEMAMix": AdEMAMixStrategy()
    }
    
    # Optimizers that manage their own LR — no external scheduler allowed
    _ADAPTIVE_OPTIMIZERS = {"Prodigy", "ProdigyPlusSF"}

    @staticmethod
    def is_adaptive(optimizer_type: str, config: dict[str, Any] | None = None) -> bool:
        """Check if optimizer manages its own LR (no external scheduler needed).

        Prodigy and ProdigyPlusSF always self-adapt.
        Adafactor is adaptive ONLY when relative_step is True — otherwise
        it uses the user-provided LR like AdamW.
        """
        if optimizer_type in OptimizerFactory._ADAPTIVE_OPTIMIZERS:
            return True
        if optimizer_type == "Adafactor" and config:
            return bool(config.get("adafactor_relative_step", False))
        return False

    @staticmethod
    def create(
        optimizer_type: str,
        params: Any,
        lr: float,
        weight_decay: float = 0.01,
        betas: tuple[float, float] | None = None,
        config: dict[str, Any] | None = None
    ) -> torch.optim.Optimizer:
        """Create an optimizer instance using the injected strategies.

        Args:
            optimizer_type: One of the supported optimizers (see SUPPORTED_OPTIMIZERS).
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            weight_decay: Weight decay factor.
            betas: Tuple of (beta1, beta2) for Adam-based optimizers.
            config: The raw training dictionary containing all optimizer-specific args.

        Returns:
            Configured optimizer instance.
        """
        adam_betas = betas or (0.9, 0.999)
        cfg = config or {}
        
        strategy: OptimizerBase = OptimizerFactory.SUPPORTED_OPTIMIZERS.get(optimizer_type)
        
        if not strategy:
            logger.info("optimizer_unknown_fallback_adamw", type=optimizer_type)
            strategy = OptimizerFactory.SUPPORTED_OPTIMIZERS["AdamW"]
            
        optimizer = strategy.create_optimizer(params, lr, weight_decay, adam_betas, cfg)
        _load_kept_dtypes_back(optimizer, strategy)
        return optimizer


def _load_kept_dtypes_back(optimizer: torch.optim.Optimizer, strategy: OptimizerBase) -> None:
    """Make ``load_state_dict`` give back state an optimizer keeps at its own dtype at that dtype.

    torch's load casts every floating state tensor but a ``step`` to its param's dtype.
    That is right for state that follows its param, even after a resume onto params
    of another dtype. But state an optimizer keeps at a dtype of its own, like
    Adafactor's bf16 ``exp_avg`` on float32 params, then has a dtype its next step
    refuses. The strategy names that state (``kept_state_dtypes``). Each named
    tensor comes back at its own dtype, on the device torch chose. If it was saved at
    that dtype, the saved tensor itself comes back, because where the kept dtype is
    the wider one, torch's cast has already rounded the values.

    bitsandbytes' optimizers have their own ``load_state_dict``, which runs no hooks,
    and they keep their non-floating state uncast.
    """
    saved: dict[str, Any] = {}

    def keep(_optimizer: torch.optim.Optimizer, state_dict: dict[str, Any]) -> None:
        saved["state"] = state_dict.get("state", {})
        saved["groups"] = [group["params"] for group in state_dict.get("param_groups", [])]

    def restore(optimizer: torch.optim.Optimizer) -> None:
        state, groups = saved.pop("state", {}), saved.pop("groups", [])
        # Matched to params as torch's own load matches them.
        for ids, group in zip(groups, optimizer.param_groups):
            for saved_id, param in zip(ids, group["params"]):
                entry = optimizer.state.get(param)
                if saved_id not in state or entry is None:
                    continue
                for key, dtype in strategy.kept_state_dtypes(optimizer, param, group).items():
                    loaded, before = entry.get(key), state[saved_id].get(key)
                    if not torch.is_tensor(loaded) or not loaded.is_floating_point():
                        continue
                    if torch.is_tensor(before) and before.dtype == dtype:
                        entry[key] = before.to(device=loaded.device)
                    else:
                        entry[key] = loaded.to(dtype)

    optimizer.register_load_state_dict_pre_hook(keep)
    optimizer.register_load_state_dict_post_hook(restore)


class LRSchedulerFactory:
    """
    Factory for creating Learning Rate Schedulers (via transformers).
    Support: constant, linear, cosine.
    """
    SUPPORTED_SCHEDULERS = ["constant", "linear", "cosine"]

    @staticmethod
    def create(
        scheduler_type: str,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        **kwargs
    ) -> Any:
        logger.debug("creating_scheduler", type=scheduler_type, warmup=num_warmup_steps)
        
        try:
            from transformers import (
                get_cosine_schedule_with_warmup, 
                get_constant_schedule_with_warmup, 
                get_linear_schedule_with_warmup
            )
            
            if scheduler_type == "cosine":
                return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
            elif scheduler_type == "linear":
                return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
            elif scheduler_type == "constant":
                return get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
            
            # Default
            return get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
            
        except ImportError:
            logger.error("transformers_library_missing_no_scheduler")
            return None
