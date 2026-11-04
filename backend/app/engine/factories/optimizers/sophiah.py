import structlog
import torch
from typing import Any
from ..optimizer_base import OptimizerBase

logger = structlog.get_logger(__name__)

# A period no run reaches, for the library's own estimate: it differentiates the
# gradients, and the trainer builds no graph for that. The trainer passes its own
# estimate to step() instead, see core/pipeline/hessian_estimate.py.
_LIBRARY_ESTIMATE_OFF = 2**62


def _trainer_sophiah(library_class: type) -> type:
    """The library's class, taking the trainer's estimate parameter by parameter."""

    class SophiaH(library_class):
        @staticmethod
        def set_hessian(param_groups, state, hessian) -> None:
            """Skip a ``None`` entry, and start a parameter's state with its estimate.

            The library needs an entry for every parameter, and storing one in empty
            state makes its ``init_group`` skip the rest, so the update raises
            ``KeyError``. Touching an unreached parameter would also fix its state on
            the device it has now: a dual-expert family's inactive expert is on the
            CPU until it is swapped in.
            """
            params = [p for group in param_groups for p in group["params"]]
            for p, estimate in zip(params, hessian):
                if estimate is None:
                    continue
                entry = state[p]
                if not entry:
                    entry["momentum"] = torch.zeros_like(p)
                    entry["hessian_moment"] = torch.zeros_like(p)
                entry["hessian"] = estimate

    return SophiaH


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
            
            optimizer_class = _trainer_sophiah(load_optimizer("sophiah"))
            optimizer = optimizer_class(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                rho=rho,
                p=p,
                update_period=_LIBRARY_ESTIMATE_OFF,
                num_samples=num_samples,
                hessian_distribution=hessian_dist
            )
            # What the training loop reads to supply the estimate.
            optimizer.hessian_period = update_period
            optimizer.hessian_num_samples = num_samples
            optimizer.hessian_distribution = hessian_dist
            return optimizer

        except ImportError:
            logger.error("pytorch_optimizer_library_missing_for_sophiah")
            raise RuntimeError("pytorch_optimizer is required for SophiaH. Please install it.")
