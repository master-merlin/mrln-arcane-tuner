"""SophiaG optimizer strategy: the standalone ``Sophia-Optimizer`` package.

The package's ``__init__.py`` raises ``ImportError`` on ``import sophia``, so no
form of that import can succeed. The class lives in ``sophia/sophia.py``, so that
file is loaded by path, after locating the package without importing it.
"""

import importlib.util
import structlog
import torch
from pathlib import Path
from typing import Any

from ..optimizer_base import OptimizerBase

logger = structlog.get_logger(__name__)


def _for_the_trainer(package_class: type) -> type:
    """The package's class, with what the trainer's checkpoints and floats need."""

    class SophiaG(package_class):
        def load_state_dict(self, state_dict: dict) -> None:
            """Keep this optimizer's fresh state over another optimizer's.

            SophiaG jobs used to train with AdamW, so their checkpoints hold AdamW's
            groups, and the first step would raise ``KeyError: 'rho'``. The weights
            resume either way.
            """
            if any("rho" not in group for group in state_dict.get("param_groups", [])):
                logger.warning("sophiag_ignored_other_optimizers_state")
                return
            super().load_state_dict(state_dict)

        def step(self, closure=None, bs=5120):
            """The package's capturable path needs a tensor lr; the scheduler owns a float."""
            if not self.defaults["capturable"]:
                return super().step(closure, bs)
            floats = [group["lr"] for group in self.param_groups]
            for group in self.param_groups:
                group["lr"] = torch.as_tensor(group["lr"], device=group["params"][0].device)
            try:
                return super().step(closure, bs)
            finally:
                for group, lr in zip(self.param_groups, floats):
                    group["lr"] = lr

    return SophiaG


def _load_sophia_class():
    """Load SophiaG from ``sophia/sophia.py``, or raise naming it.

    Never another optimizer in its place: a job that selected SophiaG and
    trained with AdamW would look like a SophiaG result.
    """
    spec = importlib.util.find_spec("sophia")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "SophiaG needs the Sophia-Optimizer package, which is not installed. "
            "Install it, or choose another optimizer."
        )
    sophia_file = Path(spec.origin).parent / "sophia.py"
    try:
        file_spec = importlib.util.spec_from_file_location(
            "sophia_optimizer_file", sophia_file
        )
        module = importlib.util.module_from_spec(file_spec)
        file_spec.loader.exec_module(module)
        return _for_the_trainer(module.SophiaG)
    except (ImportError, OSError, AttributeError) as e:
        raise RuntimeError(
            f"SophiaG could not be loaded from {sophia_file}: {e}. "
            "Reinstall Sophia-Optimizer, or choose another optimizer."
        ) from e


class SophiaGStrategy(OptimizerBase):
    """SophiaG (Second-order Clipped Stochastic Optimization) optimizer.

    Uses the standalone ``Sophia-Optimizer`` PyPI package.
    Parameter order quirk: ``(params, lr, betas, rho, weight_decay, update_period)``.
    """

    def create_optimizer(
        self,
        params: Any,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        config: dict[str, Any],
    ) -> Any:
        rho = float(config.get("sophia_rho", 0.04))
        maximize = bool(config.get("sophia_maximize", False))
        capturable = bool(config.get("sophia_capturable", False))

        logger.info(
            "creating_sophiag",
            lr=lr,
            rho=rho,
            betas=betas,
            weight_decay=weight_decay,
            maximize=maximize,
            capturable=capturable,
        )

        SophiaG = _load_sophia_class()
        return SophiaG(
            params,
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay,
            maximize=maximize,
            capturable=capturable,
        )
