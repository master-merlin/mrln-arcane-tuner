"""SophiaG optimizer strategy — standalone Sophia-Optimizer package.

Uses importlib workaround for the broken ``__init__.py`` in the
``Sophia-Optimizer`` PyPI package (tries ``from .sophiag import SophiaG``
but the class lives in ``sophia/sophia.py``).
"""

import importlib.util
import structlog
from pathlib import Path
from typing import Any

from ..optimizer_base import OptimizerBase

logger = structlog.get_logger(__name__)


def _load_sophia_class():
    """Import SophiaG, working around the broken package __init__.py.

    The official ``Sophia-Optimizer`` package has a broken ``__init__.py``
    that does ``from .sophiag import SophiaG``, but the file is actually
    ``sophia/sophia.py``.  We catch the ImportError and fall back to
    direct file loading via ``importlib.util.spec_from_file_location``.
    """
    # Try the normal import first (in case the package gets fixed)
    try:
        from sophia import SophiaG  # noqa: F401
        return SophiaG
    except ImportError:
        pass

    # Workaround: locate sophia/sophia.py in site-packages
    try:
        import sophia
        package_dir = Path(sophia.__file__).parent
        sophia_file = package_dir / "sophia.py"

        if not sophia_file.exists():
            raise ImportError(f"Could not find sophia.py in {package_dir}")

        spec = importlib.util.spec_from_file_location("sophia.sophia", sophia_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.SophiaG
    except Exception:
        raise ImportError(
            "Sophia-Optimizer package not found or broken. "
            "Install with: pip install Sophia-Optimizer"
        )


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

        try:
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
        except ImportError:
            logger.warning("sophiag_not_found_fallback_adamw")
            import torch
            return torch.optim.AdamW(
                params, lr=lr, weight_decay=weight_decay, betas=betas,
            )
