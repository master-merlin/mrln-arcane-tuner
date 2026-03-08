import structlog
from typing import Any
from ..optimizer_base import OptimizerBase

logger = structlog.get_logger(__name__)

class AdafactorStrategy(OptimizerBase):
    """Adafactor PyTorch optimizer strategy via pytorch_optimizer.
    Sub-linear memory cost, replaces variance with factored running averages.

    Note on relative_step: When True, Adafactor computes its own LR as
    min(1e-2, 1/sqrt(step)) — this is ~0.01 within the first 100 steps,
    which is 100x too high for LoRA fine-tuning. Default to False so the
    user's explicit LR (typically 1e-4) is used directly.
    """

    def create_optimizer(
        self,
        params: Any,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        config: dict[str, Any]
    ) -> Any:
        relative_step = config.get("adafactor_relative_step", False)
        warmup_init = config.get("adafactor_warmup_init", False)
        clip_threshold = float(config.get("adafactor_clip_threshold", 1.0))
        decay_rate = float(config.get("adafactor_decay_rate", -0.8))

        logger.debug(
            "creating_optimizer", type="Adafactor",
            lr=lr, relative_step=relative_step, warmup_init=warmup_init,
            scale_parameter="auto (relative_step)" if relative_step else False,
        )

        try:
            from pytorch_optimizer import load_optimizer

            optimizer_class = load_optimizer("adafactor")

            # When relative_step is True, Adafactor self-manages its LR via
            # min(1/sqrt(step), rms(params)).  The lr parameter MUST be None
            # and warmup_init/scale_parameter MUST be True per the paper.
            if relative_step:
                effective_lr = None
                warmup_init = True
                scale_parameter = True
            else:
                effective_lr = lr
                scale_parameter = False

            return optimizer_class(
                params,
                lr=effective_lr,
                weight_decay=weight_decay,
                decay_rate=decay_rate,
                clip_threshold=clip_threshold,
                relative_step=relative_step,
                warmup_init=warmup_init,
                scale_parameter=scale_parameter,
            )

        except ImportError:
            logger.error("pytorch_optimizer_library_missing_for_adafactor")
            raise RuntimeError("pytorch_optimizer is required for Adafactor. Please install it.")
