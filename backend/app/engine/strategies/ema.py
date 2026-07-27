from __future__ import annotations

import torch.nn as nn
import structlog

logger = structlog.get_logger(__name__)


class EMAHandler:
    """
    Exponential Moving Average (EMA) Handler.
    Maintains a shadow copy of model parameters and provides methods to:
    - Step/Update the shadow copy
    - Store original weights (swap preparation)
    - Copy shadow weights to model (swap execution)
    - Restore original weights (swap reversal)

    Accepts either an ``nn.Module`` (the historical contract — shadows
    ``model.named_parameters()``) or an explicit ``dict[str, nn.Parameter]``
    mapping (the W3.T10 dual-expert seam: a caller-built union of BOTH
    experts' trainable params, name-prefixed so keys stay unique across
    experts — see ``GenericTrainingPipeline._ema_parameters``). The dict path
    has no backing single ``nn.Module`` to re-query, so the mapping is
    captured once at construction and reused for the handler's lifetime —
    correct as long as the SAME ``Parameter`` objects (not copies) stay live,
    which holds for LoRA adapter params collected once after PEFT wrapping.
    """

    def __init__(
        self,
        model_or_params: nn.Module | dict[str, nn.Parameter],
        decay: float = 0.999,
    ):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._step_count = 0

        if isinstance(model_or_params, nn.Module):
            self.model: nn.Module | None = model_or_params
            self._params: dict[str, nn.Parameter] | None = None
        else:
            self.model = None
            self._params = dict(model_or_params)

        # Initialize shadow weights from trainable params only
        for name, param in self._named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

        logger.info("ema_initialized", decay=decay, shadow_params=len(self.shadow))

    def _named_parameters(self):
        """Yield ``(name, param)`` from whichever source this handler is bound to."""
        if self._params is not None:
            return self._params.items()
        return self.model.named_parameters()

    def step(self):
        """
        Update shadow weights based on current model weights.
        Should be called after optimizer.step().
        """
        for name, param in self._named_parameters():
            if name in self.shadow:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name].copy_(new_average)

        self._step_count += 1
        if self._step_count % 100 == 0:
            # Log shadow weight stats periodically
            shadow_norms = [v.norm().item() for v in self.shadow.values()]
            if shadow_norms:
                logger.debug(
                    "ema_step",
                    step_count=self._step_count,
                    decay=self.decay,
                    shadow_mean_norm=round(sum(shadow_norms) / len(shadow_norms), 4),
                )

    def store_and_swap(self):
        """
        Backup current weights and load EMA weights into the model.
        Useful before validation or saving checkpoints.

        Only the parameters tracked in ``self.shadow`` (trainable / LoRA
        adapter weights) are backed up — those are the only ones the
        swap will overwrite. Backing up the frozen base model is a pure
        waste of VRAM (≈ base-model size) that on 20B-class transformers
        like Qwen-Image pushes the sampling peak past the consumer-card
        ceiling and triggers WDDM shared-memory fallback, leaving
        training I/O-bound on system RAM after the swap is undone.
        """
        param_dict = dict(self._named_parameters())
        self.backup = {
            name: param_dict[name].data.clone()
            for name in self.shadow
            if name in param_dict
        }
        for name, shadow_data in self.shadow.items():
            if name in param_dict:
                param_dict[name].data.copy_(shadow_data)

        logger.debug("ema_swapped", direction="shadow_to_model")

    def restore(self):
        """Restore original weights from backup."""
        if not self.backup:
            return

        param_dict = dict(self._named_parameters())
        for name, backup_data in self.backup.items():
            param = param_dict.get(name)
            if param is not None:
                param.data.copy_(backup_data)
        self.backup = {}
        logger.debug("ema_restored", direction="backup_to_model")

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        # Move shadow tensors to each parameter's current device. Checkpoints
        # are saved with map_location="cpu" so without this the first step()
        # after resume mixes CPU shadow with CUDA params and crashes.
        param_devices = {name: p.device for name, p in self._named_parameters()}
        self.shadow = {
            name: (t.to(param_devices[name]) if name in param_devices else t)
            for name, t in state_dict.items()
        }
        logger.debug("ema_state_loaded", shadow_params=len(self.shadow))
