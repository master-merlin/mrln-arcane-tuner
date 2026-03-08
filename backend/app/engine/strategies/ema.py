
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
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.model = model
        self._step_count = 0
        
        # Initialize shadow weights from trainable params only
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

        logger.info("ema_initialized", decay=decay, shadow_params=len(self.shadow))

    def step(self):
        """
        Update shadow weights based on current model weights.
        Should be called after optimizer.step().
        """
        for name, param in self.model.named_parameters():
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
        """
        self.backup = [p.data.clone() for p in self.model.parameters()]
        
        param_dict = dict(self.model.named_parameters())
        for name, shadow_data in self.shadow.items():
            if name in param_dict:
                param_dict[name].data.copy_(shadow_data)

        logger.debug("ema_swapped", direction="shadow_to_model")

    def restore(self):
        """Restore original weights from backup."""
        if not self.backup:
            return
            
        for param, backup_data in zip(self.model.parameters(), self.backup):
            param.data.copy_(backup_data)
        self.backup = []
        logger.debug("ema_restored", direction="backup_to_model")

    def state_dict(self):
        return self.shadow
        
    def load_state_dict(self, state_dict):
        self.shadow = state_dict
        logger.debug("ema_state_loaded", shadow_params=len(self.shadow))
