"""
Pipeline Base Mixin — abstract hooks, setup, and component wiring.

Defines the family-specific hooks that subclasses must implement
and the shared setup / component access helpers.
"""

from abc import abstractmethod
from typing import Any

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.engine.strategies.ema import EMAHandler
from app.engine.core.interfaces import BaseTrainer

logger = structlog.get_logger(__name__)


class PipelineBaseMixin(BaseTrainer):
    """Abstract hooks + component wiring for the training pipeline."""

    # ── Family Hooks (abstract) ──────────────────────────────────────────

    def init_scheduler(self) -> Any:
        """Create and return the noise scheduler for this architecture.

        Default returns ``None`` (flow-matching families need no external
        scheduler).  Override for DDPM-based families (e.g. SDXL) that
        require a diffusers scheduler with ``alphas_cumprod``.

        Returns:
            A scheduler object with an ``add_noise`` method, or ``None``.
        """
        return None

    def get_lora_targets(self) -> list[str]:
        """Return LoRA target module names.  Delegates to the family driver."""
        return self.driver.get_lora_targets()

    def get_lora_exclude_modules(self) -> str | list[str] | None:
        """Return LoRA exclusion patterns.  Delegates to the family driver."""
        return self.driver.get_lora_exclude_modules()

    def get_te_lora_targets(self) -> list[str]:
        """Return LoRA targets for text encoders.  Delegates to the family driver."""
        return self.driver.get_te_lora_targets()

    def encode_text(
        self, captions: list[str], dtype: torch.dtype, batch: dict | None = None
    ) -> Any:
        """Encode captions into text embeddings.  Delegates to the family driver.

        ``batch`` is accepted (and ignored here) so paired-edit trainers can
        derive control-image-aware embeddings + composite cache keys; the
        shared training loop always passes it. Backward compatible.

        Returns ``None`` when the driver reports no text encoders (e.g.
        pixel-space families like HiDream-O1 that handle text encoding
        inside their ``forward_pass``).
        """
        # Task 10: tolerate no-TE families (e.g. HiDream-O1 pixel-space).
        # If the driver declares no text encoders, text encoding is handled
        # inside forward_pass — return None so the training loop passes it through.
        if not self.driver.get_text_encoders():
            return None
        return self.driver.encode_text(captions, dtype)

    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Run the denoising model forward pass.  Delegates to the family driver."""
        return self.driver.forward_pass(noisy_input, timesteps, text_embeddings, batch)

    def compute_target(
        self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Compute the training target for the loss function.

        Default: flow-matching velocity ``noise - latents``.
        Override for epsilon-prediction (SDXL): ``return noise``.
        """
        return noise - latents

    def sample_timesteps(
        self, batch_size: int, latents: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample timesteps for this batch.

        Default: flow-matching continuous [0,1] via ``TimestepSampler``.
        Override for discrete schedulers (e.g. SDXL uses uniform [0, N)).

        Returns:
            Timestep tensor on ``self.device``.
        """
        from app.engine.strategies.timestep_sampling import TimestepSampler

        mode = self.config.get("timestep_sampling", "logit_normal")
        max_steps = getattr(self, "max_train_steps", 1)
        progress = getattr(self, "global_step", 0) / max(max_steps, 1)
        return TimestepSampler.sample_scaled(
            mode, batch_size, self.device, self.config,
            latents=latents, progress=progress,
        )

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to latents at the given timesteps.

        Delegates to the shared :class:`NoiseInterpolation` component.
        Override only for models needing scheduler-specific logic.
        """
        return self.noise_interpolation.add_noise(latents, noise, timesteps)

    def prepare_noise_for_training(self, noise: torch.Tensor) -> torch.Tensor:
        """Prepare noise for the forward diffusion process.

        Delegates to ``driver.prepare_noise`` which by default calls
        ``prepare_latents``.  Override for families where noise must
        live in a different space (e.g. Flux2 where latents are
        BN-normalized but noise must stay raw N(0,1)).
        """
        return self.driver.prepare_noise(noise)

    def compute_loss_weight(self, timesteps: torch.Tensor) -> torch.Tensor | None:
        """Optionally return per-sample loss weights (e.g. Min-SNR gamma).

        Default implementation returns ``None`` (uniform weighting).
        Override in family trainers that support SNR-based weighting.

        Args:
            timesteps: The sampled timesteps for this batch.

        Returns:
            Weight tensor [B] or None.
        """
        return None

    def _compute_step_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        batch: dict[str, Any],
        grad_accum: int,
    ) -> torch.Tensor:
        """Compute loss for one gradient-accumulation step.

        Default: weighted MSE between ``pred`` and ``target``, scaled by
        ``1/grad_accum``.  Override in family trainers that bypass the
        latent/noise pipeline and compute their own recipe loss (e.g.
        HiDream-O1's pixel-space velocity loss computed inside
        ``forward_pass`` / ``compute_loss``).

        Args:
            pred: Model prediction from ``forward_pass``.
            target: Training target from ``compute_target``.
            timesteps: Sampled timesteps.
            batch: Full batch dict.
            grad_accum: Gradient accumulation steps (used for scaling).

        Returns:
            Scalar loss tensor, already divided by ``grad_accum``.
        """
        loss_weight = self.compute_loss_weight(timesteps)
        if loss_weight is not None:
            loss = F.mse_loss(pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * loss_weight
            loss = loss.mean()
        else:
            loss = F.mse_loss(pred.float(), target.float())

        return loss / grad_accum

    def build_batch_extra(self, items: list[dict]) -> dict[str, Any]:
        """Add family-specific data to the batch dict.

        Override in families that need extra conditioning (e.g. SDXL time_ids).
        Default returns empty dict.
        """
        return {}

    def prepare_latents_for_training(self, latents: torch.Tensor) -> torch.Tensor:
        """Transform latents before noise is added.

        Delegates to ``driver.prepare_latents`` for family-specific
        packing (Flux) or reshaping.
        """
        return self.driver.prepare_latents(latents)

    def on_epoch_end(self, epoch: int) -> None:
        """Hook called at the end of each virtual epoch.

        Override for family-specific epoch-end work.
        Default is a no-op — TE unloading is handled in the init
        sequence via ``_pre_cache_text_embeddings`` → ``_offload_text_encoders``.
        """

    def _create_sampler(self):
        """Create a family-specific sampler for generating images during training.

        Returns ``None`` if sampling is disabled or unsupported.
        Override in families that support sampling.
        """
        return None

    def get_te_cache(self) -> dict[str, dict[str, torch.Tensor]] | None:
        """Return text embedding caches for checkpoint persistence.

        Default returns ``{"te": self.text_cache}`` if ``text_cache`` is
        non-empty.  Override in families with additional caches (e.g.
        Flux1 stores ``t5`` + ``clip_pooled``, SDXL stores ``prompt`` +
        ``pooled``).
        """
        if hasattr(self, "text_cache") and self.text_cache:
            return {"te": dict(self.text_cache)}
        return None

    def set_te_cache(self, caches: dict[str, dict[str, torch.Tensor]]) -> None:
        """Restore text embedding caches from a loaded checkpoint.

        Default restores ``caches["te"]`` into ``self.text_cache``.
        Also checks ``"t5"`` as a fallback for backward compatibility
        with older Flux2 checkpoints.
        Override in families with additional caches.
        """
        te_data = caches.get("te") or caches.get("t5")
        if te_data:
            self.text_cache = te_data
            self.logger.info(
                "te_cache_restored",
                entries=len(self.text_cache),
            )

    # ── Setup ────────────────────────────────────────────────────────────

    async def setup(self):
        """Initialize loader, saver, and family state."""
        self.logger.info("setting_up_pipeline", family=self.__class__.__name__)
        self.ema_handler: EMAHandler | None = None
        self.text_cache: dict[str, torch.Tensor] = {}
        self._te_unloaded = False
        self._setup_family()

    @abstractmethod
    def _setup_family(self) -> None:
        """Set ``self.loader`` and ``self.saver`` for this family."""

    # ── Component Accessors ──────────────────────────────────────────────

    def _resolve_loading_dtype(self) -> torch.dtype:
        """Resolve the dtype for initial model loading.

        SDXL loads in fp32 (AMP GradScaler requires fp32 params/grads).
        Flux loads in bf16 (always uses bf16 autocast, no scaler).
        Families can override.
        """
        prec = self.config.get("mixed_precision", "fp16")
        if prec == "bf16":
            return torch.bfloat16
        return torch.float32

    def _assign_components(self) -> None:
        """Wire loaded components to the driver and set trainer aliases.

        The driver is the single owner of component references.
        Trainer-level aliases (``self.transformer``, ``self.vae``) are
        kept for backward compatibility with pipeline mixins that
        access them directly (e.g. ``_freeze_all``, ``_offload_vae``).

        Family trainers may override to add extra setup (e.g. caching
        architecture params) but **must** call ``super()._assign_components()``.
        """
        self.driver.assign_components(self.components)

        # Set common trainer aliases from driver — pipeline mixins use these.
        # Skip attributes defined as @property on the subclass (e.g.
        # QwenImageTrainer.transformer is a read-only property).
        for attr in ("transformer", "vae", "text_encoder", "tokenizer"):
            if isinstance(getattr(type(self), attr, None), property):
                continue
            setattr(self, attr, getattr(self.driver, attr, None))

    def _get_primary_model(self) -> nn.Module:
        """Return the primary trainable model.  Delegates to the family driver."""
        return self.driver.get_primary_model()

    def _get_text_encoders(self) -> dict[str, nn.Module]:
        """Return text encoder(s) as ``{name: module}``.  Delegates to the family driver."""
        return self.driver.get_text_encoders()

    def _freeze_all(self) -> None:
        """Freeze every component."""
        model = self._get_primary_model()
        if model is not None:
            model.requires_grad_(False)
        vae = self.components.get("vae")
        if vae is not None:
            vae.requires_grad_(False)
        for te in self._get_text_encoders().values():
            te.requires_grad_(False)
        self.logger.info("all_components_frozen")

    def _apply_quantization(self) -> None:
        """Apply quantization to frozen components (Unet, TEs).
        
        Called implicitly by the pipeline after setup but before training loops.
        """
        from app.engine.factories.quantization import QuantizationFactory
        
        scheme = self.config.get("quantization", "none")
        backend = self.config.get("quantization_backend", "auto")
        if scheme in ("none", "bf16"):
            return
            
        self.logger.info("applying_quantization", scheme=scheme)
        
        # Quantize Primary Model
        model = self._get_primary_model()
        if model is not None and not next(model.parameters()).requires_grad:
            quantized_model = QuantizationFactory.quantize(model, scheme, backend_name=backend, device=str(self.device))
            self._update_primary_model(quantized_model)
            
        # Quantize Text Encoders
        te_quant_scheme = self.config.get("te_quantization", "none")
        te_backend = self.config.get("te_quantization_backend", "auto")
        if te_quant_scheme not in ("none", "bf16"):
            for name, te in self._get_text_encoders().items():
                if not next(te.parameters()).requires_grad:
                    quantized_te = QuantizationFactory.quantize(te, te_quant_scheme, backend_name=te_backend, device=str(self.device))
                    self.components[name] = quantized_te
                    if hasattr(self, name):
                        setattr(self, name, quantized_te)

    def _update_primary_model(self, new_model: nn.Module) -> None:
        """Update primary model reference after quantization or PEFT wrapping."""
        self.components["unet"] = new_model
        # Subclass may also set self.model, self.unet, etc.
        if hasattr(self, "model"):
            self.model = new_model
        if hasattr(self, "unet"):
            self.unet = new_model
