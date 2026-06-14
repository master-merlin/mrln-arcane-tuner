"""Core training pipeline interfaces.

Defines the four composable interfaces that form the training lifecycle:

    Load → Train → Sample → Save
     │       │        │        │
     ▼       ▼        ▼        ▼
  IModel  IModel   IModel   IModel
  Loader  Driver   Sampler  Saver

Each model family implements these interfaces independently.
The GenericTrainer composes all four — they can be tested,
swapped, and extended in isolation.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import structlog
import torch
import torch.nn as nn

from app.engine.core.definitions import ModelDefinition


# ---------------------------------------------------------------------------
# IModelLoader — Weight I/O, path resolution, from_pretrained
# ---------------------------------------------------------------------------

class IModelLoader(ABC):
    """Strategy interface for loading model weights.

    Subclasses may append diagnostic messages to ``self.warnings``
    during :meth:`load`.  The pipeline reads these after loading and
    surfaces them to the user via ``job_warning`` WebSocket events.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.warnings: list[str] = []

    @abstractmethod
    async def load(
        self,
        definition: ModelDefinition,
        torch_dtype: torch.dtype | None = None,
        initial_device: str | None = None,
    ) -> dict[str, Any]:
        """Load model components and return them as a dictionary.

        Keys should match standard names: ``"unet"``, ``"vae"``,
        ``"text_encoder"``, ``"tokenizer"``, etc.

        Args:
            definition: Model definition with component paths.
            torch_dtype: Global dtype override (per-component overrides
                take precedence).
            initial_device: Device to place torch models on after
                loading.  ``None`` defaults to ``self.device``.
                Pass ``"cpu"`` for phased loading.

        Returns:
            Dict of loaded components keyed by standard names.
        """

    @abstractmethod
    def get_component_manifest(
        self, definition: ModelDefinition,
    ) -> list[Any]:
        """Declare what components this family needs.

        Returns a list of ``ComponentSpec`` objects describing each
        component to load: HF class, subfolder, dtype, etc.
        """


# ---------------------------------------------------------------------------
# IModelSaver — LoRA extraction, safetensors serialization, metadata
# ---------------------------------------------------------------------------

class IModelSaver(ABC):
    """Strategy interface for saving trained weights."""

    @abstractmethod
    def save(
        self,
        components: dict[str, Any],
        path: Path,
        metadata: dict[str, Any],
    ) -> None:
        """Save the trainable components to the specified path.

        Args:
            components: Dict of model components (e.g. PEFT-wrapped model).
            path: Output directory or file path.
            metadata: Extra metadata (``ss_*`` fields, training config).
        """


# ---------------------------------------------------------------------------
# IModelDriver — Family-specific training behavior
# ---------------------------------------------------------------------------

class IModelDriver(ABC):
    """Family-specific training behavior contract.

    Each model family (SDXL, Flux1, Flux2, QwenImage, ZImage)
    implements this interface. The GenericTrainer composes a driver
    rather than inheriting from family-specific mixins.

    Methods are introduced phase-by-phase:
    - **Phase 1**: Loading (assign_components, get_components, etc.)
    - **Phase 2**: Text encoding
    - **Phase 5**: Training (compute_loss, forward_pass, prepare_latents)
    """

    definition: ModelDefinition
    device: torch.device

    # --- Phase 1: Loading & Component Access ---

    @abstractmethod
    def assign_components(self, components: dict[str, Any]) -> None:
        """Wire loaded components to driver state.

        Extracts components from the loader output into driver
        attributes, caches architecture params (guidance, TE max
        length, model type), and performs family-specific init
        (e.g. guidance embedder zeroing).
        """

    @abstractmethod
    def get_components(self) -> dict[str, Any]:
        """Return currently assigned components for external access."""

    @abstractmethod
    def get_primary_model(self) -> nn.Module:
        """Return the primary trainable model (UNet / Transformer)."""

    @abstractmethod
    def get_text_encoders(self) -> dict[str, nn.Module]:
        """Return text encoder(s) as ``{name: module}`` dict.

        Returns an empty dict if the family has no text encoders
        (e.g. after offloading).
        """

    @abstractmethod
    def get_lora_targets(self) -> list[str]:
        """Return LoRA-targetable module name patterns.

        These are passed to ``peft.LoraConfig.target_modules``.
        """

    def get_lora_exclude_modules(self) -> str | list[str] | None:
        """Return module patterns to exclude from LoRA targeting.

        Passed to ``peft.LoraConfig.exclude_modules``.  Useful when a
        suffix in ``get_lora_targets()`` matches both valid ``Linear``
        modules and unsupported container modules (e.g. ``ModuleList``).

        Default: ``None`` (no exclusions).
        """
        return None

    @abstractmethod
    def init_scheduler(self) -> Any:
        """Create and return the noise scheduler for this family."""

    @abstractmethod
    def resolve_loading_dtype(self) -> torch.dtype:
        """Determine the dtype for initial model loading.

        SDXL loads in fp32 (AMP GradScaler requires fp32 params).
        Flux/Klein loads in bf16 (always uses bf16 autocast).
        """

    # --- Phase 2: Text Encoding ---

    @abstractmethod
    def encode_text(
        self,
        captions: list[str],
        dtype: torch.dtype,
    ) -> Any:
        """Encode a batch of captions into text embeddings.

        Pure encoding — no caching logic.  The pipeline's
        ``TextEncodingCache`` handles caching and TE offload.

        Args:
            captions: Processed captions (trigger words / dropout
                already applied by the pipeline).
            dtype: Target dtype for the returned tensors.

        Returns:
            ``TextEncoderOutput`` with family-appropriate fields.
        """

    # --- Phase 4: Precision, LoRA Targets & Layer Manifest ---

    @abstractmethod
    def get_te_lora_targets(self) -> list[str]:
        """Return LoRA target module patterns for text encoders.

        Return an empty list if text encoder training is not
        supported by this family.
        """

    def get_layer_manifest(self) -> Any:
        """Return a :class:`ModelLayerManifest` describing the model topology.

        Default implementation builds a minimal manifest from
        ``get_lora_targets()`` and ``get_te_lora_targets()``.
        Override in drivers that need block-level topology for
        block swapping or targeted training.
        """
        from app.engine.core.layer_manifest import ModelLayerManifest

        return ModelLayerManifest(
            lora_targets=self.get_lora_targets(),
            te_lora_targets=self.get_te_lora_targets(),
        )

    def get_precision_spec(
        self, mixed_precision: str, *, is_adaptive_optimizer: bool = False,
    ) -> Any:
        """Return a :class:`PrecisionSpec` for this family.

        Default delegates to ``PrecisionSpec.from_config()``, passing the
        loaded primary model's dtype so autocast follows the actual
        weights when they are bf16/fp16 (audit R-TENSOR-10).  Override
        in families with non-standard precision requirements.
        """
        from app.engine.core.layer_manifest import PrecisionSpec

        # Inspect the loaded primary model so autocast follows the actual
        # weights, not the config string.  Falls back to config-only when
        # no model is loaded yet (e.g. unit tests).
        model_dtype: torch.dtype | None = None
        try:
            model = self.get_primary_model()
            if model is not None:
                model_dtype = next(model.parameters()).dtype
        except (StopIteration, AttributeError, NotImplementedError):
            model_dtype = None

        return PrecisionSpec.from_config(
            mixed_precision,
            is_adaptive_optimizer=is_adaptive_optimizer,
            model_dtype=model_dtype,
        )

    # --- Phase 5: Training Loop Hooks ---

    @abstractmethod
    def forward_pass(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: Any,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """Run the denoising model forward pass.

        Args:
            noisy_input: Noisy latents (after ``add_noise``).
            timesteps: Sampled timesteps.
            text_embeddings: Output of ``encode_text()``.
            batch: Full batch dict (for extra conditioning).

        Returns:
            Model prediction tensor.
        """

    def compute_target(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the training target for the loss function.

        Default: flow-matching velocity ``noise - latents``.
        Override for epsilon prediction (e.g. SDXL).
        """
        return noise - latents

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        config: dict[str, Any],
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample timesteps for this batch.

        Default: flow-matching continuous ``[0, 1]`` via ``TimestepSampler``.
        Override for discrete schedulers (e.g. SDXL uniform ``[0, N)``).
        """
        from app.engine.strategies.timestep_sampling import TimestepSampler

        mode = config.get("timestep_sampling", "logit_normal")
        return TimestepSampler.sample_scaled(
            mode, batch_size, device, config, latents=latents,
        )

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to latents at the given timesteps.

        Default: linear interpolation ``t * noise + (1 - t) * latents``.
        Override for scheduler-based noise addition (e.g. SDXL DDPM).
        """
        # Reshape timesteps for broadcasting: [B] → [B, 1, ...]
        t = timesteps
        while t.ndim < latents.ndim:
            t = t.unsqueeze(-1)
        return t * noise + (1 - t) * latents

    def prepare_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Transform latents before noise is added.

        Default: identity (no transform).
        Override for packing (Flux) or reshaping.
        """
        return latents

    def prepare_noise(self, noise: torch.Tensor) -> torch.Tensor:
        """Prepare noise for the forward diffusion process.

        Default: delegates to ``prepare_latents``.
        Override when noise must stay in a different space (e.g. Flux2
        where latents are BN-normalized but noise stays raw).
        """
        return self.prepare_latents(noise)

    def on_optimizer_step(self, optimizer_step: int) -> None:
        """Hook called once per completed optimizer step.

        Invoked by the training loop right after ``optimizer.step()`` /
        ``zero_grad`` for the just-finished step ``optimizer_step``. The
        default is a **no-op**, so existing single-model families are entirely
        unaffected. Multi-model families (e.g. WAN 2.2's dual-expert MoE)
        override this to advance their per-step state — e.g. choosing the
        active expert for the next step and swapping it onto the GPU.

        Args:
            optimizer_step: The 0-based index of the optimizer step that just
                completed (equals the training loop's ``global_step``).
        """
        return None

    # --- Phase 6: LoRA Output & Saver ---

    @abstractmethod
    def get_saver(self) -> "IModelSaver":
        """Return the family's LoRA saver instance.

        Used by ``CheckpointManager`` to produce distribution-format
        LoRA files (safetensors) during periodic and final saves.
        """

    def get_save_metadata(self) -> dict[str, str]:
        """Return base metadata for safetensors headers.

        Default: empty dict.  Override to add family-specific fields
        like ``modelspec.architecture`` or ``ss_base_model_version``.
        """
        return {}

    # --- Phase 7: Data Pipeline ---

    def build_batch_extra(self, items: list[dict]) -> dict[str, Any]:
        """Add family-specific data to the training batch.

        Default: empty dict.  Override for families that need extra
        conditioning (e.g. SDXL ``time_ids``).
        """
        return {}

    # --- Phase 8: Checkpoint Resume ---

    def get_te_cache(self) -> dict[str, dict[str, torch.Tensor]] | None:
        """Return text embedding caches for checkpoint persistence.

        Default: ``{"te": self.text_cache}`` if non-empty.
        Override for multi-cache families (Flux1: t5+clip_pooled,
        SDXL: prompt+pooled).
        """
        text_cache = getattr(self, "text_cache", {})
        if text_cache:
            return {"te": dict(text_cache)}
        return None

    def set_te_cache(self, caches: dict[str, dict[str, torch.Tensor]]) -> None:
        """Restore text embedding caches from a loaded checkpoint.

        Default: restores ``caches["te"]`` into ``self.text_cache``.
        Override for multi-cache families.
        """
        te_data = caches.get("te") or caches.get("t5")
        if te_data:
            self.text_cache = te_data
            self.logger.info("te_cache_restored", entries=len(self.text_cache))

    # --- Phase 9: Advanced Memory & Training Features ---

    def get_block_topology(self) -> list[dict[str, Any]]:
        """Return block groups for VRAM management UI.

        Each dict describes one group of swappable transformer blocks::

            {
                "name": "double_blocks",        # display name
                "attr_path": "transformer_blocks",  # model attribute path
                "count": 19,                    # number of blocks in group
                "approx_vram_mb": 640,          # per-block VRAM estimate
            }

        The frontend renders a numeric slider (0..count) per group,
        following kohya_ss patterns (``single_blocks_to_swap``,
        ``double_blocks_to_swap``).

        Default: ``[]`` (no block swapping available).
        """
        return []

    def get_trainable_layer_names(self) -> list[str]:
        """Return all parameter names in the primary model.

        Used by the frontend to render a layer checklist with
        select-all / deselect-all and a name filter textbox.

        Default: lists ``named_parameters()`` from the primary model.
        """
        model = self.get_primary_model()
        if model is None:
            return []
        return [name for name, _ in model.named_parameters()]


# ---------------------------------------------------------------------------
# IDataPipeline — Data pipeline contract (pre-cacher and training)
# ---------------------------------------------------------------------------

class IDataPipeline(ABC):
    """Data pipeline contract for pre-caching and training.

    Provides the inventory of images/captions and creates
    DataLoader instances for iteration.
    """

    inventory: list[dict[str, Any]]

    @abstractmethod
    def create_dataloader(self) -> Any:
        """Create and return a DataLoader for the current dataset."""


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------
# Existing savers use ``ModelSaver`` — keep working.

ModelSaver = IModelSaver


# ---------------------------------------------------------------------------
# BaseTrainer — Abstract base for training pipelines
# ---------------------------------------------------------------------------

class BaseTrainer(ABC):
    """Abstract base class for training pipelines.

    Provides shared state (components, optimizer, epoch/step counters)
    and checkpoint save/load.  Subclassed by ``PipelineBaseMixin``.
    """

    def __init__(self, definition: ModelDefinition, run_config: dict[str, Any]):
        self.definition = definition
        self.config = run_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = structlog.get_logger(self.__class__.__name__)

        # State
        self.components: dict[str, Any] = {}
        self.epoch = 0
        self.global_step = 0
        self.optimizer: torch.optim.Optimizer | None = None

        # Strategies (set by subclass or DI)
        self.loader: IModelLoader | None = None
        self.saver: IModelSaver | None = None

    @abstractmethod
    async def setup(self):
        """Initialize accelerator, seeds, and logging."""

    @abstractmethod
    async def load_model(self):
        """Use self.loader to load components into self.components."""

    @abstractmethod
    async def prepare_data(self):
        """Initialize the DataLoader."""

    @abstractmethod
    async def train(self):
        """Execute the main training loop."""

    def save_state(self, path: str):
        """Save full training state for checkpoint resume."""
        import os
        import traceback
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            comp_states = {}
            for name, comp in self.components.items():
                if isinstance(comp, nn.Module):
                    comp_states[name] = comp.state_dict()

            state = {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "config": self.config,
                "components": comp_states,
            }
            if self.optimizer:
                state["optimizer"] = self.optimizer.state_dict()

            torch.save(state, path)
            self.logger.info("state_saved", path=path)
        except (OSError, RuntimeError) as e:
            self.logger.error(
                "state_save_failed", path=path, error=str(e),
                traceback=traceback.format_exc(),
            )

    def load_state(self, path: str):
        """Load full training state from a checkpoint."""
        try:
            state = torch.load(path, map_location=self.device)

            self.epoch = state.get("epoch", 0)
            self.global_step = state.get("global_step", 0)

            for name, comp_state in state.get("components", {}).items():
                if name in self.components and isinstance(
                    self.components[name], nn.Module,
                ):
                    self.components[name].load_state_dict(comp_state)

            if "optimizer" in state and self.optimizer:
                self.optimizer.load_state_dict(state["optimizer"])

            self.logger.info(
                "state_loaded", path=path, step=self.global_step,
                epoch=self.epoch,
            )
        except (FileNotFoundError, RuntimeError) as e:
            self.logger.error("state_load_failed", path=path, error=str(e))
