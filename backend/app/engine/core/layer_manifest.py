"""Layer manifest — structured model topology for LoRA, block swapping, and targeted training.

Provides :class:`BlockInfo` and :class:`ModelLayerManifest` to describe
a model's trainable topology.  All consumers (PEFT, ``BlockSwappingManager``,
``TargetedLayerManager``) share this manifest to ensure consistent layer knowledge.

Also contains :class:`PrecisionSpec` for family-declared precision strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# PrecisionSpec — family precision strategy
# ---------------------------------------------------------------------------


@dataclass
class PrecisionSpec:
    """Declares how a family prefers to handle mixed-precision training.

    The pipeline's ``_configure_optimization`` reads this instead of
    parsing raw config strings inline.

    Attributes:
        autocast_dtype: Dtype for ``torch.autocast`` context.
        use_amp: Whether to enable autocast at all.
        grad_scaler_enabled: Whether ``GradScaler`` should be active.
            SDXL fp16 needs it; bf16 families don't.
    """

    autocast_dtype: torch.dtype
    use_amp: bool
    grad_scaler_enabled: bool

    @staticmethod
    def from_config(
        mixed_precision: str,
        *,
        is_adaptive_optimizer: bool = False,
        model_dtype: torch.dtype | None = None,
    ) -> PrecisionSpec:
        """Build from the ``mixed_precision`` config string.

        This is the **default** factory used when no driver overrides
        ``get_precision_spec()``.

        Args:
            mixed_precision: One of ``"bf16"``, ``"fp16"``, ``"fp32"``.
                Gates ``use_amp`` and ``grad_scaler_enabled`` flags.
            is_adaptive_optimizer: Prodigy-style adaptive LR — disables
                grad scaler even for fp16.
            model_dtype: Dtype of the actually-loaded primary model
                parameters.  When the model is already in a low-precision
                float (bf16 / fp16), ``autocast_dtype`` follows the
                **model** rather than the config string — autocasting
                bf16 weights through an fp16 context silently re-promotes
                every op and accumulates precision drift (audit
                R-TENSOR-10, symmetrical to the sampler fix in 287b840).
                When the model is in fp32, the config string is honored
                so that genuine AMP training (e.g. SDXL fp32 params +
                fp16 autocast + GradScaler) keeps working.  ``None``
                preserves the legacy config-only behavior for callers
                that have no model handle (tests, factories run before
                load).
        """
        # Resolve autocast dtype: model wins when it's already low-precision,
        # config wins when model is fp32 (legitimate AMP) or unknown.
        def _resolve_autocast(default: torch.dtype) -> torch.dtype:
            if model_dtype in (torch.bfloat16, torch.float16):
                return model_dtype
            return default

        if mixed_precision == "bf16":
            return PrecisionSpec(
                autocast_dtype=_resolve_autocast(torch.bfloat16),
                use_amp=True,
                grad_scaler_enabled=False,
            )
        if mixed_precision == "fp16":
            return PrecisionSpec(
                autocast_dtype=_resolve_autocast(torch.float16),
                use_amp=True,
                grad_scaler_enabled=not is_adaptive_optimizer,
            )
        # fp32 / anything else
        return PrecisionSpec(
            autocast_dtype=torch.float32,
            use_amp=False,
            grad_scaler_enabled=False,
        )


# ---------------------------------------------------------------------------
# BlockInfo — describes one transformer block
# ---------------------------------------------------------------------------


@dataclass
class BlockInfo:
    """Metadata about a single transformer block.

    Used by ``BlockSwappingManager`` to identify which ``nn.Module``
    instances can be swapped between CPU and GPU.

    Attributes:
        name: Fully qualified ``named_modules`` path,
            e.g. ``"transformer_blocks.5"``.
        block_type: Semantic category — ``"joint"``, ``"single"``,
            ``"down"``, ``"mid"``, ``"up"``.
        param_count: Total parameter count for this block.
        depth_index: Zero-based position in the stack.
    """

    name: str
    block_type: str
    param_count: int
    depth_index: int


# ---------------------------------------------------------------------------
# ModelLayerManifest — rich layer metadata
# ---------------------------------------------------------------------------


@dataclass
class ModelLayerManifest:
    """Rich layer metadata consumed by LoRA, block swapping, and targeted training.

    Each driver's ``get_layer_manifest()`` fills this once, and
    all downstream consumers read from it:

    - **PEFT**: reads ``lora_targets`` / ``te_lora_targets``.
    - **BlockSwappingManager**: reads ``transformer_blocks``.
    - **TargetedLayerManager**: reads ``targetable_modules``.

    Attributes:
        transformer_blocks: Ordered list of block descriptors.
        lora_targets: Module name patterns for ``peft.LoraConfig.target_modules``.
        te_lora_targets: Module name patterns for text encoder LoRA.
        targetable_modules: All named-module paths eligible for regex matching.
    """

    transformer_blocks: list[BlockInfo] = field(default_factory=list)
    lora_targets: list[str] = field(default_factory=list)
    te_lora_targets: list[str] = field(default_factory=list)
    targetable_modules: list[str] = field(default_factory=list)

    @property
    def total_block_params(self) -> int:
        """Sum of parameter counts across all blocks."""
        return sum(b.param_count for b in self.transformer_blocks)

    @property
    def block_count(self) -> int:
        """Number of transformer blocks."""
        return len(self.transformer_blocks)

    def blocks_by_type(self, block_type: str) -> list[BlockInfo]:
        """Filter blocks by type (e.g. ``"joint"``, ``"single"``)."""
        return [b for b in self.transformer_blocks if b.block_type == block_type]
