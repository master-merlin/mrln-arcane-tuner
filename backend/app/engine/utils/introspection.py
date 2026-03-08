"""
Model introspection utilities.
Extracts precision, layer manifests, architecture params, and LoRA-targetable modules
from loaded model components.
"""
import torch
import torch.nn as nn
import structlog
from pydantic import BaseModel, Field
from typing import Any

logger = structlog.get_logger(__name__)


class LayerInfo(BaseModel):
    """Describes a single named module in the model."""
    name: str
    type: str
    shape: list[int] | None = None
    param_count: int = 0


class IntrospectionResult(BaseModel):
    """Structured result from model introspection."""
    detected_precision: dict[str, str] = Field(
        default_factory=dict,
        description="Per-component detected dtype, e.g. {'unet': 'torch.float16'}"
    )
    architecture_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted architecture parameters (hidden_size, num_heads, etc.)"
    )
    lora_targetable_modules: list[str] = Field(
        default_factory=list,
        description="Module names eligible for LoRA injection (nn.Linear, nn.Conv2d)"
    )
    total_params: int = Field(0, description="Total parameter count of trainable components")
    layer_count: int = Field(0, description="Total number of named modules")


class ModelIntrospector:
    """
    Inspects loaded model components to extract architecture metadata.

    Usage:
        introspector = ModelIntrospector()
        result = introspector.introspect(components)
    """

    # Module types that are valid LoRA targets
    LORA_TARGET_TYPES = (nn.Linear, nn.Conv2d)

    def introspect(self, components: dict[str, Any], component_key: str = "unet") -> IntrospectionResult:
        """
        Introspect loaded model components.

        Args:
            components: Dict of loaded model components (unet, vae, text_encoder, etc.)
            component_key: Primary component to analyze for LoRA targets (default: unet)

        Returns:
            IntrospectionResult with extracted metadata.
        """
        result = IntrospectionResult()

        # 1. Detect precision for each nn.Module component
        for name, comp in components.items():
            if isinstance(comp, nn.Module):
                dtype = self._detect_dtype(comp)
                if dtype:
                    result.detected_precision[name] = str(dtype)

        # 2. Extract architecture params from the primary component
        primary = components.get(component_key)
        if primary and isinstance(primary, nn.Module):
            result.architecture_params = self._extract_architecture(primary)
            result.total_params = sum(p.numel() for p in primary.parameters())
            result.layer_count = sum(1 for _ in primary.named_modules())

            # 3. Find LoRA-targetable modules
            result.lora_targetable_modules = self._find_lora_targets(primary)

        logger.info(
            "introspection_complete",
            precisions=result.detected_precision,
            lora_targets=len(result.lora_targetable_modules),
            total_params=result.total_params,
            arch_params=list(result.architecture_params.keys()),
        )

        return result

    def _detect_dtype(self, module: nn.Module) -> torch.dtype | None:
        """
        Detect the dominant dtype of a module's parameters.
        Returns the dtype of the first parameter found.
        """
        for param in module.parameters():
            return param.dtype
        return None

    def _extract_architecture(self, module: nn.Module) -> dict[str, Any]:
        """
        Extract architecture parameters from a model by inspecting its structure.
        Tries common attribute names used across diffusion model architectures.
        """
        params: dict[str, Any] = {}

        # Common architecture attributes to look for
        attribute_map = {
            "hidden_size": ["hidden_size", "config.hidden_size", "inner_dim"],
            "num_heads": ["num_heads", "config.num_attention_heads", "heads"],
            "depth": ["depth", "config.num_layers", "num_layers"],
            "depth_single_blocks": ["depth_single_blocks", "config.num_single_layers"],
            "in_channels": ["in_channels", "config.in_channels"],
            "context_dim": ["context_in_dim", "config.joint_attention_dim", "cross_attention_dim"],
            "sample_size": ["sample_size", "config.sample_size"],
        }

        for canonical_name, attr_paths in attribute_map.items():
            for attr_path in attr_paths:
                value = self._get_nested_attr(module, attr_path)
                if value is not None and not isinstance(value, nn.Module):
                    params[canonical_name] = value
                    break

        return params

    def _get_nested_attr(self, obj: Any, attr_path: str) -> Any:
        """Safely get a nested attribute like 'config.hidden_size'."""
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            return None

    def _find_lora_targets(self, module: nn.Module) -> list[str]:
        """
        Find all module paths eligible for LoRA injection.
        Returns fully qualified names of nn.Linear and nn.Conv2d modules.
        """
        targets = []
        for name, child in module.named_modules():
            if isinstance(child, self.LORA_TARGET_TYPES) and name:
                targets.append(name)
        return targets
