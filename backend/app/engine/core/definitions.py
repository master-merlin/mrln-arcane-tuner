from pydantic import BaseModel, Field
from typing import Any
from abc import ABC

class ModelComponent(BaseModel):
    """
    Defines a component of the model (e.g., UNet, VAE) and its source.
    """
    path: str
    type: str = "diffusers" # diffusers, checkpoint, safetensors
    params: dict[str, Any] = Field(default_factory=dict)

class ModelDefinition(BaseModel):
    """
    Data contract for a model definition YAML file.
    Introspection fields are auto-populated on first model load.
    """
    id: str
    family: str  # Link to ModelFamily registry key
    name: str
    version: str = "1.0"
    defaults: dict[str, Any] = Field(default_factory=dict)
    components: dict[str, ModelComponent] = Field(default_factory=dict)

    # Number of paired control images this model conditions on (0 = standard
    # text-to-image; 1 = FLUX.1 Kontext / Qwen-Image-Edit; up to 3 for
    # multi-reference edit variants). Drives the edit-dataset capability
    # surface, field gating, and run-config validation.
    control_inputs: int = Field(
        0, description="Paired control-image inputs (0 = standard T2I)"
    )

    # --- Introspection fields (auto-enriched) ---
    detected_precision: dict[str, str] = Field(
        default_factory=dict,
        description="Per-component detected dtype, e.g. {'unet': 'torch.float16'}"
    )
    architecture_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted architecture parameters"
    )
    model_size_mb: dict[str, float] = Field(
        default_factory=dict,
        description="Per-component native-precision size on disk in MB, e.g. {'transformer': 64400}"
    )
    lora_targetable_modules: list[str] = Field(
        default_factory=list,
        description="Module names eligible for LoRA injection"
    )
    block_topology: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Block groups for VRAM management UI (populated by enrichment)"
    )
    enrich_pinned_keys: list[str] = Field(
        default_factory=list,
        description=(
            "architecture_params keys whose hand-written YAML value must "
            "survive enrichment (deliberate divergence from the checkpoint's "
            "own config, e.g. a model card's recommended scheduler.shift)"
        ),
    )

class ModelFamily(ABC):
    """
    Abstract Base Class for a Model Logic Provider.
    """
    family_id: str
    # Structured caption format id for this family ("plain" = flat text/tags).
    caption_format: str = "plain"

    def __init__(self, definition: ModelDefinition, config: dict[str, Any]):
        self.definition = definition
        self.config = config

    @property
    def tokenizer_count(self) -> int:
        return 1

    @property
    def text_encoder_count(self) -> int:
        return 1
