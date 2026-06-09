"""Model definition schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class CreateDefinitionRequest(BaseModel):
    """Request body for creating a new model definition."""
    id: str
    family: str
    name: str
    version: str = "1.0"
    defaults: dict[str, Any] = {}
    components: dict[str, dict[str, Any]] = {}


class UpdateDefinitionRequest(BaseModel):
    """Request body for updating an existing model definition."""
    name: str | None = None
    version: str | None = None
    defaults: dict[str, Any] | None = None
    components: dict[str, dict[str, Any]] | None = None


class VRAMEstimateRequest(BaseModel):
    """Request body for VRAM estimation."""
    definition_id: str
    config: dict[str, Any]


# ── Response models ──────────────────────────────────────────────────────


class DeleteDefinitionResponse(BaseModel):
    """Ack for deleting a model definition."""
    status: str = "deleted"
    id: str


class EnrichDefinitionResponse(BaseModel):
    """Ack for triggering enrichment of a model definition."""
    status: str = "enriched"
    id: str
    block_topology: list[Any]
    lora_targetable_modules: list[Any]


class ModelSettingsResponse(BaseModel):
    """Global model settings (offline mode + default storage path).

    ``hf_token_set`` is a masked indicator — the raw Hugging Face token is
    never returned to clients. Set/clear it via PUT's ``hf_token`` field.
    """
    global_offline_mode: bool
    default_model_path: str
    hf_token_set: bool = False


class ModelSourceResponse(BaseModel):
    """Source override for a model definition (mirrors ``ModelOverride``)."""
    source_type: str = "hf_hub"
    local_path: str | None = None
    skip_update: bool = False


class DeleteModelSourceResponse(BaseModel):
    """Ack for removing a model source override."""
    status: str = "removed"
    id: str


class ValidateModelPathResponse(BaseModel):
    """Result of probing a local path for model components."""
    valid: bool = False
    type: str = "unknown"
    components_found: list[str] = []
    warnings: list[str] = []


class VRAMEstimateResponse(BaseModel):
    """Per-category VRAM breakdown + fit assessment (``VRAMReport.to_dict``)."""
    model_weights_mb: float
    lora_adapters_mb: float
    optimizer_states_mb: float
    gradients_mb: float
    activations_mb: float
    overhead_mb: float
    caching_peak_mb: float
    training_peak_mb: float
    peak_mb: float
    available_mb: float
    total_mb: float
    used_mb: float
    fits: bool
    warnings: list[str]


class TrainingEstimateMetric(BaseModel):
    """One calibrated metric block (wall_time / output_size / throughput / disk)."""
    seconds: float | None = None
    bytes: float | None = None
    steps_per_sec: float | None = None
    display: str
    samples: int
    calibrated: bool


class TrainingEstimateResponse(BaseModel):
    """Full data-calibrated training estimate for the Quick Train wall."""
    definition_id: str
    stats_available: bool
    samples: int
    # Epoch-seconds timestamp from definition_stats_service.estimate (NOT a str).
    updated_at: float | None = None
    wall_time: TrainingEstimateMetric
    output_size: TrainingEstimateMetric
    throughput: TrainingEstimateMetric
    disk_footprint: TrainingEstimateMetric
    vram: dict[str, Any] | None = None


class ModelCapabilitiesResponse(BaseModel):
    """Block topology, trainable layers, and archetype capability descriptor."""
    enriched: bool
    block_topology: list[Any]
    lora_targetable_modules: list[Any]
    trainable_layers: list[Any]
    archetype: str
    capabilities: dict[str, Any]
    field_visibility: dict[str, Any]
    defaults: dict[str, Any]
