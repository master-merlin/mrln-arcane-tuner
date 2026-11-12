import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from abc import ABC

# ── Distillation handling ────────────────────────────────────────────────────
#
# A distilled model (fewer sampling steps, guidance baked in) drifts when it is
# trained with plain flow-matching MSE. A definition that is distilled says so in
# its ``distillation`` block, and the job chooses how training compensates:
#
# * ``adapter``: a frozen training adapter (a LoRA the definition names, pinned by
#   revision and sha256) is live in every training forward pass and off for every
#   preview and validation sample (``core/training_adapter.py``);
# * ``contrastive_guidance``: the target becomes ``u + s*(v - u)``, ``u`` the
#   blank-prompt prediction (``core/contrastive_guidance.py``);
# * ``both``: the two together; ``none``: train as any other model.
#
# A definition without the block is not distilled and trains exactly as before.

DistillationHandling = Literal["adapter", "contrastive_guidance", "both", "none"]
DISTILLATION_HANDLINGS: tuple[str, ...] = get_args(DistillationHandling)
GuidanceSchedule = Literal["constant", "sigma"]
GUIDANCE_SCHEDULES: tuple[str, ...] = get_args(GuidanceSchedule)
DEFAULT_GUIDANCE_SCHEDULE = "constant"
# With both remedies the adapter already pulls toward the distilled outputs, so the
# guidance target starts lower to avoid counting the guidance twice.
BOTH_GUIDANCE_FACTOR = 0.7

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TrainingAdapterSpec(BaseModel):
    """A frozen LoRA a distilled definition trains with, pinned to one file."""

    model_config = ConfigDict(extra="forbid")

    repo: str = Field(..., min_length=1, description="Hugging Face repo id")
    revision: str = Field(..., min_length=1, description="The commit the file is pinned to")
    file: str = Field(..., min_length=1, description="The safetensors file inside the repo")
    sha256: str = Field(..., description="The file's sha256, lowercase hex")

    @field_validator("sha256")
    @classmethod
    def _check_sha256(cls, value: str) -> str:
        if not _SHA256.match(value):
            raise ValueError(f"training_adapter.sha256 must be 64 lowercase hex digits, got {value!r}")
        return value


class DistillationSpec(BaseModel):
    """What a definition's model was distilled for, and which handlings that allows."""

    model_config = ConfigDict(extra="forbid")

    step: bool = Field(..., description="Distilled to fewer sampling steps")
    guidance: bool = Field(..., description="Guidance distilled in (sampled without CFG)")
    guidance_scale: float | None = Field(
        ..., description="The guidance target contrastive guidance trains toward (needs guidance: true)"
    )
    training_adapter: TrainingAdapterSpec | None = Field(..., description="The frozen training adapter, or null")
    default_handling: DistillationHandling = Field(..., description="The handling an unset job field means")

    @model_validator(mode="after")
    def _check(self) -> "DistillationSpec":
        if self.guidance:
            scale = self.guidance_scale
            if scale is None or not math.isfinite(scale) or scale < 1.0:
                raise ValueError(
                    f"distillation.guidance_scale must be a number >= 1 when guidance is true, got {scale!r}"
                )
        elif self.guidance_scale is not None:
            raise ValueError("distillation.guidance_scale is only meaningful with guidance: true")
        allowed = self.allowed_handlings()
        if self.default_handling not in allowed:
            raise ValueError(
                f"distillation.default_handling {self.default_handling!r} is not allowed by this block; "
                f"allowed: {allowed}"
            )
        return self

    def allowed_handlings(self) -> list[str]:
        """``adapter`` with an adapter, ``contrastive_guidance`` with guidance, ``both`` with
        both, and always ``none``, in that order."""
        out = []
        if self.training_adapter is not None:
            out.append("adapter")
        if self.guidance:
            out.append("contrastive_guidance")
        if self.training_adapter is not None and self.guidance:
            out.append("both")
        out.append("none")
        return out

    def default_guidance_scale(self, handling: str) -> float | None:
        """The guidance target a handling starts from: ``guidance_scale`` for contrastive
        guidance alone, 0.7 of it (to 0.1) with the adapter too, ``None`` without guidance."""
        if handling == "contrastive_guidance":
            return self.guidance_scale
        if handling == "both":
            # Decimal, half up: 0.7 x 3.5 is 2.45 exactly, which float rounding takes to 2.4.
            product = Decimal(str(BOTH_GUIDANCE_FACTOR)) * Decimal(str(self.guidance_scale))
            return float(product.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
        return None


@dataclass(frozen=True)
class ResolvedDistillation:
    """A job's distillation handling after the definition's defaults are applied."""

    handling: str
    guidance_scale: float | None
    guidance_schedule: str | None

    @property
    def uses_adapter(self) -> bool:
        return self.handling in ("adapter", "both")

    @property
    def uses_contrastive_guidance(self) -> bool:
        return self.handling in ("contrastive_guidance", "both")


def resolve_distillation(definition: Any, config: Mapping[str, Any]) -> ResolvedDistillation:
    """The job's distillation handling: an unset field means the definition's default.

    Refuses (``ValueError``, naming the field and what is allowed) a handling the
    definition does not allow, a guidance target below 1 or not a number, and an unknown
    schedule. A definition without a ``distillation`` block allows only ``none``. The
    guidance fields are read only when contrastive guidance is in play."""
    spec = getattr(definition, "distillation", None)
    if not isinstance(spec, DistillationSpec):  # no block (or a stand-in object without one)
        spec = None
    def_id = getattr(definition, "id", "?")
    handling = config.get("distillation_handling")
    if spec is None:
        if handling not in (None, "", "none"):
            raise ValueError(
                f"{def_id}: distillation_handling={handling!r} is not supported; this model is not "
                "distilled, so the only handling is none"
            )
        return ResolvedDistillation("none", None, None)
    if handling in (None, ""):
        handling = spec.default_handling
    allowed = spec.allowed_handlings()
    if handling not in allowed:
        raise ValueError(f"{def_id}: distillation_handling={handling!r} is not supported; use one of {allowed}")
    if handling not in ("contrastive_guidance", "both"):
        return ResolvedDistillation(handling, None, None)

    scale = config.get("distillation_guidance_scale")
    if scale in (None, ""):
        scale = spec.default_guidance_scale(handling)
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        raise ValueError(f"{def_id}: distillation_guidance_scale={scale!r} is not a number") from None
    if not math.isfinite(scale) or scale < 1.0:
        raise ValueError(
            f"{def_id}: distillation_guidance_scale={scale!r} must be >= 1 (1 trains exactly as without guidance)"
        )
    schedule = config.get("distillation_guidance_schedule")
    if schedule in (None, ""):
        schedule = DEFAULT_GUIDANCE_SCHEDULE
    if schedule not in GUIDANCE_SCHEDULES:
        raise ValueError(
            f"{def_id}: distillation_guidance_schedule={schedule!r} is not supported; use one of "
            f"{list(GUIDANCE_SCHEDULES)}"
        )
    return ResolvedDistillation(handling, scale, schedule)


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

    # Why this definition is not offered to the user, or None when it is.
    # ECOSYSTEM §6 (reserved 2026-08-31, LANE-45 / DECISION-26 (a)). A non-empty
    # value gates the definition out of every USER-FACING enumeration and is the
    # message the job seam refuses with; None (the default) means available, so
    # no existing definition changes meaning (ARCHITECTURE D2: additive).
    #
    # Deliberately ONE string rather than `available: bool` + a separate reason:
    # two fields can disagree, and a gate that cannot state why is exactly the
    # unsourced silencing CONVENTIONS forbids — you cannot hide a definition
    # here without writing down what is missing.
    #
    # INVARIANT: this gates the USER-FACING layer only. ModelRegistry keeps every
    # definition, gated or not, because the registry-wide coverage sweeps (a VRAM
    # entry per family, LoRA target lists, TE-loading contracts,
    # resolve_capabilities) enumerate them to catch a family that misses a
    # surface — those are the guards that make ungating safe later. Read it
    # through registry.is_definition_available() / list_available_models() /
    # available_definitions(), never by re-deriving the truthiness at a call site.
    unavailable_reason: str | None = Field(
        None,
        description=(
            "Why this definition is not offered to the user (None = available). "
            "Gates user-facing enumeration and job creation; never the registry."
        ),
    )

    # How the model was distilled, and which distillation handlings a job may choose
    # (see the module's "Distillation handling" note). None = not distilled: the job
    # trains exactly as before and the Training form shows nothing new.
    distillation: DistillationSpec | None = Field(
        None, description="Distillation block (None = not a distilled model)"
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
    capability_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Per-definition capability flags merged into resolve_capabilities' "
            "descriptor AFTER the family-level capability_overrides — for a "
            "flag that varies BETWEEN a family's own definitions (e.g. "
            "bernini_r's 1.3B single-expert vs 14B MoE dual_expert), which the "
            "family-level mapping alone cannot distinguish."
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
