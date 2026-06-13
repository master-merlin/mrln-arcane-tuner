"""Dataset CRUD and management schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Open set: gates pair UI/validation, deliberately a plain string in the
# model so future kinds ("video", "mixed") only extend this set.
_ALLOWED_DATASET_KINDS = {"standard", "edit"}


def _validate_kind(v: str | None) -> str | None:
    if v is not None and v not in _ALLOWED_DATASET_KINDS:
        raise ValueError(
            f"Unknown dataset kind '{v}'. "
            f"Allowed: {sorted(_ALLOWED_DATASET_KINDS)}"
        )
    return v


class CreateDatasetRequest(BaseModel):
    """Request body for dataset creation."""
    name: str
    description: str = ""
    classifier: str = ""
    trigger_word: str = ""
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    kind: str = "standard"

    _check_kind = field_validator("kind")(_validate_kind)


class UpdateDatasetRequest(BaseModel):
    """Request body for dataset update."""
    name: str
    description: str
    classifier: str = ""
    trigger_word: str = ""
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    # None = leave unchanged, so older clients can't reset an edit
    # dataset back to standard by omitting the field.
    kind: str | None = None

    _check_kind = field_validator("kind")(_validate_kind)


class CaptionRequest(BaseModel):
    """Request body for caption save."""
    content: str


class ToggleEnabledRequest(BaseModel):
    """Request body for image enable/disable toggle."""
    enabled: bool


class ImportPathRequest(BaseModel):
    archive_path: str
    # Constrained so an unexpected value 422s instead of silently becoming a 409.
    on_conflict: Literal["rename", "overwrite"] | None = None
    new_name: str | None = None


# ── Response models ──────────────────────────────────────────────────────
# Replace the raw-dict returns the CRUD routes used to emit, per the
# API_CONVENTIONS "every response is a Pydantic model" rule. Field sets mirror
# exactly what the handlers returned so wiring ``response_model=`` is a pure
# typing change with no payload drift.


class DatasetDeletedResponse(BaseModel):
    """Ack for unregistering a dataset."""
    status: str = "deleted"
    name: str


class MediaPairDeletedResponse(BaseModel):
    """Ack for deleting a media file + its caption sidecar."""
    status: str = "deleted"
    file: str


class UploadResponse(BaseModel):
    """Ack for a single-file upload into a dataset."""
    filename: str
    status: str = "uploaded"


class CaptionContentResponse(BaseModel):
    """A caption file's contents."""
    content: str


class CaptionSavedResponse(BaseModel):
    """Ack for a caption save."""
    status: str = "saved"


class ToggleEnabledResponse(BaseModel):
    """New enabled state for a single image."""
    media_file: str
    enabled: bool


class EnableAllResponse(BaseModel):
    """Count of images flipped back to enabled."""
    reset_count: int


class PairOrderRequest(BaseModel):
    """Set (or clear with null) one pair group's logical role order."""
    role_order: list[str] | None = None


class PairOrderResponse(BaseModel):
    """New role order for a single pair group."""
    media_file: str
    role_order: list[str] | None = None


class PairOrderApplyAllRequest(BaseModel):
    """Dataset-wide role order (the BACKWARD flip)."""
    role_order: list[str] = Field(min_length=1)


class PairOrderApplyAllResponse(BaseModel):
    """Counts for a dataset-wide role-order application."""
    applied: int
    skipped: int


class OrphansDeletedResponse(BaseModel):
    """Count of orphaned control files removed."""
    deleted: int


class OrphanControl(BaseModel):
    """A control file whose stem has no target image."""
    slot: str
    rel_path: str


class PairWarning(BaseModel):
    """Per-stem pair-health warning."""
    stem: str
    type: Literal[
        "dim_mismatch", "target_edited_after_control", "role_order_invalid",
    ]


class PairHealthResponse(BaseModel):
    """Pair-health report for an edit dataset (all findings are warnings)."""
    kind: str
    target_count: int
    paired_count: int
    fully_paired: bool
    active_slots: list[str] = Field(default_factory=list)
    missing_by_slot: dict[str, list[str]] = Field(default_factory=dict)
    orphans: list[OrphanControl] = Field(default_factory=list)
    warnings: list[PairWarning] = Field(default_factory=list)


class DatasetPairResponse(BaseModel):
    """One image/caption pair row. Canonical contract mirrored by the frontend
    ``DatasetPair`` interface (services/dataset.ts). Results are filtered to rows
    that have a media file, so ``media_file``/``media_type`` are always set."""
    stem: str
    media_file: str
    media_type: Literal["image", "video"]
    # null (not absent) when the image has no caption sidecar.
    caption_file: str | None = None
    # Present only for media rows (which is all returned rows).
    size_bytes: int | None = None
    caption_content: str = ""
    masked_caption_content: str | None = None
    # Free-form per-item media_metadata; shape varies by enrichment state.
    metadata: dict | None = None

    # ── Paired edit datasets ─────────────────────────────────────────
    # Physical control slot rel-paths in slot order (control/, control_2/,
    # control_3/); empty for standard datasets.
    control_files: list[str] = Field(default_factory=list)
    # Per-slot dims + role_order + target_edited_at (mirrors mask_info).
    control_info: dict | None = None
    # Logical ordering: permutation of physical slot names ("root" +
    # control dirs), position 0 = training target. None = default order.
    role_order: list[str] | None = None
    # Resolved roles — what training and the grid consume. The caption
    # stays keyed to the stem regardless of ordering.
    effective_target: str = ""
    effective_controls: list[str] = Field(default_factory=list)
