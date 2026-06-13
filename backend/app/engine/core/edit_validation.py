"""Run-config validation for paired-image (edit/kontext) training.

Pure logic — no disk or registry access. The caller supplies a
``kind_of(dataset_name) -> str | None`` lookup (backed by the dataset
DB) so the same function validates at config-save time (route layer)
and at run start (data pipeline). Errors block the run; warnings are
advisory (logged, surfaced in the run log).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class EditConfigReport:
    """Outcome of an edit-config validation pass."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing blocks the run (warnings don't block)."""
        return not self.errors


def validate_edit_config(
    definition,
    config: dict[str, Any],
    kind_of: Callable[[str], str | None],
) -> EditConfigReport:
    """Validate a training config against the selected definition's edit mode.

    Edit model (``control_inputs > 0``):
      - every attached dataset must be ``kind == "edit"`` (paired controls)
      - flip augmentation is forbidden (breaks control/target correspondence)
      - masked variants are forbidden (mutually exclusive with edit training)

    Standard model + an edit dataset: allowed, but its control images are
    ignored — emit a warning so the mismatch is visible.
    """
    report = EditConfigReport()
    control_inputs = int(getattr(definition, "control_inputs", 0) or 0)
    is_edit_model = control_inputs > 0
    datasets = config.get("datasets") or []

    if is_edit_model:
        if config.get("h_flip") or config.get("v_flip"):
            report.errors.append(
                "Edit models cannot use horizontal/vertical flip augmentation — "
                "it breaks control/target pixel correspondence. Disable h_flip "
                "and v_flip."
            )
        for ds in datasets:
            name = ds.get("dataset_name", "")
            if ds.get("masking_enabled"):
                report.errors.append(
                    f"Dataset '{name}': masked training is mutually exclusive "
                    "with paired edit training. Disable masking for this dataset."
                )
            kind = kind_of(name)
            if kind != "edit":
                report.errors.append(
                    f"Dataset '{name}' is kind='{kind or 'standard'}' but the "
                    "selected model is an edit model and requires an edit "
                    "dataset with control images."
                )
    else:
        for ds in datasets:
            name = ds.get("dataset_name", "")
            if kind_of(name) == "edit":
                report.warnings.append(
                    f"Dataset '{name}' is an edit (paired) dataset but the "
                    "selected model is not an edit model — control images will "
                    "be ignored."
                )

    return report
