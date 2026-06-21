"""Select the caption text to train on for one image: a per-definition variant
overrides the general caption, controlled by per-dataset flags on the item.

This is intentionally defensive — ANY problem resolving a variant falls back to
the general caption so it can never break a training run.
"""

from __future__ import annotations

import os
from typing import Any

from app.core.captioning import caption_variants
from app.core.logger import get_logger

logger = get_logger(__name__)


def select_training_caption(
    item: dict[str, Any], definition_id: str | None, masked: bool = False
) -> str:
    """Resolve the caption to train on for one image.

    Per-dataset flags ride on the item (attached in _prepare_data):
    - ``use_captions`` False → return '' (no file caption, no variant lookup).
      Trigger word / caption prefix are assembled later in _get_batch, so this
      is the trigger-word-only training mode.
    - ``use_model_aware_captions`` False → skip the per-definition variant
      lookup; base-caption fallback chain unchanged.
    Missing flags default to True/True (old job configs behave like today).

    A per-definition variant overrides the base caption.
    Original axis (masked=False): base = item["caption"].
    Masked axis (masked=True):    base = item["masked_caption"] or item["caption"]
        — i.e. a missing/empty masked caption falls back to the original caption.
        To restore the prior subject-only behavior (empty when no masked caption),
        change the masked base to ``item.get("masked_caption") or ""``.

    Defensive: ANY problem resolving a variant falls back to ``base`` so it can
    never break a training run.
    """
    if not item.get("use_captions", True):
        return ""
    if masked:
        base = item.get("masked_caption") or item.get("caption") or ""
    else:
        base = item.get("caption") or ""
    if not item.get("use_model_aware_captions", True) or not definition_id:
        return base
    try:
        ds_path = item.get("dataset_path")
        path = item.get("path")
        if ds_path and path:
            stem = os.path.splitext(os.path.basename(path))[0]
            variant = caption_variants.read_variant(ds_path, definition_id, stem, masked)
            if variant:
                return variant
    except Exception:  # noqa: BLE001 — never break training over a caption variant
        logger.exception("variant_caption_resolution_failed")
    return base


def summarize_caption_sources(
    items: list[dict[str, Any]], definition_id: str | None
) -> list[dict[str, Any]]:
    """Read-only audit of which caption SOURCE each item trains on, by dataset.

    Mirrors ``select_training_caption``'s decision on the original axis so a
    training log can PROVE whether the per-definition *model-aware* variant was
    actually consumed for this run (vs. the generic caption). Pure / read-only;
    never raises.

    One entry per ``dataset_path`` with::

        dataset_path, definition_id, model_aware, use_captions,
        total, variant, base, empty, example_stem, example_is_json
    """
    groups: dict[str, dict[str, Any]] = {}
    for item in items:
        ds = item.get("dataset_path") or "?"
        g = groups.get(ds)
        if g is None:
            g = groups[ds] = {
                "dataset_path": ds,
                "definition_id": definition_id,
                "model_aware": bool(item.get("use_model_aware_captions", True)),
                "use_captions": bool(item.get("use_captions", True)),
                "total": 0,
                "variant": 0,
                "base": 0,
                "empty": 0,
                "example_stem": None,
                "example_is_json": False,
            }
        g["total"] += 1

        try:
            resolved = select_training_caption(item, definition_id)
        except Exception:  # noqa: BLE001
            resolved = ""

        # Did the resolved caption come from a per-definition variant? Mirror the
        # selector's gate exactly (use_captions + model_aware + def_id + a
        # non-empty variant file), so the audit can't disagree with training.
        is_variant = False
        if (
            item.get("use_captions", True)
            and item.get("use_model_aware_captions", True)
            and definition_id
        ):
            path = item.get("path")
            if ds != "?" and path:
                stem = os.path.splitext(os.path.basename(path))[0]
                try:
                    is_variant = bool(
                        caption_variants.read_variant(ds, definition_id, stem)
                    )
                except Exception:  # noqa: BLE001
                    is_variant = False

        if not resolved:
            g["empty"] += 1
        elif is_variant:
            g["variant"] += 1
            if g["example_stem"] is None:
                g["example_stem"] = os.path.splitext(
                    os.path.basename(item.get("path", ""))
                )[0]
                g["example_is_json"] = resolved.lstrip().startswith("{")
        else:
            g["base"] += 1

    return list(groups.values())
