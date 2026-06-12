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
