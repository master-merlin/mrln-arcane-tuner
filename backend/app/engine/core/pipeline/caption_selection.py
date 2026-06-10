"""Select the caption text to train on for one image: a per-definition variant
overrides the general caption, unless the run opts out via ``use_general``.

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
    item: dict[str, Any], definition_id: str | None, use_general: bool, masked: bool = False
) -> str:
    """Resolve the caption to train on for one image.

    A per-definition variant overrides the base caption unless ``use_general``.
    Original axis (masked=False): base = item["caption"].
    Masked axis (masked=True):    base = item["masked_caption"] or item["caption"]
        — i.e. a missing/empty masked caption falls back to the original caption.
        To restore the prior subject-only behavior (empty when no masked caption),
        change the masked base to ``item.get("masked_caption") or ""``.

    Defensive: ANY problem resolving a variant falls back to ``base`` so it can
    never break a training run.
    """
    if masked:
        base = item.get("masked_caption") or item.get("caption") or ""
    else:
        base = item.get("caption") or ""
    if use_general or not definition_id:
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
