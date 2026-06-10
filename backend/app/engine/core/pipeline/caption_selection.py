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


def select_training_caption(item: dict[str, Any], definition_id: str | None, use_general: bool) -> str:
    base = item.get("caption") or ""
    if use_general or not definition_id:
        return base
    try:
        ds_path = item.get("dataset_path")
        path = item.get("path")
        if ds_path and path:
            stem = os.path.splitext(os.path.basename(path))[0]
            variant = caption_variants.read_variant(ds_path, definition_id, stem)
            if variant:
                return variant
    except Exception:  # noqa: BLE001 — never break training over a caption variant
        logger.exception("variant_caption_resolution_failed")
    return base
