# backend/app/core/captioning/caption_variants.py
"""Per-definition caption variant storage + the variant-or-general resolver.

Layout mirrors the existing ``masked/{stem}.txt`` convention:
    {dataset_path}/captions/{definition_id}/{stem}.txt   # model-specific variant
    {dataset_path}/{stem}.txt                            # general (source/fallback)
"""

from __future__ import annotations

import os

_VARIANTS_SUBDIR = "captions"


def variant_dir(dataset_path: str, definition_id: str) -> str:
    return os.path.join(dataset_path, _VARIANTS_SUBDIR, definition_id)


def variant_path(dataset_path: str, definition_id: str, stem: str) -> str:
    return os.path.join(variant_dir(dataset_path, definition_id), f"{stem}.txt")


def has_variant(dataset_path: str, definition_id: str, stem: str) -> bool:
    return os.path.exists(variant_path(dataset_path, definition_id, stem))


def read_variant(dataset_path: str, definition_id: str, stem: str) -> str | None:
    path = variant_path(dataset_path, definition_id, stem)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_variant(dataset_path: str, definition_id: str, stem: str, text: str) -> None:
    path = variant_path(dataset_path, definition_id, stem)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _read_general(dataset_path: str, stem: str) -> str | None:
    for ext in (".txt", ".caption"):
        path = os.path.join(dataset_path, f"{stem}{ext}")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    return None


def resolve_caption(dataset_path: str, stem: str, definition_id: str | None) -> str:
    """Return the variant caption when present, else the general caption, else ''."""
    if definition_id:
        variant = read_variant(dataset_path, definition_id, stem)
        if variant is not None:
            return variant
    general = _read_general(dataset_path, stem)
    return general if general is not None else ""


def list_variant_definition_ids(dataset_path: str) -> list[str]:
    root = os.path.join(dataset_path, _VARIANTS_SUBDIR)
    if not os.path.isdir(root):
        return []
    return [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
